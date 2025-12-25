"""Differential privacy utilities for federated learning.

Provides privacy guarantees for model updates by:
- Gradient clipping
- Noise addition
- Privacy budget tracking
"""

from dataclasses import dataclass
from typing import Optional
import math

import numpy as np
import torch
import torch.nn as nn


@dataclass
class DPConfig:
    """Differential privacy configuration."""
    
    # Privacy parameters
    epsilon: float = 1.0  # Privacy budget
    delta: float = 1e-5   # Failure probability
    
    # Mechanism parameters
    max_grad_norm: float = 1.0  # Gradient clipping bound
    noise_multiplier: float = 1.0  # Noise scale
    
    # Accounting
    target_epsilon: float = 8.0  # Total privacy budget
    accounting_method: str = "rdp"  # 'rdp' or 'gdp'


def clip_gradients(
    model: nn.Module,
    max_norm: float,
    per_sample: bool = False,
) -> float:
    """Clip gradients to bounded norm.
    
    Args:
        model: Model with computed gradients
        max_norm: Maximum gradient norm
        per_sample: If True, clip per-sample gradients (requires special setup)
        
    Returns:
        Total gradient norm before clipping
    """
    if per_sample:
        # Per-sample gradient clipping (for DP-SGD)
        # Requires model to store per-sample gradients
        return _clip_per_sample_gradients(model, max_norm)
    else:
        # Standard gradient clipping
        return torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm
        ).item()


def _clip_per_sample_gradients(
    model: nn.Module,
    max_norm: float,
) -> float:
    """Clip per-sample gradients for DP-SGD."""
    # Get per-sample gradients
    per_sample_grads = []
    for param in model.parameters():
        if hasattr(param, 'grad_sample'):
            per_sample_grads.append(param.grad_sample)
    
    if not per_sample_grads:
        # Fall back to standard clipping
        return clip_gradients(model, max_norm, per_sample=False)
    
    # Compute per-sample norms
    norms = []
    for grad in per_sample_grads:
        # grad: (batch, *param_shape)
        flat_grad = grad.reshape(grad.shape[0], -1)
        norm = torch.norm(flat_grad, dim=1)
        norms.append(norm)
    
    total_norms = torch.sqrt(sum(n ** 2 for n in norms))
    
    # Compute clipping factors
    clip_factors = torch.clamp(max_norm / (total_norms + 1e-6), max=1.0)
    
    # Apply clipping
    for param in model.parameters():
        if hasattr(param, 'grad_sample'):
            for i in range(param.grad_sample.shape[0]):
                param.grad_sample[i] *= clip_factors[i]
            
            # Average to get batch gradient
            param.grad = param.grad_sample.mean(dim=0)
    
    return total_norms.mean().item()


def add_noise(
    gradients: dict[str, torch.Tensor],
    noise_multiplier: float,
    max_grad_norm: float,
) -> dict[str, torch.Tensor]:
    """Add Gaussian noise to gradients for DP.
    
    Args:
        gradients: Dict of parameter gradients
        noise_multiplier: Noise scale (sigma = noise_multiplier * max_grad_norm)
        max_grad_norm: Clipping bound
        
    Returns:
        Noisy gradients
    """
    sigma = noise_multiplier * max_grad_norm
    
    noisy_grads = {}
    for name, grad in gradients.items():
        noise = torch.randn_like(grad) * sigma
        noisy_grads[name] = grad + noise
    
    return noisy_grads


def compute_privacy_budget(
    noise_multiplier: float,
    n_steps: int,
    batch_size: int,
    n_samples: int,
    delta: float = 1e-5,
) -> float:
    """Compute epsilon for given DP-SGD parameters using RDP accounting.
    
    Args:
        noise_multiplier: Noise multiplier sigma
        n_steps: Number of training steps
        batch_size: Batch size
        n_samples: Total number of samples
        delta: Target delta
        
    Returns:
        Epsilon (privacy budget spent)
    """
    # Sampling probability
    q = batch_size / n_samples
    
    # Use Renyi DP accounting
    orders = list(range(2, 64))
    rdp = compute_rdp(
        q=q,
        noise_multiplier=noise_multiplier,
        steps=n_steps,
        orders=orders,
    )
    
    # Convert to (epsilon, delta)-DP
    epsilon = rdp_to_epsilon(rdp, delta, orders)
    
    return epsilon


def compute_rdp(
    q: float,
    noise_multiplier: float,
    steps: int,
    orders: list[int],
) -> np.ndarray:
    """Compute Renyi Differential Privacy for Gaussian mechanism.
    
    Args:
        q: Sampling probability
        noise_multiplier: Noise multiplier
        steps: Number of composition steps
        orders: RDP orders to compute
        
    Returns:
        RDP values for each order
    """
    rdp = np.zeros(len(orders))
    
    for i, order in enumerate(orders):
        if noise_multiplier == 0:
            rdp[i] = float('inf')
        else:
            # RDP for subsampled Gaussian mechanism
            rdp[i] = compute_rdp_single_order(q, noise_multiplier, order)
    
    # Composition
    return rdp * steps


def compute_rdp_single_order(
    q: float,
    noise_multiplier: float,
    order: int,
) -> float:
    """Compute RDP for single order."""
    if order == 1:
        return 0
    
    if q == 0:
        return 0
    
    if q == 1:
        # Full batch: standard Gaussian mechanism
        return order / (2 * noise_multiplier ** 2)
    
    # Subsampled Gaussian mechanism (approximation)
    # Use log-space for numerical stability
    log_a = math.log1p(-q)
    log_b = math.log(q) + (order - 1) / (2 * noise_multiplier ** 2)
    
    return (1 / (order - 1)) * math.log(
        math.exp((order - 1) * log_a) + math.exp((order - 1) * log_b)
    )


def rdp_to_epsilon(
    rdp: np.ndarray,
    delta: float,
    orders: list[int],
) -> float:
    """Convert RDP to (epsilon, delta)-DP.
    
    Args:
        rdp: RDP values for each order
        delta: Target delta
        orders: RDP orders
        
    Returns:
        Minimum epsilon
    """
    epsilons = []
    
    for rdp_val, order in zip(rdp, orders):
        if rdp_val == float('inf'):
            epsilons.append(float('inf'))
        else:
            # eps = rdp - log(delta) / (order - 1)
            eps = rdp_val - math.log(delta) / (order - 1)
            epsilons.append(eps)
    
    return min(epsilons)


class PrivacyAccountant:
    """Track privacy budget across training."""
    
    def __init__(
        self,
        n_samples: int,
        batch_size: int,
        noise_multiplier: float,
        target_epsilon: float = 8.0,
        target_delta: float = 1e-5,
    ):
        """Initialize accountant.
        
        Args:
            n_samples: Total training samples
            batch_size: Batch size
            noise_multiplier: Noise multiplier
            target_epsilon: Target privacy budget
            target_delta: Target delta
        """
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.noise_multiplier = noise_multiplier
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        
        self.n_steps = 0
        self.epsilon_history = []
    
    def step(self) -> float:
        """Record a training step and return current epsilon."""
        self.n_steps += 1
        
        epsilon = compute_privacy_budget(
            noise_multiplier=self.noise_multiplier,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_samples=self.n_samples,
            delta=self.target_delta,
        )
        
        self.epsilon_history.append(epsilon)
        return epsilon
    
    def is_budget_exceeded(self) -> bool:
        """Check if privacy budget is exceeded."""
        if not self.epsilon_history:
            return False
        return self.epsilon_history[-1] >= self.target_epsilon
    
    def get_remaining_budget(self) -> float:
        """Get remaining privacy budget."""
        if not self.epsilon_history:
            return self.target_epsilon
        return max(0, self.target_epsilon - self.epsilon_history[-1])
    
    def get_max_steps(self) -> int:
        """Estimate maximum steps before budget exhaustion."""
        # Binary search for max steps
        low, high = self.n_steps, self.n_steps * 100
        
        while low < high:
            mid = (low + high + 1) // 2
            
            eps = compute_privacy_budget(
                noise_multiplier=self.noise_multiplier,
                n_steps=mid,
                batch_size=self.batch_size,
                n_samples=self.n_samples,
                delta=self.target_delta,
            )
            
            if eps <= self.target_epsilon:
                low = mid
            else:
                high = mid - 1
        
        return low

