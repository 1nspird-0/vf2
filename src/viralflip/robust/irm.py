"""Invariant Risk Minimization (IRM) for shift-robust learning.

IRM learns features that remain predictive across "environments" (different
distributions). In mobile health, environments are:
- High/low mobility regimes
- Weekday/weekend patterns
- School vs break periods
- High/low mic uptime

This is stronger than linear confound projection because it enforces that
the illness prediction holds *invariantly* across these regimes.

Reference: Invariant Risk Minimization (Arjovsky et al., 2019)
https://arxiv.org/abs/1907.02893
"""

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EnvironmentDefinition:
    """Definition of an environment for IRM."""
    
    name: str
    # Indices of samples belonging to this environment
    indices: np.ndarray
    # Optional weight for this environment
    weight: float = 1.0


class IRMPenalty(nn.Module):
    """Invariant Risk Minimization penalty.
    
    The IRM penalty encourages the classifier to be optimal simultaneously
    across all environments. For each environment e:
    
    penalty_e = ||grad_w (w * R_e(Phi))||^2
    
    where R_e is the risk in environment e, Phi is the representation,
    and w is a dummy classifier (initialized to 1).
    
    The gradient of the risk w.r.t. w should be zero if the representation
    is invariant across environments.
    """
    
    def __init__(self, penalty_weight: float = 1.0, anneal_iters: int = 500):
        """Initialize IRM penalty.
        
        Args:
            penalty_weight: Weight for IRM penalty (lambda_irm)
            anneal_iters: Iterations over which to anneal penalty weight
        """
        super().__init__()
        self.penalty_weight = penalty_weight
        self.anneal_iters = anneal_iters
        self.current_iter = 0
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        env_masks: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute IRM penalty.
        
        Args:
            logits: Model logits, shape (batch, n_classes) or (batch,)
            targets: Target labels, shape (batch,) or (batch, n_classes)
            env_masks: Dict mapping environment name to boolean mask
                      Each mask: (batch,) True = belongs to this environment
                      
        Returns:
            IRM penalty (scalar)
        """
        device = logits.device
        
        # Handle binary case
        if logits.dim() == 1:
            logits = logits.unsqueeze(-1)
        
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1)
        
        penalty = torch.tensor(0.0, device=device)
        n_envs = 0
        
        for env_name, env_mask in env_masks.items():
            if env_mask.sum() == 0:
                continue
            
            # Get environment-specific data
            env_logits = logits[env_mask]
            env_targets = targets[env_mask]
            
            # Compute gradient penalty for this environment
            env_penalty = self._compute_env_penalty(env_logits, env_targets)
            penalty = penalty + env_penalty
            n_envs += 1
        
        if n_envs > 0:
            penalty = penalty / n_envs
        
        # Anneal penalty weight
        effective_weight = self._get_annealed_weight()
        self.current_iter += 1
        
        return effective_weight * penalty
    
    def _compute_env_penalty(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute IRM penalty for a single environment.
        
        Uses the gradient penalty formulation:
        penalty = ||grad_w (w * BCE)||^2
        
        where w is a dummy scalar initialized to 1.
        """
        # Dummy classifier weight
        dummy_w = torch.tensor(1.0, device=logits.device, requires_grad=True)
        
        # Scale logits by dummy weight
        scaled_logits = logits * dummy_w
        
        # Compute loss
        loss = F.binary_cross_entropy_with_logits(
            scaled_logits, targets.float(), reduction='mean'
        )
        
        # Compute gradient w.r.t. dummy weight
        grad = torch.autograd.grad(
            loss, dummy_w, create_graph=True, retain_graph=True
        )[0]
        
        # Penalty is squared gradient norm
        penalty = grad ** 2
        
        return penalty
    
    def _get_annealed_weight(self) -> float:
        """Get annealed penalty weight."""
        if self.current_iter >= self.anneal_iters:
            return self.penalty_weight
        
        # Linear annealing from 0 to penalty_weight
        return self.penalty_weight * (self.current_iter / self.anneal_iters)
    
    def reset_annealing(self) -> None:
        """Reset annealing counter."""
        self.current_iter = 0


class IRMLoss(nn.Module):
    """Combined loss with IRM penalty.
    
    L_total = L_erm + lambda_irm * L_irm
    
    where L_erm is the standard empirical risk and L_irm is the IRM penalty.
    """
    
    def __init__(
        self,
        base_loss: nn.Module,
        irm_penalty_weight: float = 1.0,
        anneal_iters: int = 500,
        use_v1: bool = False,
    ):
        """Initialize IRM loss.
        
        Args:
            base_loss: Base loss function (e.g., BCE, focal loss)
            irm_penalty_weight: Weight for IRM penalty
            anneal_iters: Iterations for penalty annealing
            use_v1: Use IRMv1 (gradient penalty) vs IRM (direct optimization)
        """
        super().__init__()
        
        self.base_loss = base_loss
        self.irm_penalty = IRMPenalty(irm_penalty_weight, anneal_iters)
        self.use_v1 = use_v1
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        env_masks: dict[str, torch.Tensor],
        reg_penalty: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute total loss with IRM penalty.
        
        Args:
            logits: Model logits
            targets: Target labels
            env_masks: Dict mapping environment name to boolean mask
            reg_penalty: Optional regularization penalty from model
            
        Returns:
            total_loss: Scalar loss
            loss_dict: Dict with loss components
        """
        # Base ERM loss
        probs = torch.sigmoid(logits)
        if hasattr(self.base_loss, 'forward'):
            erm_loss = self.base_loss(probs, targets)
            if isinstance(erm_loss, tuple):
                erm_loss = erm_loss[0]
        else:
            erm_loss = F.binary_cross_entropy_with_logits(logits, targets.float())
        
        # IRM penalty
        irm_loss = self.irm_penalty(logits, targets, env_masks)
        
        # Total loss
        total_loss = erm_loss + irm_loss
        
        if reg_penalty is not None:
            total_loss = total_loss + reg_penalty
        
        loss_dict = {
            "erm": erm_loss,
            "irm": irm_loss,
            "total": total_loss,
        }
        
        if reg_penalty is not None:
            loss_dict["reg"] = reg_penalty
        
        return total_loss, loss_dict


class EnvironmentClassifier(nn.Module):
    """Classify samples into environments based on behavior features.
    
    Uses behavior features (GPS, IMU, screen) to determine which
    environment a sample belongs to. This is used to create
    environment masks for IRM training.
    """
    
    def __init__(
        self,
        input_dim: int,
        n_environments: int = 4,
        hidden_dim: int = 64,
    ):
        """Initialize environment classifier.
        
        Args:
            input_dim: Input feature dimension
            n_environments: Number of discrete environments
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.n_environments = n_environments
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_environments),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify into environments.
        
        Args:
            x: Behavior features, shape (batch, input_dim)
            
        Returns:
            Environment logits, shape (batch, n_environments)
        """
        return self.net(x)
    
    def get_env_masks(
        self,
        x: torch.Tensor,
        hard: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Get environment masks for samples.
        
        Args:
            x: Behavior features, shape (batch, input_dim)
            hard: If True, use hard assignment; else soft probabilities
            
        Returns:
            Dict mapping environment name to mask
        """
        logits = self.forward(x)
        
        if hard:
            env_idx = logits.argmax(dim=-1)
            masks = {
                f"env_{i}": (env_idx == i)
                for i in range(self.n_environments)
            }
        else:
            probs = F.softmax(logits, dim=-1)
            masks = {
                f"env_{i}": probs[:, i]
                for i in range(self.n_environments)
            }
        
        return masks


class BehaviorEnvironmentDetector:
    """Detect environments from behavior patterns.
    
    Defines environments based on:
    - Mobility level (high/low based on GPS entropy)
    - Time patterns (weekday/weekend)
    - Data availability (high/low sensor uptime)
    - Activity level (sedentary/active)
    
    These are used as "environments" for IRM to ensure illness
    predictions are invariant to behavioral context.
    """
    
    # Default environment definitions
    ENVIRONMENTS = [
        "high_mobility",
        "low_mobility",
        "weekday",
        "weekend",
        "high_data_quality",
        "low_data_quality",
        "sedentary",
        "active",
    ]
    
    def __init__(
        self,
        mobility_threshold: float = 0.5,
        quality_threshold: float = 0.7,
        activity_threshold: float = 0.3,
        use_temporal: bool = True,
    ):
        """Initialize detector.
        
        Args:
            mobility_threshold: Threshold for high/low mobility (0-1)
            quality_threshold: Threshold for high/low quality
            activity_threshold: Threshold for sedentary/active
            use_temporal: Whether to use weekday/weekend environments
        """
        self.mobility_threshold = mobility_threshold
        self.quality_threshold = quality_threshold
        self.activity_threshold = activity_threshold
        self.use_temporal = use_temporal
    
    def detect_environments(
        self,
        gps_entropy: np.ndarray,
        quality_scores: np.ndarray,
        activity_levels: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
    ) -> dict[str, np.ndarray]:
        """Detect environment assignments.
        
        Args:
            gps_entropy: GPS location entropy, shape (n_samples,)
            quality_scores: Mean quality scores, shape (n_samples,)
            activity_levels: Activity levels (0-1), shape (n_samples,)
            timestamps: Optional Unix timestamps for weekday/weekend
            
        Returns:
            Dict mapping environment name to boolean mask
        """
        n_samples = len(gps_entropy)
        
        # Normalize to [0, 1] if needed
        gps_norm = self._normalize(gps_entropy)
        quality_norm = self._normalize(quality_scores)
        activity_norm = self._normalize(activity_levels)
        
        environments = {}
        
        # Mobility environments
        environments["high_mobility"] = gps_norm >= self.mobility_threshold
        environments["low_mobility"] = gps_norm < self.mobility_threshold
        
        # Quality environments
        environments["high_data_quality"] = quality_norm >= self.quality_threshold
        environments["low_data_quality"] = quality_norm < self.quality_threshold
        
        # Activity environments
        environments["sedentary"] = activity_norm < self.activity_threshold
        environments["active"] = activity_norm >= self.activity_threshold
        
        # Temporal environments
        if self.use_temporal and timestamps is not None:
            import datetime
            
            is_weekend = np.array([
                datetime.datetime.fromtimestamp(ts).weekday() >= 5
                for ts in timestamps
            ])
            
            environments["weekend"] = is_weekend
            environments["weekday"] = ~is_weekend
        
        return environments
    
    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize to [0, 1]."""
        x_min, x_max = x.min(), x.max()
        if x_max - x_min < 1e-8:
            return np.zeros_like(x)
        return (x - x_min) / (x_max - x_min)
    
    def to_torch_masks(
        self,
        environments: dict[str, np.ndarray],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Convert numpy masks to torch tensors."""
        return {
            k: torch.from_numpy(v).bool().to(device)
            for k, v in environments.items()
        }
    
    def select_diverse_environments(
        self,
        environments: dict[str, np.ndarray],
        n_select: int = 4,
    ) -> dict[str, np.ndarray]:
        """Select diverse environments for IRM.
        
        Selects environments that provide good coverage and are
        not too correlated with each other.
        
        Args:
            environments: All detected environments
            n_select: Number of environments to select
            
        Returns:
            Selected environments
        """
        # Prioritize these environment pairs (diverse coverage)
        priority_pairs = [
            ("high_mobility", "low_mobility"),
            ("weekday", "weekend"),
            ("high_data_quality", "low_data_quality"),
            ("sedentary", "active"),
        ]
        
        selected = {}
        
        for pair in priority_pairs:
            if len(selected) >= n_select:
                break
            
            for env in pair:
                if env in environments and len(selected) < n_select:
                    # Check this environment has enough samples
                    if environments[env].sum() >= 10:
                        selected[env] = environments[env]
        
        return selected


class VRExPenalty(nn.Module):
    """Variance Risk Extrapolation (VREx) penalty.
    
    An alternative to IRM that penalizes variance of risks across environments.
    Often more stable than IRM in practice.
    
    VREx penalty = Var_e[R_e]
    
    where R_e is the risk in environment e.
    """
    
    def __init__(self, penalty_weight: float = 1.0, anneal_iters: int = 500):
        """Initialize VREx penalty.
        
        Args:
            penalty_weight: Weight for VREx penalty
            anneal_iters: Iterations for annealing
        """
        super().__init__()
        self.penalty_weight = penalty_weight
        self.anneal_iters = anneal_iters
        self.current_iter = 0
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        env_masks: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute VREx penalty.
        
        Args:
            logits: Model logits
            targets: Target labels
            env_masks: Environment masks
            
        Returns:
            VREx penalty (scalar)
        """
        device = logits.device
        env_losses = []
        
        for env_name, env_mask in env_masks.items():
            if env_mask.sum() == 0:
                continue
            
            env_logits = logits[env_mask]
            env_targets = targets[env_mask]
            
            env_loss = F.binary_cross_entropy_with_logits(
                env_logits, env_targets.float(), reduction='mean'
            )
            env_losses.append(env_loss)
        
        if len(env_losses) < 2:
            return torch.tensor(0.0, device=device)
        
        # Compute variance of losses
        env_losses = torch.stack(env_losses)
        penalty = env_losses.var()
        
        # Anneal weight
        effective_weight = self._get_annealed_weight()
        self.current_iter += 1
        
        return effective_weight * penalty
    
    def _get_annealed_weight(self) -> float:
        """Get annealed penalty weight."""
        if self.current_iter >= self.anneal_iters:
            return self.penalty_weight
        return self.penalty_weight * (self.current_iter / self.anneal_iters)

