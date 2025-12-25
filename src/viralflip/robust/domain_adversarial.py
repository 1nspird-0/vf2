"""Domain Adversarial Neural Networks (DANN) for domain-invariant learning.

Complements IRM by learning representations that are invariant to
domain/environment through adversarial training.

The model learns representations that:
1. Are predictive of illness
2. Are NOT predictive of the environment/domain

This is achieved by a gradient reversal layer that makes the feature
extractor adversarial to a domain discriminator.

Reference: Domain-Adversarial Training of Neural Networks (Ganin et al., 2016)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GradientReversalFunction(Function):
    """Gradient Reversal Layer function.
    
    In forward pass: identity
    In backward pass: negate gradients and scale by lambda
    """
    
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    """Gradient Reversal Layer for domain adversarial training.
    
    Passes input unchanged during forward pass but reverses
    gradients during backward pass.
    """
    
    def __init__(self, lambda_: float = 1.0):
        """Initialize GRL.
        
        Args:
            lambda_: Scaling factor for reversed gradients
        """
        super().__init__()
        self.lambda_ = lambda_
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)
    
    def set_lambda(self, lambda_: float) -> None:
        """Update lambda value."""
        self.lambda_ = lambda_


class DomainDiscriminator(nn.Module):
    """Discriminator that tries to predict domain/environment from features.
    
    Used with GRL to make representations domain-invariant.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        n_domains: int = 2,
        dropout: float = 0.1,
    ):
        """Initialize discriminator.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            n_domains: Number of domains/environments
            dropout: Dropout rate
        """
        super().__init__()
        
        self.n_domains = n_domains
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_domains if n_domains > 2 else 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict domain from features.
        
        Args:
            x: Input features, shape (batch, input_dim)
            
        Returns:
            Domain logits, shape (batch, n_domains) or (batch,) for binary
        """
        return self.net(x)


class DomainAdversarialLoss(nn.Module):
    """Domain adversarial loss for invariant representation learning.
    
    Combines:
    - Task loss (illness prediction)
    - Domain loss (environment prediction through GRL)
    
    The GRL ensures the feature extractor learns representations that
    confuse the domain discriminator.
    """
    
    def __init__(
        self,
        task_loss: nn.Module,
        domain_weight: float = 0.1,
        n_domains: int = 2,
        hidden_dim: int = 128,
        anneal_iters: int = 500,
    ):
        """Initialize DANN loss.
        
        Args:
            task_loss: Loss function for main task
            domain_weight: Weight for domain adversarial loss
            n_domains: Number of domains
            hidden_dim: Hidden dim for discriminator
            anneal_iters: Iterations for lambda annealing
        """
        super().__init__()
        
        self.task_loss = task_loss
        self.domain_weight = domain_weight
        self.n_domains = n_domains
        self.anneal_iters = anneal_iters
        self.current_iter = 0
        
        # Domain discriminator (will be set when feature_dim is known)
        self.discriminator = None
        self.grl = GradientReversalLayer(lambda_=0.0)
        self._hidden_dim = hidden_dim
    
    def init_discriminator(self, feature_dim: int) -> None:
        """Initialize discriminator with known feature dimension."""
        self.discriminator = DomainDiscriminator(
            input_dim=feature_dim,
            hidden_dim=self._hidden_dim,
            n_domains=self.n_domains,
        )
    
    def forward(
        self,
        task_logits: torch.Tensor,
        task_targets: torch.Tensor,
        features: torch.Tensor,
        domain_labels: torch.Tensor,
        reg_penalty: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute combined task and domain adversarial loss.
        
        Args:
            task_logits: Logits for main task
            task_targets: Labels for main task
            features: Intermediate features for domain prediction
            domain_labels: Domain/environment labels
            reg_penalty: Optional regularization penalty
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dict with loss components
        """
        # Initialize discriminator if needed
        if self.discriminator is None:
            self.init_discriminator(features.shape[-1])
            self.discriminator = self.discriminator.to(features.device)
        
        # Task loss
        task_probs = torch.sigmoid(task_logits)
        if hasattr(self.task_loss, 'forward'):
            task_loss = self.task_loss(task_probs, task_targets)
            if isinstance(task_loss, tuple):
                task_loss = task_loss[0]
        else:
            task_loss = F.binary_cross_entropy_with_logits(
                task_logits, task_targets.float()
            )
        
        # Update GRL lambda with annealing
        lambda_ = self._get_annealed_lambda()
        self.grl.set_lambda(lambda_)
        self.current_iter += 1
        
        # Domain loss through GRL
        reversed_features = self.grl(features)
        domain_logits = self.discriminator(reversed_features)
        
        if self.n_domains == 2:
            domain_loss = F.binary_cross_entropy_with_logits(
                domain_logits.squeeze(), domain_labels.float()
            )
        else:
            domain_loss = F.cross_entropy(domain_logits, domain_labels)
        
        # Combine losses
        total_loss = task_loss + self.domain_weight * domain_loss
        
        if reg_penalty is not None:
            total_loss = total_loss + reg_penalty
        
        loss_dict = {
            "task": task_loss,
            "domain": domain_loss,
            "total": total_loss,
            "lambda": torch.tensor(lambda_),
        }
        
        if reg_penalty is not None:
            loss_dict["reg"] = reg_penalty
        
        return total_loss, loss_dict
    
    def _get_annealed_lambda(self) -> float:
        """Get annealed lambda for GRL.
        
        Uses sigmoid schedule:
        lambda = 2 / (1 + exp(-10 * p)) - 1
        where p = iter / anneal_iters
        """
        import math
        
        if self.current_iter >= self.anneal_iters:
            p = 1.0
        else:
            p = self.current_iter / self.anneal_iters
        
        lambda_ = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0
        return lambda_ * self.domain_weight


class MultiDomainBatchSampler:
    """Batch sampler that ensures each batch has samples from all domains.
    
    This helps with stable training in domain adversarial setting.
    """
    
    def __init__(
        self,
        domain_labels: torch.Tensor,
        batch_size: int,
        drop_last: bool = True,
    ):
        """Initialize sampler.
        
        Args:
            domain_labels: Domain labels for all samples
            batch_size: Batch size
            drop_last: Whether to drop last incomplete batch
        """
        self.domain_labels = domain_labels
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # Group indices by domain
        self.domain_indices = {}
        unique_domains = domain_labels.unique()
        
        for d in unique_domains:
            self.domain_indices[d.item()] = (domain_labels == d).nonzero().squeeze(-1)
        
        self.n_domains = len(self.domain_indices)
        self.samples_per_domain = batch_size // self.n_domains
    
    def __iter__(self):
        """Generate batches with samples from each domain."""
        # Shuffle indices within each domain
        shuffled = {
            d: idx[torch.randperm(len(idx))]
            for d, idx in self.domain_indices.items()
        }
        
        # Find minimum domain size
        min_size = min(len(idx) for idx in shuffled.values())
        n_batches = min_size // self.samples_per_domain
        
        if self.drop_last and n_batches == 0:
            return
        
        for b in range(n_batches):
            batch = []
            start = b * self.samples_per_domain
            end = start + self.samples_per_domain
            
            for d, idx in shuffled.items():
                batch.extend(idx[start:end].tolist())
            
            yield batch
    
    def __len__(self) -> int:
        """Number of batches."""
        min_size = min(len(idx) for idx in self.domain_indices.values())
        return min_size // self.samples_per_domain


class ConditionalDomainAdversarial(nn.Module):
    """Conditional Domain Adversarial Network (CDAN).
    
    Extends DANN by conditioning domain discrimination on task predictions.
    This allows learning class-wise domain-invariant features.
    """
    
    def __init__(
        self,
        feature_dim: int,
        n_classes: int,
        n_domains: int = 2,
        hidden_dim: int = 128,
        domain_weight: float = 0.1,
    ):
        """Initialize CDAN.
        
        Args:
            feature_dim: Feature dimension
            n_classes: Number of task classes
            n_domains: Number of domains
            hidden_dim: Hidden dimension
            domain_weight: Domain loss weight
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.n_classes = n_classes
        self.n_domains = n_domains
        self.domain_weight = domain_weight
        
        self.grl = GradientReversalLayer()
        
        # Discriminator on joint feature-prediction space
        joint_dim = feature_dim * n_classes
        self.discriminator = DomainDiscriminator(
            input_dim=joint_dim,
            hidden_dim=hidden_dim,
            n_domains=n_domains,
        )
    
    def forward(
        self,
        features: torch.Tensor,
        predictions: torch.Tensor,
        domain_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute conditional domain adversarial loss.
        
        Args:
            features: Features, shape (batch, feature_dim)
            predictions: Softmax predictions, shape (batch, n_classes)
            domain_labels: Domain labels, shape (batch,)
            
        Returns:
            Domain adversarial loss
        """
        # Outer product of features and predictions
        # (batch, feature_dim, 1) x (batch, 1, n_classes) -> (batch, feature_dim, n_classes)
        joint = torch.bmm(
            features.unsqueeze(2),
            predictions.unsqueeze(1)
        )
        joint = joint.view(features.shape[0], -1)  # (batch, feature_dim * n_classes)
        
        # Apply GRL and discriminator
        reversed_joint = self.grl(joint)
        domain_logits = self.discriminator(reversed_joint)
        
        if self.n_domains == 2:
            loss = F.binary_cross_entropy_with_logits(
                domain_logits.squeeze(), domain_labels.float()
            )
        else:
            loss = F.cross_entropy(domain_logits, domain_labels)
        
        return self.domain_weight * loss

