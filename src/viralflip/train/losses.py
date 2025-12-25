"""Loss functions for multi-horizon illness prediction and virus classification.

Key considerations:
- Imbalanced classes (rare positive events)
- Multiple horizons with different weights
- Focal loss to focus on hard examples
- Multi-task learning: risk prediction + virus classification
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from viralflip.model.virus_types import NUM_VIRUS_CLASSES, VirusType


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Focuses learning on hard examples by down-weighting easy negatives.
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[float] = None,
        reduction: str = "mean",
    ):
        """Initialize focal loss.
        
        Args:
            gamma: Focusing parameter (higher = more focus on hard examples).
            alpha: Class weight for positive class. If None, no weighting.
            reduction: Reduction method ('mean', 'sum', 'none').
        """
        super().__init__()
        
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute focal loss.
        
        Args:
            inputs: Predicted probabilities, shape (batch, ...).
            targets: Binary targets, shape (batch, ...).
            weights: Optional sample weights, shape (batch, ...).
            
        Returns:
            Loss value.
        """
        # Clamp for numerical stability
        p = inputs.clamp(1e-7, 1 - 1e-7)
        
        # Binary cross entropy
        bce = -targets * torch.log(p) - (1 - targets) * torch.log(1 - p)
        
        # Focal weight
        p_t = targets * p + (1 - targets) * (1 - p)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weight
        if self.alpha is not None:
            alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
            focal_weight = focal_weight * alpha_t
        
        # Apply focal weighting
        loss = focal_weight * bce
        
        # Apply sample weights
        if weights is not None:
            loss = loss * weights
        
        # Reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class MultiHorizonLoss(nn.Module):
    """Multi-horizon loss combining losses across prediction horizons.
    
    L = Σ_H w_H * L_H(y_H, r_H)
    """
    
    def __init__(
        self,
        horizons: list[int],
        horizon_weights: Optional[list[float]] = None,
        use_focal: bool = True,
        focal_gamma: float = 2.0,
        pos_weight: float = 5.0,
    ):
        """Initialize multi-horizon loss.
        
        Args:
            horizons: List of prediction horizons.
            horizon_weights: Weights for each horizon. If None, equal weights.
            use_focal: Whether to use focal loss.
            focal_gamma: Focal loss gamma parameter.
            pos_weight: Weight for positive class.
        """
        super().__init__()
        
        self.horizons = horizons
        self.n_horizons = len(horizons)
        
        if horizon_weights is None:
            horizon_weights = [1.0] * self.n_horizons
        self.horizon_weights = torch.tensor(horizon_weights)
        
        self.use_focal = use_focal
        self.pos_weight = pos_weight
        
        if use_focal:
            # Alpha is derived from pos_weight
            alpha = pos_weight / (1 + pos_weight)
            self.loss_fn = FocalLoss(gamma=focal_gamma, alpha=alpha, reduction="none")
        else:
            self.loss_fn = None
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute multi-horizon loss.
        
        Args:
            predictions: Predicted probabilities, shape (batch, [seq,] n_horizons).
            targets: Binary targets, shape (batch, [seq,] n_horizons).
            mask: Optional mask for valid samples, shape (batch, [seq]).
            
        Returns:
            Tuple of (total_loss, loss_dict with per-horizon losses).
        """
        device = predictions.device
        self.horizon_weights = self.horizon_weights.to(device)
        
        # Handle sequence dimension
        if predictions.dim() == 3:
            # (batch, seq, n_horizons) -> flatten batch and seq
            batch, seq, n_h = predictions.shape
            predictions = predictions.view(-1, n_h)
            targets = targets.view(-1, n_h)
            if mask is not None:
                mask = mask.view(-1)
        
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=device)
        
        for i, horizon in enumerate(self.horizons):
            pred_h = predictions[:, i]
            target_h = targets[:, i]
            
            if self.use_focal:
                loss_h = self.loss_fn(pred_h, target_h)
            else:
                # Weighted BCE
                pos_weight = torch.tensor([self.pos_weight], device=device)
                loss_h = F.binary_cross_entropy(
                    pred_h, target_h, reduction="none"
                )
                # Apply pos_weight manually
                weight = target_h * self.pos_weight + (1 - target_h)
                loss_h = loss_h * weight
            
            # Apply mask
            if mask is not None:
                loss_h = loss_h * mask.float()
                loss_h = loss_h.sum() / (mask.sum() + 1e-6)
            else:
                loss_h = loss_h.mean()
            
            loss_dict[f"loss_{horizon}h"] = loss_h
            total_loss = total_loss + self.horizon_weights[i] * loss_h
        
        loss_dict["total"] = total_loss
        
        return total_loss, loss_dict


class VirusClassificationLoss(nn.Module):
    """Loss for multi-class virus type classification.
    
    Uses cross-entropy with class weights to handle imbalance.
    Only computed when illness is detected (risk > threshold).
    """
    
    def __init__(
        self,
        n_classes: int = NUM_VIRUS_CLASSES,
        class_weights: Optional[list[float]] = None,
        label_smoothing: float = 0.1,
    ):
        """Initialize virus classification loss.
        
        Args:
            n_classes: Number of virus classes.
            class_weights: Optional weights per class for imbalance.
            label_smoothing: Label smoothing factor.
        """
        super().__init__()
        
        self.n_classes = n_classes
        self.label_smoothing = label_smoothing
        
        if class_weights is not None:
            self.register_buffer("class_weights", torch.tensor(class_weights))
        else:
            self.class_weights = None
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        illness_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute virus classification loss.
        
        Args:
            logits: Virus class logits, shape (batch, n_classes).
            targets: Virus class indices, shape (batch,).
            illness_mask: Optional mask for samples with illness, shape (batch,).
                         Only compute loss on samples where illness is present.
            
        Returns:
            Tuple of (loss, loss_dict).
        """
        device = logits.device
        
        # Apply illness mask if provided
        if illness_mask is not None:
            # Only compute loss for samples with illness
            if illness_mask.sum() == 0:
                # No illness samples in batch
                return torch.tensor(0.0, device=device), {"virus_loss": torch.tensor(0.0, device=device)}
            
            logits = logits[illness_mask]
            targets = targets[illness_mask]
        
        # Cross-entropy with label smoothing
        loss = F.cross_entropy(
            logits,
            targets.long(),
            weight=self.class_weights.to(device) if self.class_weights is not None else None,
            label_smoothing=self.label_smoothing,
        )
        
        # Compute accuracy for logging
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            accuracy = (preds == targets).float().mean()
        
        loss_dict = {
            "virus_loss": loss,
            "virus_accuracy": accuracy,
        }
        
        return loss, loss_dict


class CombinedLoss(nn.Module):
    """Combined loss with prediction loss, virus classification, and regularization."""
    
    def __init__(
        self,
        horizons: list[int],
        horizon_weights: Optional[list[float]] = None,
        use_focal: bool = True,
        focal_gamma: float = 2.0,
        pos_weight: float = 5.0,
        reg_weight: float = 1.0,
        use_virus_loss: bool = True,
        virus_loss_weight: float = 0.5,
        virus_class_weights: Optional[list[float]] = None,
    ):
        """Initialize combined loss.
        
        Args:
            horizons: List of prediction horizons.
            horizon_weights: Weights for each horizon.
            use_focal: Whether to use focal loss.
            focal_gamma: Focal loss gamma parameter.
            pos_weight: Weight for positive class.
            reg_weight: Weight for regularization penalty.
            use_virus_loss: Whether to use virus classification loss.
            virus_loss_weight: Weight for virus classification loss.
            virus_class_weights: Optional class weights for virus classification.
        """
        super().__init__()
        
        self.prediction_loss = MultiHorizonLoss(
            horizons=horizons,
            horizon_weights=horizon_weights,
            use_focal=use_focal,
            focal_gamma=focal_gamma,
            pos_weight=pos_weight,
        )
        self.reg_weight = reg_weight
        
        self.use_virus_loss = use_virus_loss
        self.virus_loss_weight = virus_loss_weight
        
        if use_virus_loss:
            self.virus_loss = VirusClassificationLoss(
                class_weights=virus_class_weights,
            )
        else:
            self.virus_loss = None
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        reg_penalty: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        virus_logits: Optional[torch.Tensor] = None,
        virus_targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute combined loss.
        
        Args:
            predictions: Predicted probabilities.
            targets: Binary targets for risk prediction.
            reg_penalty: Regularization penalty from model.
            mask: Optional valid sample mask.
            virus_logits: Optional virus class logits.
            virus_targets: Optional virus class targets.
            
        Returns:
            Tuple of (total_loss, loss_dict).
        """
        pred_loss, loss_dict = self.prediction_loss(predictions, targets, mask)
        
        reg_loss = self.reg_weight * reg_penalty
        total_loss = pred_loss + reg_loss
        
        loss_dict["reg"] = reg_loss
        loss_dict["pred_loss"] = pred_loss
        
        # Add virus classification loss if enabled
        if (self.use_virus_loss and self.virus_loss is not None 
                and virus_logits is not None and virus_targets is not None):
            
            # Create illness mask: any horizon has positive label
            illness_mask = targets.sum(dim=-1) > 0 if targets.dim() > 1 else targets > 0
            
            virus_loss_val, virus_dict = self.virus_loss(
                virus_logits, virus_targets, illness_mask
            )
            
            total_loss = total_loss + self.virus_loss_weight * virus_loss_val
            loss_dict.update(virus_dict)
        
        loss_dict["total"] = total_loss
        
        return total_loss, loss_dict

