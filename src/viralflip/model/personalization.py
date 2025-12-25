"""Per-user calibration for personalized predictions.

Per-user calibration per horizon:
  r'_H = sigmoid(A_uH * logit(r_H) + c_uH)

- Initialize A=1, c=0
- Update online with small learning rate when labels arrive
- Constrain A in [0.5, 2.0]

This allows adapting to individual baseline risk levels while
maintaining the model's relative ordering.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class PersonalizationLayer(nn.Module):
    """Per-user calibration layer.
    
    Applies a linear transformation in logit space followed by sigmoid
    to personalize predictions for each user.
    """
    
    def __init__(
        self,
        n_horizons: int,
        scale_bounds: tuple[float, float] = (0.5, 2.0),
        learning_rate: float = 0.01,
    ):
        """Initialize personalization layer.
        
        Args:
            n_horizons: Number of prediction horizons.
            scale_bounds: Bounds for scale parameter A.
            learning_rate: Learning rate for online updates.
        """
        super().__init__()
        
        self.n_horizons = n_horizons
        self.scale_bounds = scale_bounds
        self.learning_rate = learning_rate
        
        # User parameters: user_id -> (scale, bias) tensors
        self._user_params: dict[str, dict[str, torch.Tensor]] = {}
    
    def register_user(self, user_id: str) -> None:
        """Register a new user with default parameters.
        
        Args:
            user_id: User identifier.
        """
        if user_id not in self._user_params:
            self._user_params[user_id] = {
                "scale": torch.ones(self.n_horizons),
                "bias": torch.zeros(self.n_horizons),
            }
    
    def forward(
        self,
        probs: torch.Tensor,
        user_ids: Optional[list[str]] = None,
    ) -> torch.Tensor:
        """Apply personalization to probabilities.
        
        Args:
            probs: Probability tensor of shape (batch, [seq,] n_horizons).
            user_ids: Optional list of user IDs for each batch element.
                     If None, returns probabilities unchanged.
                     
        Returns:
            Personalized probabilities of same shape.
        """
        if user_ids is None:
            return probs
        
        # Move to logit space
        eps = 1e-7
        logits = torch.log(probs.clamp(eps, 1 - eps) / (1 - probs.clamp(eps, 1 - eps)))
        
        # Apply per-user transformation
        has_seq = probs.dim() == 3
        
        if has_seq:
            # (batch, seq, n_horizons)
            calibrated_logits = torch.zeros_like(logits)
            for i, user_id in enumerate(user_ids):
                if user_id not in self._user_params:
                    self.register_user(user_id)
                
                scale = self._user_params[user_id]["scale"].to(logits.device)
                bias = self._user_params[user_id]["bias"].to(logits.device)
                
                # Apply: A * logit + c
                calibrated_logits[i] = logits[i] * scale.view(1, -1) + bias.view(1, -1)
        else:
            # (batch, n_horizons)
            calibrated_logits = torch.zeros_like(logits)
            for i, user_id in enumerate(user_ids):
                if user_id not in self._user_params:
                    self.register_user(user_id)
                
                scale = self._user_params[user_id]["scale"].to(logits.device)
                bias = self._user_params[user_id]["bias"].to(logits.device)
                
                calibrated_logits[i] = logits[i] * scale + bias
        
        # Back to probability space
        calibrated_probs = torch.sigmoid(calibrated_logits)
        
        return calibrated_probs
    
    def update_user(
        self,
        user_id: str,
        predictions: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """Update user parameters based on observed labels.
        
        Uses simple gradient descent on calibration parameters.
        
        Args:
            user_id: User identifier.
            predictions: Predicted probabilities, shape (n_samples, n_horizons).
            labels: Binary labels, shape (n_samples, n_horizons).
        """
        if user_id not in self._user_params:
            self.register_user(user_id)
        
        predictions = predictions.detach()
        labels = labels.detach().float()
        
        # Current parameters
        scale = self._user_params[user_id]["scale"].clone()
        bias = self._user_params[user_id]["bias"].clone()
        
        # Move to logit space
        eps = 1e-7
        logits = torch.log(predictions.clamp(eps, 1 - eps) / (1 - predictions.clamp(eps, 1 - eps)))
        
        # Apply current calibration
        calibrated_logits = logits * scale.view(1, -1) + bias.view(1, -1)
        calibrated_probs = torch.sigmoid(calibrated_logits)
        
        # Compute gradient of BCE loss
        # d(BCE)/d(calibrated_logits) = calibrated_probs - labels
        grad_logits = calibrated_probs - labels  # (n_samples, n_horizons)
        
        # Gradient w.r.t. scale and bias
        # d(calibrated_logits)/d(scale) = logits
        # d(calibrated_logits)/d(bias) = 1
        grad_scale = (grad_logits * logits).mean(dim=0)
        grad_bias = grad_logits.mean(dim=0)
        
        # Update with gradient descent
        scale = scale - self.learning_rate * grad_scale
        bias = bias - self.learning_rate * grad_bias
        
        # Clamp scale to bounds
        scale = scale.clamp(self.scale_bounds[0], self.scale_bounds[1])
        
        # Store updated parameters
        self._user_params[user_id]["scale"] = scale
        self._user_params[user_id]["bias"] = bias
    
    def get_user_params(self, user_id: str) -> Optional[dict]:
        """Get calibration parameters for a user.
        
        Args:
            user_id: User identifier.
            
        Returns:
            Dict with 'scale' and 'bias' arrays, or None if not registered.
        """
        if user_id not in self._user_params:
            return None
        
        return {
            "scale": self._user_params[user_id]["scale"].numpy(),
            "bias": self._user_params[user_id]["bias"].numpy(),
        }
    
    def set_user_params(self, user_id: str, scale: np.ndarray, bias: np.ndarray) -> None:
        """Set calibration parameters for a user.
        
        Args:
            user_id: User identifier.
            scale: Scale array.
            bias: Bias array.
        """
        self._user_params[user_id] = {
            "scale": torch.from_numpy(scale).float(),
            "bias": torch.from_numpy(bias).float(),
        }
    
    def get_state_dict(self) -> dict:
        """Get serializable state dictionary."""
        state = {}
        for user_id, params in self._user_params.items():
            state[user_id] = {
                "scale": params["scale"].numpy().tolist(),
                "bias": params["bias"].numpy().tolist(),
            }
        return state
    
    def load_state_dict(self, state: dict) -> None:
        """Load state from dictionary."""
        self._user_params = {}
        for user_id, params in state.items():
            self._user_params[user_id] = {
                "scale": torch.tensor(params["scale"]),
                "bias": torch.tensor(params["bias"]),
            }

