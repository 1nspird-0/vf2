"""Drift-Lattice Hazard Network (DLHN) for multi-horizon forecasting.

Multi-horizon logistic hazard model with lag lattice:

For each horizon H:
  logit(r_H(t)) = b_H + Σ_m Σ_ℓ w_H,m,ℓ * s_m(t-ℓ) + missing_indicators

Constraints:
- w >= 0 (monotone; interpretability via softplus)
- Sparsity across m and ℓ via L1

This captures which modalities at which lags are predictive of each horizon.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LagLatticeHazardModel(nn.Module):
    """Multi-horizon logistic hazard model with lag structure.
    
    Predicts P(onset in (t, t+H]) for H in {24, 48, 72} hours.
    Uses drift scores at multiple lags to capture temporal patterns.
    """
    
    def __init__(
        self,
        n_modalities: int,
        horizons: list[int],
        max_lag: int = 12,
        l1_lambda: float = 0.01,
        use_missing_indicators: bool = True,
    ):
        """Initialize lag lattice model.
        
        Args:
            n_modalities: Number of drift score modalities.
            horizons: List of prediction horizons (e.g., [24, 48, 72]).
            max_lag: Maximum lag to consider (in bins).
            l1_lambda: L1 regularization for sparsity.
            use_missing_indicators: Whether to use missing data indicators.
        """
        super().__init__()
        
        self.n_modalities = n_modalities
        self.horizons = horizons
        self.n_horizons = len(horizons)
        self.max_lag = max_lag
        self.l1_lambda = l1_lambda
        self.use_missing_indicators = use_missing_indicators
        
        # Bias per horizon
        self.bias = nn.Parameter(torch.zeros(self.n_horizons))
        
        # Weights: (n_horizons, n_modalities, max_lag + 1)
        # Use softplus for non-negativity
        self._weights_raw = nn.Parameter(
            torch.zeros(self.n_horizons, n_modalities, max_lag + 1)
        )
        
        # Missing indicator weights (if enabled)
        if use_missing_indicators:
            self._missing_weights_raw = nn.Parameter(
                torch.zeros(self.n_horizons, n_modalities, max_lag + 1)
            )
        else:
            self._missing_weights_raw = None
    
    def forward(
        self,
        drift_scores: torch.Tensor,
        missing_indicators: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            drift_scores: Tensor of shape (batch, seq_len, n_modalities).
                         Contains drift scores at each time step.
            missing_indicators: Optional tensor of shape (batch, seq_len, n_modalities).
                               1 = missing, 0 = present.
                               
        Returns:
            Risk probabilities of shape (batch, seq_len, n_horizons).
        """
        batch_size, seq_len, n_mod = drift_scores.shape
        
        # Get non-negative weights
        weights = F.softplus(self._weights_raw)  # (n_horizons, n_modalities, n_lags)
        
        # Build lagged features
        # For each time t, we need s(t), s(t-1), ..., s(t-max_lag)
        logits = self.bias.view(1, 1, -1).expand(batch_size, seq_len, -1)
        
        for lag in range(self.max_lag + 1):
            if lag == 0:
                lagged = drift_scores
            elif lag >= seq_len:
                # All values are beyond the sequence - use zeros
                lagged = torch.zeros_like(drift_scores)
            else:
                # Shift and pad with zeros at the beginning
                lagged = F.pad(drift_scores[:, :-lag, :], (0, 0, lag, 0), value=0.0)
            
            # Weight contribution from this lag
            # lagged: (batch, seq, n_mod)
            # weights[:, :, lag]: (n_horizons, n_mod)
            # Result: (batch, seq, n_horizons)
            lag_weights = weights[:, :, lag]  # (n_horizons, n_mod)
            contribution = torch.einsum("bsm,hm->bsh", lagged, lag_weights)
            logits = logits + contribution
        
        # Missing indicators (prevent missing = healthy bias)
        if self.use_missing_indicators and missing_indicators is not None:
            missing_weights = F.softplus(self._missing_weights_raw)
            
            for lag in range(self.max_lag + 1):
                if lag == 0:
                    lagged_miss = missing_indicators
                elif lag >= seq_len:
                    # All values are beyond the sequence - use zeros
                    lagged_miss = torch.zeros_like(missing_indicators)
                else:
                    lagged_miss = F.pad(
                        missing_indicators[:, :-lag, :], 
                        (0, 0, lag, 0), 
                        value=0.0
                    )
                
                miss_lag_weights = missing_weights[:, :, lag]
                miss_contribution = torch.einsum("bsm,hm->bsh", lagged_miss.float(), miss_lag_weights)
                logits = logits + miss_contribution
        
        # Convert to probabilities
        probs = torch.sigmoid(logits)
        
        return probs
    
    def get_weights(self) -> torch.Tensor:
        """Get non-negative weight tensor.
        
        Returns:
            Weights of shape (n_horizons, n_modalities, max_lag + 1).
        """
        return F.softplus(self._weights_raw)
    
    def get_weight_for_horizon(self, horizon_idx: int) -> torch.Tensor:
        """Get weights for a specific horizon.
        
        Args:
            horizon_idx: Index of horizon.
            
        Returns:
            Weights of shape (n_modalities, max_lag + 1).
        """
        return self.get_weights()[horizon_idx]
    
    def l1_penalty(self) -> torch.Tensor:
        """Compute L1 penalty on weights."""
        weights = self.get_weights()
        penalty = torch.sum(weights)
        
        if self.use_missing_indicators and self._missing_weights_raw is not None:
            missing_weights = F.softplus(self._missing_weights_raw)
            penalty = penalty + torch.sum(missing_weights)
        
        return self.l1_lambda * penalty
    
    def get_top_contributors(
        self,
        drift_scores: torch.Tensor,
        horizon_idx: int,
        top_k: int = 5,
    ) -> list[tuple[int, int, float, float]]:
        """Get top contributing (modality, lag) pairs for a horizon.
        
        Args:
            drift_scores: Drift scores of shape (seq_len, n_modalities) or
                         (batch, seq_len, n_modalities).
            horizon_idx: Horizon index.
            top_k: Number of top contributors.
            
        Returns:
            List of (modality_idx, lag, weight, contribution) tuples.
        """
        weights = self.get_weights()[horizon_idx]  # (n_mod, n_lags)
        
        if drift_scores.dim() == 3:
            drift_scores = drift_scores[-1, -1, :]  # Use last timestep
        elif drift_scores.dim() == 2:
            drift_scores = drift_scores[-1, :]
        
        contributions = []
        
        for m in range(self.n_modalities):
            for lag in range(self.max_lag + 1):
                w = weights[m, lag].item()
                s = drift_scores[m].item() if lag == 0 else 0.0  # Simplified
                contrib = w * max(s, 0)
                contributions.append((m, lag, w, contrib))
        
        # Sort by contribution descending
        contributions.sort(key=lambda x: -x[3])
        
        return contributions[:top_k]
    
    def get_state_summary(self) -> dict:
        """Get summary of learned weights."""
        weights = self.get_weights().detach().cpu().numpy()
        
        summary = {}
        for h_idx, horizon in enumerate(self.horizons):
            w = weights[h_idx]
            summary[f"horizon_{horizon}"] = {
                "total_weight": float(w.sum()),
                "active_cells": int((w > 0.01).sum()),
                "max_weight": float(w.max()),
                "weight_by_modality": w.sum(axis=1).tolist(),
                "weight_by_lag": w.sum(axis=0).tolist(),
            }
        
        return summary

