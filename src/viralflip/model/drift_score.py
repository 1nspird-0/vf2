"""Drift score module (φ): compress modality drift to scalar score.

For modality m with debiased drift z_m:
  s_m(t) = Σ_k a_mk * clip(z_mk(t), 0, +6)

Constraints:
- a_mk >= 0 (monotone risk contribution via softplus)
- Sparsity via L1 regularization

This provides feature-level interpretability: which features contributed
to the drift score for each modality.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DriftScoreModule(nn.Module):
    """Compress per-modality drift vectors to scalar drift scores.
    
    Each modality has learned non-negative weights that combine feature-level
    drifts into a single drift score. Only positive drifts contribute (one-sided).
    """
    
    def __init__(
        self,
        modality_dims: dict[str, int],
        l1_lambda: float = 0.01,
        clip_max: float = 6.0,
    ):
        """Initialize drift score module.
        
        Args:
            modality_dims: Dict mapping modality name to drift dimension.
            l1_lambda: L1 regularization strength for sparsity.
            clip_max: Maximum drift value (positive side).
        """
        super().__init__()
        
        self.modality_dims = modality_dims
        self.modalities = list(modality_dims.keys())
        self.l1_lambda = l1_lambda
        self.clip_max = clip_max
        
        # Raw weights (will be transformed via softplus for non-negativity)
        self._weights_raw = nn.ParameterDict()
        for modality, dim in modality_dims.items():
            self._weights_raw[modality] = nn.Parameter(torch.zeros(dim))
    
    def forward(
        self,
        drift_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute drift scores for each modality.
        
        Args:
            drift_dict: Dict mapping modality name to drift tensor.
                       Shape: (batch, seq_len, n_features) or (batch, n_features).
                       
        Returns:
            Dict mapping modality name to drift score tensor.
            Shape: (batch, seq_len) or (batch,).
        """
        scores = {}
        
        for modality in self.modalities:
            if modality not in drift_dict:
                continue
            
            z = drift_dict[modality]
            
            # Get non-negative weights via softplus
            a = F.softplus(self._weights_raw[modality])
            
            # Clip drift to [0, clip_max] (only positive drifts contribute)
            z_clipped = torch.clamp(z, 0.0, self.clip_max)
            
            # Weighted sum across features
            # z: (batch, [seq,] features), a: (features,)
            if z.dim() == 3:
                # (batch, seq, features) @ (features,) -> (batch, seq)
                score = torch.einsum("bsf,f->bs", z_clipped, a)
            else:
                # (batch, features) @ (features,) -> (batch,)
                score = torch.einsum("bf,f->b", z_clipped, a)
            
            scores[modality] = score
        
        return scores
    
    def get_weights(self, modality: str) -> torch.Tensor:
        """Get non-negative weights for a modality.
        
        Args:
            modality: Modality name.
            
        Returns:
            Non-negative weight tensor.
        """
        if modality not in self._weights_raw:
            raise ValueError(f"Unknown modality: {modality}")
        return F.softplus(self._weights_raw[modality])
    
    def get_feature_contributions(
        self,
        modality: str,
        drift: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get per-feature contributions to drift score.
        
        Args:
            modality: Modality name.
            drift: Drift tensor of shape (n_features,) or (batch, n_features).
            
        Returns:
            Tuple of (weights, clipped_drift, contributions).
        """
        a = self.get_weights(modality)
        z_clipped = torch.clamp(drift, 0.0, self.clip_max)
        contributions = a * z_clipped
        
        return a, z_clipped, contributions
    
    def l1_penalty(self) -> torch.Tensor:
        """Compute L1 penalty on weights for regularization."""
        penalty = torch.tensor(0.0, device=next(self.parameters()).device)
        
        for modality in self.modalities:
            a = self.get_weights(modality)
            penalty = penalty + torch.sum(a)
        
        return self.l1_lambda * penalty
    
    def get_state_summary(self) -> dict:
        """Get summary of learned weights."""
        summary = {}
        for modality in self.modalities:
            a = self.get_weights(modality).detach().cpu().numpy()
            summary[modality] = {
                "weights": a.tolist(),
                "n_active": int((a > 0.01).sum()),
                "total_weight": float(a.sum()),
            }
        return summary


class MultiModalityDriftScore(nn.Module):
    """Convenience wrapper that handles missing modalities."""
    
    def __init__(
        self,
        modality_dims: dict[str, int],
        l1_lambda: float = 0.01,
        clip_max: float = 6.0,
    ):
        """Initialize multi-modality drift score module.
        
        Args:
            modality_dims: Dict mapping modality name to drift dimension.
            l1_lambda: L1 regularization strength.
            clip_max: Maximum drift value.
        """
        super().__init__()
        
        self.drift_score = DriftScoreModule(modality_dims, l1_lambda, clip_max)
        self.modalities = list(modality_dims.keys())
    
    def forward(
        self,
        drift_dict: dict[str, torch.Tensor],
        missing_mask: Optional[dict[str, torch.Tensor]] = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Compute drift scores with missing data handling.
        
        Args:
            drift_dict: Dict mapping modality to drift tensor.
            missing_mask: Optional dict mapping modality to missing indicator.
            
        Returns:
            Tuple of (scores_dict, concatenated_scores).
            Missing modalities get score of 0.
        """
        scores = self.drift_score(drift_dict)
        
        # Build concatenated tensor in consistent order
        score_list = []
        for modality in self.modalities:
            if modality in scores:
                s = scores[modality]
                if missing_mask and modality in missing_mask:
                    # Zero out scores for missing data
                    s = s * (1 - missing_mask[modality].float())
            else:
                # Missing modality: zero score
                if drift_dict:
                    # Get shape from first available tensor
                    ref = next(iter(drift_dict.values()))
                    if ref.dim() == 3:
                        s = torch.zeros(ref.shape[0], ref.shape[1], device=ref.device)
                    else:
                        s = torch.zeros(ref.shape[0], device=ref.device)
                else:
                    s = torch.zeros(1)
            
            score_list.append(s.unsqueeze(-1))
        
        # (batch, [seq,] n_modalities)
        concatenated = torch.cat(score_list, dim=-1)
        
        return scores, concatenated
    
    def l1_penalty(self) -> torch.Tensor:
        """Get L1 penalty."""
        return self.drift_score.l1_penalty()

