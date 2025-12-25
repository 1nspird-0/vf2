"""Sparse interaction module for modality pairs.

Optional interactions (only if improving validation AUPRC):
  int_H,i,j(t) = g_H,i,j * softplus(u_i,j * s_i(t) * s_j(t))

With g >= 0 and heavy L1 to keep only a few interactions.

Candidate pairs: (voice, rppg), (voice, cough), (rppg, gait), etc.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class InteractionModule(nn.Module):
    """Sparse pairwise interactions between modality drift scores.
    
    Captures synergistic effects where combined drift in two modalities
    is more predictive than individual drifts.
    """
    
    def __init__(
        self,
        modality_names: list[str],
        horizons: list[int],
        interaction_pairs: Optional[list[tuple[str, str]]] = None,
        l1_lambda: float = 0.1,
    ):
        """Initialize interaction module.
        
        Args:
            modality_names: List of modality names.
            horizons: List of prediction horizons.
            interaction_pairs: List of (modality_i, modality_j) pairs.
                              If None, uses default pairs.
            l1_lambda: L1 regularization strength (heavy for sparsity).
        """
        super().__init__()
        
        self.modality_names = modality_names
        self.modality_to_idx = {m: i for i, m in enumerate(modality_names)}
        self.horizons = horizons
        self.n_horizons = len(horizons)
        self.l1_lambda = l1_lambda
        
        # Default interaction pairs
        if interaction_pairs is None:
            interaction_pairs = [
                ("voice", "rppg"),
                ("voice", "cough"),
                ("rppg", "gait_active"),
                ("cough", "rppg"),
                ("tap", "voice"),
            ]
        
        # Filter to valid pairs
        self.pairs = []
        for m1, m2 in interaction_pairs:
            if m1 in self.modality_to_idx and m2 in self.modality_to_idx:
                self.pairs.append((m1, m2))
        
        self.n_pairs = len(self.pairs)
        
        if self.n_pairs == 0:
            # No valid pairs, module is a no-op
            self.enabled = False
            return
        
        self.enabled = True
        
        # Interaction scale parameters (per pair)
        self._u_raw = nn.Parameter(torch.zeros(self.n_pairs))
        
        # Interaction weights (per horizon, per pair)
        # Heavy L1 will zero out most
        self._g_raw = nn.Parameter(torch.zeros(self.n_horizons, self.n_pairs))
    
    def forward(
        self,
        drift_scores: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute interaction contributions.
        
        Args:
            drift_scores: Dict mapping modality to drift score tensor.
                         Shape: (batch,) or (batch, seq_len).
                         
        Returns:
            Interaction contributions of shape (batch, [seq,] n_horizons).
        """
        if not self.enabled:
            # Return zeros
            ref = next(iter(drift_scores.values()))
            if ref.dim() == 2:
                return torch.zeros(ref.shape[0], ref.shape[1], self.n_horizons, device=ref.device)
            else:
                return torch.zeros(ref.shape[0], self.n_horizons, device=ref.device)
        
        # Get shapes
        ref = next(iter(drift_scores.values()))
        has_seq = ref.dim() == 2
        
        # Get non-negative weights
        u = F.softplus(self._u_raw)  # (n_pairs,)
        g = F.softplus(self._g_raw)  # (n_horizons, n_pairs)
        
        # Compute interactions
        interactions = []
        
        for pair_idx, (m1, m2) in enumerate(self.pairs):
            if m1 not in drift_scores or m2 not in drift_scores:
                # Missing modality, zero interaction
                if has_seq:
                    interactions.append(torch.zeros_like(ref).unsqueeze(-1).expand(-1, -1, self.n_horizons))
                else:
                    interactions.append(torch.zeros(ref.shape[0], self.n_horizons, device=ref.device))
                continue
            
            s1 = drift_scores[m1]  # (batch,) or (batch, seq)
            s2 = drift_scores[m2]
            
            # Interaction: g * softplus(u * s1 * s2)
            product = s1 * s2  # Element-wise
            scaled = u[pair_idx] * product
            activated = F.softplus(scaled)
            
            # Weight by g for each horizon
            # activated: (batch,) or (batch, seq)
            # g[:, pair_idx]: (n_horizons,)
            if has_seq:
                # (batch, seq) * (n_horizons,) -> (batch, seq, n_horizons)
                weighted = activated.unsqueeze(-1) * g[:, pair_idx].view(1, 1, -1)
            else:
                # (batch,) * (n_horizons,) -> (batch, n_horizons)
                weighted = activated.unsqueeze(-1) * g[:, pair_idx].view(1, -1)
            
            interactions.append(weighted)
        
        # Sum all interactions
        total = torch.stack(interactions, dim=0).sum(dim=0)
        
        return total
    
    def l1_penalty(self) -> torch.Tensor:
        """Compute L1 penalty on interaction weights."""
        if not self.enabled:
            return torch.tensor(0.0)
        
        g = F.softplus(self._g_raw)
        return self.l1_lambda * torch.sum(g)
    
    def get_active_interactions(self, threshold: float = 0.01) -> list[dict]:
        """Get list of active (non-zero) interactions.
        
        Args:
            threshold: Weight threshold for considering active.
            
        Returns:
            List of dicts with interaction info.
        """
        if not self.enabled:
            return []
        
        g = F.softplus(self._g_raw).detach().cpu().numpy()
        u = F.softplus(self._u_raw).detach().cpu().numpy()
        
        active = []
        for pair_idx, (m1, m2) in enumerate(self.pairs):
            for h_idx, horizon in enumerate(self.horizons):
                weight = g[h_idx, pair_idx]
                if weight > threshold:
                    active.append({
                        "modality_1": m1,
                        "modality_2": m2,
                        "horizon": horizon,
                        "weight": float(weight),
                        "scale": float(u[pair_idx]),
                    })
        
        return active
    
    def get_state_summary(self) -> dict:
        """Get summary of interaction weights."""
        if not self.enabled:
            return {"enabled": False, "n_pairs": 0}
        
        g = F.softplus(self._g_raw).detach().cpu().numpy()
        
        return {
            "enabled": True,
            "n_pairs": self.n_pairs,
            "pairs": self.pairs,
            "active_interactions": self.get_active_interactions(),
            "total_weight": float(g.sum()),
        }

