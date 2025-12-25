"""Explanation engine for ViralFlip predictions.

Provides:
- Top contributing (modality, lag) pairs with weights
- Feature-level drift breakdown for top modalities
- Counterfactual risk deltas (risk if contributor removed)
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch

from viralflip.model.viralflip import ViralFlip


@dataclass
class FeatureContribution:
    """Contribution of a single feature within a modality."""
    
    feature_name: str
    feature_idx: int
    drift_z: float  # Z-score drift
    weight: float  # Learned weight
    contribution: float  # weight * max(drift_z, 0)
    
    def to_dict(self) -> dict:
        return {
            "feature_name": self.feature_name,
            "feature_idx": self.feature_idx,
            "drift_z": self.drift_z,
            "weight": self.weight,
            "contribution": self.contribution,
        }


@dataclass
class ModalityContribution:
    """Contribution of a modality at a specific lag."""
    
    modality: str
    lag: int
    drift_score: float  # Aggregate drift score
    weight: float  # Lag lattice weight
    contribution: float  # weight * drift_score
    delta_if_removed: float  # Counterfactual risk change
    feature_contributions: list[FeatureContribution] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "modality": self.modality,
            "lag": self.lag,
            "drift_score": self.drift_score,
            "weight": self.weight,
            "contribution": self.contribution,
            "delta_if_removed": self.delta_if_removed,
            "feature_contributions": [f.to_dict() for f in self.feature_contributions],
        }


@dataclass
class Explanation:
    """Complete explanation for a prediction."""
    
    horizon: int
    risk: float
    confidence: float
    
    # Top contributors
    top_contributors: list[ModalityContribution]
    
    # Summary
    total_explained: float  # Sum of contributions
    n_active_modalities: int
    n_missing_modalities: int
    
    # Quality info
    quality_summary: dict
    
    def to_dict(self) -> dict:
        return {
            "horizon": self.horizon,
            "risk": self.risk,
            "confidence": self.confidence,
            "top_contributors": [c.to_dict() for c in self.top_contributors],
            "total_explained": self.total_explained,
            "n_active_modalities": self.n_active_modalities,
            "n_missing_modalities": self.n_missing_modalities,
            "quality_summary": self.quality_summary,
        }
    
    def summary_text(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Risk ({self.horizon}h horizon): {self.risk:.1%} (confidence: {self.confidence:.1%})",
            "",
            "Top Contributors:",
        ]
        
        for i, contrib in enumerate(self.top_contributors[:5], 1):
            mod = contrib.modality
            lag = contrib.lag
            score = contrib.drift_score
            delta = contrib.delta_if_removed
            
            lag_str = f"(lag {lag})" if lag > 0 else "(current)"
            lines.append(f"  {i}. {mod} {lag_str}: score={score:.2f}, Δ={delta:+.1%}")
            
            # Top features
            for feat in contrib.feature_contributions[:3]:
                lines.append(f"       - {feat.feature_name}: z={feat.drift_z:.1f}")
        
        return "\n".join(lines)


class ExplanationEngine:
    """Generate explanations for ViralFlip predictions."""
    
    # Feature names for each modality
    FEATURE_NAMES = {
        "voice": [
            "f0_mean", "f0_std", "f0_range", "f0_slope",
            "jitter", "shimmer", "hnr",
            "spec_centroid", "spec_bandwidth", "spec_rolloff", "spec_slope",
            "mfcc1_mean", "mfcc1_std", "mfcc2_mean", "mfcc2_std",
            "mfcc3_mean", "mfcc3_std", "mfcc4_mean", "mfcc4_std",
            "mfcc5_mean", "mfcc5_std", "mfcc6_mean", "mfcc6_std",
            "speaking_rate", "pause_ratio",
        ],
        "cough": [
            "count_total", "count_night", "count_day",
            "burstiness", "max_hourly", "conf_mean",
        ],
        "tap": [
            "tap_count", "iti_mean", "iti_std", "iti_cv",
            "outlier_rate", "fatigue_slope",
        ],
        "gait_active": [
            "cadence", "interval_mean", "interval_std", "interval_cv",
            "jerk_mean", "jerk_std", "sway", "regularity",
        ],
        "rppg": [
            "hr_mean", "ibi_std", "hrv_rmssd", "pulse_amp_mean", "pulse_amp_cv",
        ],
        "light": [
            "lux_mean", "lux_std", "lux_evening", "circadian_score",
        ],
        "baro": [
            "pressure_mean", "pressure_std", "pressure_slope", "jump_count",
        ],
    }
    
    def __init__(
        self,
        model: ViralFlip,
        top_k: int = 5,
    ):
        """Initialize explanation engine.
        
        Args:
            model: Trained ViralFlip model.
            top_k: Number of top contributors to include.
        """
        self.model = model
        self.top_k = top_k
    
    def explain(
        self,
        drift_dict: dict[str, np.ndarray],
        horizon: int,
        missing_mask: Optional[dict[str, bool]] = None,
        quality_scores: Optional[dict[str, float]] = None,
        user_id: Optional[str] = None,
    ) -> Explanation:
        """Generate explanation for a prediction.
        
        Args:
            drift_dict: Dict mapping modality to drift z-scores.
                       Shape: (seq_len, n_features) or (n_features,).
            horizon: Horizon to explain.
            missing_mask: Dict mapping modality to missing flag.
            quality_scores: Dict mapping modality to quality score.
            user_id: Optional user ID for personalization.
            
        Returns:
            Explanation object.
        """
        self.model.eval()
        
        # Get horizon index
        horizon_idx = self.model.horizons.index(horizon)
        
        with torch.no_grad():
            # Get prediction
            output = self.model.predict(
                drift_dict, missing_mask, quality_scores, user_id
            )
            
            risk = output.risks[horizon]
            confidence = output.confidences[horizon]
            drift_scores = output.drift_scores
            
            # Get lag lattice weights
            lattice_weights = self.model.lag_lattice.get_weight_for_horizon(horizon_idx)
            lattice_weights = lattice_weights.cpu().numpy()
            
            # Get drift score weights
            drift_weights = {}
            for mod in self.model.physiology_modalities:
                weights = self.model.drift_score.get_weights(mod)
                drift_weights[mod] = weights.cpu().numpy()
        
        # Compute contributions
        contributions = []
        
        for m_idx, modality in enumerate(self.model.physiology_modalities):
            if modality not in drift_dict:
                continue
            
            drift = drift_dict[modality]
            if drift.ndim == 2:
                drift = drift[-1]  # Use last timestep
            
            # Drift score
            ds = drift_scores.get(modality, 0.0)
            
            for lag in range(self.model.lag_lattice.max_lag + 1):
                w = lattice_weights[m_idx, lag]
                contrib = w * ds
                
                if contrib > 0:
                    # Compute counterfactual
                    delta = self._compute_counterfactual(
                        drift_dict, modality, lag, horizon_idx
                    )
                    
                    # Get feature contributions
                    feat_contribs = self._get_feature_contributions(
                        modality, drift, drift_weights.get(modality, np.array([]))
                    )
                    
                    contributions.append(ModalityContribution(
                        modality=modality,
                        lag=lag,
                        drift_score=ds,
                        weight=float(w),
                        contribution=float(contrib),
                        delta_if_removed=delta,
                        feature_contributions=feat_contribs,
                    ))
        
        # Sort by contribution
        contributions.sort(key=lambda c: -c.contribution)
        top_contributors = contributions[:self.top_k]
        
        # Summary stats
        total_explained = sum(c.contribution for c in contributions)
        n_active = len(set(c.modality for c in contributions if c.contribution > 0))
        n_missing = sum(1 for m in self.model.physiology_modalities 
                       if missing_mask and missing_mask.get(m, False))
        
        return Explanation(
            horizon=horizon,
            risk=risk,
            confidence=confidence,
            top_contributors=top_contributors,
            total_explained=total_explained,
            n_active_modalities=n_active,
            n_missing_modalities=n_missing,
            quality_summary=output.quality_summary,
        )
    
    def _compute_counterfactual(
        self,
        drift_dict: dict[str, np.ndarray],
        modality: str,
        lag: int,
        horizon_idx: int,
    ) -> float:
        """Compute risk change if removing a contributor.
        
        Args:
            drift_dict: Current drift dict.
            modality: Modality to remove.
            lag: Lag to remove.
            horizon_idx: Horizon index.
            
        Returns:
            Risk change (negative = risk decreases).
        """
        # For simplicity, set modality drift to zero
        modified_dict = {k: v.copy() for k, v in drift_dict.items()}
        
        if modality in modified_dict:
            modified_dict[modality] = np.zeros_like(modified_dict[modality])
        
        with torch.no_grad():
            original = self.model.predict(drift_dict, None, None, None)
            modified = self.model.predict(modified_dict, None, None, None)
        
        horizon = self.model.horizons[horizon_idx]
        delta = modified.risks[horizon] - original.risks[horizon]
        
        return float(delta)
    
    def _get_feature_contributions(
        self,
        modality: str,
        drift: np.ndarray,
        weights: np.ndarray,
    ) -> list[FeatureContribution]:
        """Get per-feature contributions within a modality.
        
        Args:
            modality: Modality name.
            drift: Drift z-scores.
            weights: Feature weights.
            
        Returns:
            List of FeatureContribution objects.
        """
        if len(weights) == 0 or len(drift) == 0:
            return []
        
        # Ensure same length
        n = min(len(drift), len(weights))
        drift = drift[:n]
        weights = weights[:n]
        
        # Feature names
        names = self.FEATURE_NAMES.get(modality, [f"f{i}" for i in range(n)])
        if len(names) < n:
            names = names + [f"f{i}" for i in range(len(names), n)]
        
        contributions = []
        for i in range(n):
            z = float(drift[i])
            w = float(weights[i])
            c = w * max(z, 0)
            
            contributions.append(FeatureContribution(
                feature_name=names[i],
                feature_idx=i,
                drift_z=z,
                weight=w,
                contribution=c,
            ))
        
        # Sort by contribution
        contributions.sort(key=lambda f: -f.contribution)
        
        return contributions
    
    def explain_all_horizons(
        self,
        drift_dict: dict[str, np.ndarray],
        missing_mask: Optional[dict[str, bool]] = None,
        quality_scores: Optional[dict[str, float]] = None,
        user_id: Optional[str] = None,
    ) -> dict[int, Explanation]:
        """Generate explanations for all horizons.
        
        Args:
            drift_dict: Drift dictionary.
            missing_mask: Missing mask.
            quality_scores: Quality scores.
            user_id: User ID.
            
        Returns:
            Dict mapping horizon to Explanation.
        """
        explanations = {}
        for horizon in self.model.horizons:
            explanations[horizon] = self.explain(
                drift_dict, horizon, missing_mask, quality_scores, user_id
            )
        return explanations

