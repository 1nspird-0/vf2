"""Behavior-Drift Debiasing (BDD) using Ridge regression.

Key idea: Remove drift in physiological features that is explained by
behavior/routine changes rather than health changes.

For each physiological modality m:
  z_m_debiased(t) = z_m(t) - P_m * b(t)

where b(t) is the behavior drift vector (GPS, IMU passive, screen) and
P_m is learned via ridge regression on healthy bins.

This is CRITICAL for accuracy: mobility drops and sleep changes can cause
physiological feature drift without illness.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.linear_model import Ridge


@dataclass
class DebiasResult:
    """Result of debiasing operation."""
    
    original: np.ndarray
    debiased: np.ndarray
    behavior_contribution: np.ndarray
    modality: str


class BehaviorDriftDebiaser:
    """Debias physiological features by removing behavior-explained variance.
    
    Training:
    - Collect behavior drift (GPS, IMU, screen) and physiology drift on healthy bins
    - Fit ridge regression: Z_physio = P * B + epsilon
    - Save P for each physiological modality
    
    Inference:
    - z_debiased = z_physio - P * b
    """
    
    # Behavior modalities (confounds)
    BEHAVIOR_BLOCKS = ["gps", "imu_passive", "screen"]
    
    # Physiological modalities (to debias)
    PHYSIOLOGY_BLOCKS = ["voice", "cough", "tap", "gait_active", "rppg", "light", "baro"]
    
    def __init__(
        self,
        feature_dims: dict[str, int],
        ridge_lambda: float = 1.0,
        behavior_blocks: Optional[list[str]] = None,
        physiology_blocks: Optional[list[str]] = None,
    ):
        """Initialize debiaser.
        
        Args:
            feature_dims: Dict mapping modality name to feature dimension.
            ridge_lambda: Ridge regularization strength.
            behavior_blocks: Override behavior modalities.
            physiology_blocks: Override physiology modalities.
        """
        self.feature_dims = feature_dims
        self.ridge_lambda = ridge_lambda
        self.behavior_blocks = behavior_blocks or self.BEHAVIOR_BLOCKS
        self.physiology_blocks = physiology_blocks or self.PHYSIOLOGY_BLOCKS
        
        # Compute behavior vector dimension
        self.behavior_dim = sum(
            feature_dims.get(m, 0) for m in self.behavior_blocks
        )
        
        # Projection matrices: modality -> Ridge model
        self._models: dict[str, Ridge] = {}
        self._is_fitted = False
    
    @property
    def is_fitted(self) -> bool:
        """Check if debiaser has been fitted."""
        return self._is_fitted
    
    def fit(
        self,
        behavior_drifts: dict[str, np.ndarray],
        physiology_drifts: dict[str, np.ndarray],
        sample_weights: Optional[np.ndarray] = None,
    ) -> "BehaviorDriftDebiaser":
        """Fit debiasing models on healthy data.
        
        Args:
            behavior_drifts: Dict mapping behavior modality to drift array
                            (n_samples, n_features).
            physiology_drifts: Dict mapping physiology modality to drift array
                              (n_samples, n_features).
            sample_weights: Optional sample weights.
            
        Returns:
            Self for chaining.
        """
        # Build behavior matrix B (n_samples, behavior_dim)
        B = self._build_behavior_matrix(behavior_drifts)
        
        if B.shape[0] == 0:
            raise ValueError("No samples provided for fitting")
        
        # Fit ridge regression for each physiological modality
        for modality in self.physiology_blocks:
            if modality not in physiology_drifts:
                continue
            
            Z = physiology_drifts[modality]
            
            if Z.shape[0] != B.shape[0]:
                raise ValueError(
                    f"Sample count mismatch: {modality} has {Z.shape[0]} samples, "
                    f"behavior has {B.shape[0]}"
                )
            
            # Fit ridge model
            model = Ridge(alpha=self.ridge_lambda, fit_intercept=False)
            model.fit(B, Z, sample_weight=sample_weights)
            
            self._models[modality] = model
        
        self._is_fitted = True
        return self
    
    def transform(
        self,
        behavior_drift: dict[str, np.ndarray],
        physiology_drift: dict[str, np.ndarray],
    ) -> dict[str, DebiasResult]:
        """Apply debiasing to physiology drifts.
        
        Args:
            behavior_drift: Dict mapping behavior modality to drift vector.
            physiology_drift: Dict mapping physiology modality to drift vector.
            
        Returns:
            Dict mapping physiology modality to DebiasResult.
        """
        if not self._is_fitted:
            raise RuntimeError("Debiaser has not been fitted")
        
        # Build behavior vector
        b = self._build_behavior_vector(behavior_drift)
        
        results = {}
        for modality in self.physiology_blocks:
            if modality not in physiology_drift:
                continue
            
            z = physiology_drift[modality]
            
            if modality in self._models:
                # Predict behavior-explained drift
                b_contribution = self._models[modality].predict(b.reshape(1, -1)).flatten()
                z_debiased = z - b_contribution
            else:
                # No model for this modality, return unchanged
                b_contribution = np.zeros_like(z)
                z_debiased = z
            
            results[modality] = DebiasResult(
                original=z,
                debiased=z_debiased,
                behavior_contribution=b_contribution,
                modality=modality,
            )
        
        return results
    
    def debias(
        self,
        behavior_drift: dict[str, np.ndarray],
        physiology_drift: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Simple interface to get debiased drift vectors.
        
        Args:
            behavior_drift: Dict mapping behavior modality to drift vector.
            physiology_drift: Dict mapping physiology modality to drift vector.
            
        Returns:
            Dict mapping physiology modality to debiased drift vector.
        """
        results = self.transform(behavior_drift, physiology_drift)
        return {m: r.debiased for m, r in results.items()}
    
    def _build_behavior_matrix(
        self,
        behavior_drifts: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Build behavior matrix from modality drifts.
        
        Args:
            behavior_drifts: Dict mapping behavior modality to drift array.
            
        Returns:
            Behavior matrix of shape (n_samples, behavior_dim).
        """
        blocks = []
        n_samples = None
        
        for modality in self.behavior_blocks:
            if modality in behavior_drifts:
                drift = behavior_drifts[modality]
                if n_samples is None:
                    n_samples = drift.shape[0]
                blocks.append(drift)
            else:
                # Missing modality: fill with zeros
                dim = self.feature_dims.get(modality, 0)
                if n_samples is not None:
                    blocks.append(np.zeros((n_samples, dim)))
        
        if not blocks or n_samples is None:
            return np.array([])
        
        return np.hstack(blocks)
    
    def _build_behavior_vector(
        self,
        behavior_drift: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Build behavior vector from modality drifts.
        
        Args:
            behavior_drift: Dict mapping behavior modality to drift vector.
            
        Returns:
            Concatenated behavior vector.
        """
        blocks = []
        
        for modality in self.behavior_blocks:
            if modality in behavior_drift:
                blocks.append(behavior_drift[modality])
            else:
                dim = self.feature_dims.get(modality, 0)
                blocks.append(np.zeros(dim))
        
        return np.concatenate(blocks)
    
    def get_projection_matrix(self, modality: str) -> Optional[np.ndarray]:
        """Get the projection matrix P for a modality.
        
        Args:
            modality: Physiological modality name.
            
        Returns:
            Projection matrix or None if not fitted.
        """
        if modality not in self._models:
            return None
        return self._models[modality].coef_
    
    def get_state_dict(self) -> dict:
        """Get serializable state dictionary."""
        state = {
            "feature_dims": self.feature_dims,
            "ridge_lambda": self.ridge_lambda,
            "behavior_blocks": self.behavior_blocks,
            "physiology_blocks": self.physiology_blocks,
            "is_fitted": self._is_fitted,
            "models": {},
        }
        
        for modality, model in self._models.items():
            state["models"][modality] = {
                "coef": model.coef_.tolist(),
            }
        
        return state
    
    def load_state_dict(self, state: dict) -> None:
        """Load state from dictionary."""
        self.feature_dims = state["feature_dims"]
        self.ridge_lambda = state["ridge_lambda"]
        self.behavior_blocks = state["behavior_blocks"]
        self.physiology_blocks = state["physiology_blocks"]
        self._is_fitted = state["is_fitted"]
        
        self._models = {}
        for modality, model_state in state["models"].items():
            model = Ridge(alpha=self.ridge_lambda, fit_intercept=False)
            model.coef_ = np.array(model_state["coef"])
            # Set dummy intercept
            model.intercept_ = np.zeros(model.coef_.shape[0])
            self._models[modality] = model
        
        # Recompute behavior dim
        self.behavior_dim = sum(
            self.feature_dims.get(m, 0) for m in self.behavior_blocks
        )

