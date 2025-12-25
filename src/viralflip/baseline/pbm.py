"""Personal Baseline Memory (PBM) for within-person normalization.

The PBM converts raw features into "within-person drift" z-scores robustly:
1. Initialize baseline using first N healthy days/bins with median/MAD
2. Compute drift z_k(t) = clip((f_k(t) - mu_k) / sig_k, -6, +6)
3. Update baseline using robust EMA only on "safe" bins

This is CRITICAL for accuracy: population-level features are confounded by
individual differences. PBM captures personal variation to detect drift.

LEAKAGE PREVENTION:
- Baseline at time t uses only data from times <= t-1
- Safe bin determination uses only predictions available at update time
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class BaselineState:
    """State of baseline for one user and one modality."""
    
    # Location (mean proxy)
    mu: np.ndarray
    
    # Scale (std proxy)
    sigma: np.ndarray
    
    # Initialization info
    n_init_samples: int = 0
    is_initialized: bool = False
    
    # Update tracking
    n_updates: int = 0
    last_update_time: Optional[int] = None


class PersonalBaselineMemory:
    """Personal Baseline Memory for z-score normalization.
    
    Each user has a separate baseline per modality. Baselines are initialized
    from the first N healthy bins and updated via robust EMA.
    """
    
    def __init__(
        self,
        feature_dims: dict[str, int],
        alpha: float = 0.03,
        beta: float = 0.02,
        safe_risk_threshold: float = 0.30,
        clip_z: float = 6.0,
        min_quality: float = 0.3,
        init_bins: int = 56,  # 14 days * 4 bins/day
        eps: float = 1e-6,
    ):
        """Initialize PBM.
        
        Args:
            feature_dims: Dict mapping modality name to feature dimension.
            alpha: EMA update rate for mean.
            beta: EMA update rate for variance.
            safe_risk_threshold: Max predicted risk to allow baseline update.
            clip_z: Z-score clipping value.
            min_quality: Minimum quality for baseline update.
            init_bins: Number of bins for initialization.
            eps: Small value for numerical stability.
        """
        self.feature_dims = feature_dims
        self.alpha = alpha
        self.beta = beta
        self.safe_risk_threshold = safe_risk_threshold
        self.clip_z = clip_z
        self.min_quality = min_quality
        self.init_bins = init_bins
        self.eps = eps
        
        # User baselines: user_id -> modality -> BaselineState
        self._baselines: dict[str, dict[str, BaselineState]] = {}
        
        # Initialization buffers: user_id -> modality -> list of features
        self._init_buffers: dict[str, dict[str, list]] = {}
    
    def get_user_modalities(self, user_id: str) -> list[str]:
        """Get list of modalities with baselines for a user."""
        if user_id not in self._baselines:
            return []
        return list(self._baselines[user_id].keys())
    
    def is_initialized(self, user_id: str, modality: str) -> bool:
        """Check if baseline is initialized for user/modality."""
        if user_id not in self._baselines:
            return False
        if modality not in self._baselines[user_id]:
            return False
        return self._baselines[user_id][modality].is_initialized
    
    def add_init_sample(
        self,
        user_id: str,
        modality: str,
        features: np.ndarray,
        quality: float = 1.0,
    ) -> bool:
        """Add sample to initialization buffer.
        
        Args:
            user_id: User identifier.
            modality: Modality name.
            features: Feature vector.
            quality: Quality score (0-1).
            
        Returns:
            True if baseline was just initialized.
        """
        if quality < self.min_quality:
            return False
        
        # Create buffers if needed
        if user_id not in self._init_buffers:
            self._init_buffers[user_id] = {}
        if modality not in self._init_buffers[user_id]:
            self._init_buffers[user_id][modality] = []
        
        # Add to buffer
        self._init_buffers[user_id][modality].append(features.copy())
        
        # Check if we have enough samples to initialize
        if len(self._init_buffers[user_id][modality]) >= self.init_bins:
            self._initialize_baseline(user_id, modality)
            return True
        
        return False
    
    def _initialize_baseline(self, user_id: str, modality: str) -> None:
        """Initialize baseline from buffer using median/MAD."""
        samples = np.array(self._init_buffers[user_id][modality])
        
        # Compute median and MAD for each dimension
        mu = np.median(samples, axis=0)
        mad = np.median(np.abs(samples - mu), axis=0)
        sigma = mad * 1.4826 + self.eps  # MAD to std approximation
        
        # Create baseline state
        if user_id not in self._baselines:
            self._baselines[user_id] = {}
        
        self._baselines[user_id][modality] = BaselineState(
            mu=mu,
            sigma=sigma,
            n_init_samples=len(samples),
            is_initialized=True,
        )
        
        # Clear buffer
        del self._init_buffers[user_id][modality]
    
    def compute_drift(
        self,
        user_id: str,
        modality: str,
        features: np.ndarray,
    ) -> np.ndarray:
        """Compute drift z-scores for features.
        
        CRITICAL: This only uses baseline computed from PAST data.
        
        Args:
            user_id: User identifier.
            modality: Modality name.
            features: Feature vector.
            
        Returns:
            Z-score drift vector (clipped).
        """
        if not self.is_initialized(user_id, modality):
            # Return zeros if not initialized
            return np.zeros_like(features)
        
        state = self._baselines[user_id][modality]
        
        # Compute z-scores
        z = (features - state.mu) / (state.sigma + self.eps)
        
        # Clip to prevent extreme values
        z = np.clip(z, -self.clip_z, self.clip_z)
        
        return z
    
    def update_baseline(
        self,
        user_id: str,
        modality: str,
        features: np.ndarray,
        quality: float,
        predicted_risk: float,
        is_labeled_sick: bool = False,
        time_index: Optional[int] = None,
    ) -> bool:
        """Update baseline using robust EMA.
        
        Updates only if conditions are met (safe bin):
        - Not labeled sick
        - Predicted risk below threshold
        - Quality above threshold
        
        Args:
            user_id: User identifier.
            modality: Modality name.
            features: Feature vector.
            quality: Quality score (0-1).
            predicted_risk: Maximum predicted risk (max of 24/48/72h).
            is_labeled_sick: Whether user is currently labeled as sick.
            time_index: Optional time index for tracking.
            
        Returns:
            True if baseline was updated.
        """
        if not self.is_initialized(user_id, modality):
            # Try to initialize
            return self.add_init_sample(user_id, modality, features, quality)
        
        # Check safe conditions
        if is_labeled_sick:
            return False
        if predicted_risk >= self.safe_risk_threshold:
            return False
        if quality < self.min_quality:
            return False
        
        state = self._baselines[user_id][modality]
        
        # Robust update using Huber-like approach
        # For mean: use clipped difference
        diff = features - state.mu
        huber_diff = np.clip(diff, -2 * state.sigma, 2 * state.sigma)
        state.mu = (1 - self.alpha) * state.mu + self.alpha * (state.mu + huber_diff)
        
        # For scale: robust update based on absolute deviation
        abs_diff = np.abs(features - state.mu)
        # Use current sigma as reference for robustness
        robust_scale = np.clip(abs_diff, 0, 3 * state.sigma)
        target_sigma = robust_scale * 1.4826 + self.eps
        state.sigma = (1 - self.beta) * state.sigma + self.beta * target_sigma
        
        # Update tracking
        state.n_updates += 1
        state.last_update_time = time_index
        
        return True
    
    def get_baseline(
        self,
        user_id: str,
        modality: str,
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Get current baseline (mu, sigma) for user/modality.
        
        Args:
            user_id: User identifier.
            modality: Modality name.
            
        Returns:
            Tuple of (mu, sigma) or None if not initialized.
        """
        if not self.is_initialized(user_id, modality):
            return None
        
        state = self._baselines[user_id][modality]
        return state.mu.copy(), state.sigma.copy()
    
    def set_baseline(
        self,
        user_id: str,
        modality: str,
        mu: np.ndarray,
        sigma: np.ndarray,
    ) -> None:
        """Set baseline directly (for loading saved state).
        
        Args:
            user_id: User identifier.
            modality: Modality name.
            mu: Mean vector.
            sigma: Scale vector.
        """
        if user_id not in self._baselines:
            self._baselines[user_id] = {}
        
        self._baselines[user_id][modality] = BaselineState(
            mu=mu.copy(),
            sigma=sigma.copy(),
            is_initialized=True,
        )
    
    def compute_drift_batch(
        self,
        user_id: str,
        features_dict: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Compute drift z-scores for multiple modalities.
        
        Args:
            user_id: User identifier.
            features_dict: Dict mapping modality to feature vector.
            
        Returns:
            Dict mapping modality to drift z-score vector.
        """
        drift_dict = {}
        for modality, features in features_dict.items():
            drift_dict[modality] = self.compute_drift(user_id, modality, features)
        return drift_dict
    
    def get_state_dict(self) -> dict:
        """Get serializable state dictionary."""
        state = {}
        for user_id, user_baselines in self._baselines.items():
            state[user_id] = {}
            for modality, baseline in user_baselines.items():
                state[user_id][modality] = {
                    "mu": baseline.mu.tolist(),
                    "sigma": baseline.sigma.tolist(),
                    "n_init_samples": baseline.n_init_samples,
                    "n_updates": baseline.n_updates,
                }
        return state
    
    def load_state_dict(self, state: dict) -> None:
        """Load state from dictionary."""
        self._baselines = {}
        for user_id, user_state in state.items():
            self._baselines[user_id] = {}
            for modality, baseline_state in user_state.items():
                self._baselines[user_id][modality] = BaselineState(
                    mu=np.array(baseline_state["mu"]),
                    sigma=np.array(baseline_state["sigma"]),
                    n_init_samples=baseline_state.get("n_init_samples", 0),
                    is_initialized=True,
                    n_updates=baseline_state.get("n_updates", 0),
                )
    
    def reset_user(self, user_id: str) -> None:
        """Reset all baselines for a user."""
        if user_id in self._baselines:
            del self._baselines[user_id]
        if user_id in self._init_buffers:
            del self._init_buffers[user_id]

