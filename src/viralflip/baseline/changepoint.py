"""Change-Point Aware Dynamic Baselines.

Rather than one EMA baseline, maintain multiple baselines and detect
distribution shifts. When a change point triggers:
1. Freeze updates to current baseline
2. Start a new baseline bank
3. Use transition-aware smoothing

This pairs well with conformal prediction to maintain calibration
across routine changes (new school schedule, travel, new phone, etc.)
"""

from dataclasses import dataclass, field
from typing import Optional
from collections import deque

import numpy as np


@dataclass
class BaselineBank:
    """A single baseline bank with its own statistics."""
    
    # Mean and variance estimates
    mu: np.ndarray
    sigma: np.ndarray
    
    # When this bank started
    start_idx: int
    
    # Number of updates
    n_updates: int = 0
    
    # Is this bank active (receiving updates)?
    is_active: bool = True
    
    # Confidence in this bank (based on stability)
    confidence: float = 1.0


@dataclass 
class ChangePointEvent:
    """Record of a detected change point."""
    
    # When detected
    time_idx: int
    
    # Change statistics
    z_score: float
    affected_features: list[int] = field(default_factory=list)
    
    # Type of change
    change_type: str = "unknown"  # 'gradual', 'abrupt', 'device'


class ChangePointDetector:
    """Detects distribution shifts in time series data.
    
    Uses multiple detection methods:
    - CUSUM (cumulative sum)
    - Page-Hinkley test
    - Feature-wise monitoring
    """
    
    def __init__(
        self,
        n_features: int,
        cusum_threshold: float = 4.0,
        ph_threshold: float = 50.0,
        window_size: int = 50,
        min_samples_between: int = 20,
    ):
        """Initialize detector.
        
        Args:
            n_features: Number of features to monitor
            cusum_threshold: Threshold for CUSUM detection
            ph_threshold: Threshold for Page-Hinkley detection
            window_size: Window for statistics computation
            min_samples_between: Minimum samples between change points
        """
        self.n_features = n_features
        self.cusum_threshold = cusum_threshold
        self.ph_threshold = ph_threshold
        self.window_size = window_size
        self.min_samples_between = min_samples_between
        
        # Running statistics
        self.running_mean = np.zeros(n_features)
        self.running_var = np.ones(n_features)
        self.n_samples = 0
        
        # CUSUM state
        self.cusum_pos = np.zeros(n_features)
        self.cusum_neg = np.zeros(n_features)
        
        # Page-Hinkley state
        self.ph_sum = 0.0
        self.ph_min = 0.0
        
        # History
        self.history = deque(maxlen=window_size)
        self.last_change_idx = -min_samples_between
    
    def update(
        self,
        x: np.ndarray,
        time_idx: int,
    ) -> Optional[ChangePointEvent]:
        """Update detector with new sample.
        
        Args:
            x: Feature vector
            time_idx: Current time index
            
        Returns:
            ChangePointEvent if change detected, else None
        """
        self.history.append(x)
        
        # Update running statistics
        self.n_samples += 1
        if self.n_samples == 1:
            self.running_mean = x.copy()
            self.running_var = np.ones(self.n_features)
        else:
            delta = x - self.running_mean
            self.running_mean += delta / self.n_samples
            delta2 = x - self.running_mean
            self.running_var = (
                (self.n_samples - 1) * self.running_var + delta * delta2
            ) / self.n_samples
        
        # Check if enough samples since last change
        if time_idx - self.last_change_idx < self.min_samples_between:
            return None
        
        # Run detectors
        cusum_change = self._cusum_update(x)
        ph_change = self._page_hinkley_update(x)
        
        if cusum_change is not None or ph_change:
            # Find which features changed most
            affected = self._identify_affected_features(x)
            
            # Classify change type
            change_type = self._classify_change()
            
            event = ChangePointEvent(
                time_idx=time_idx,
                z_score=float(np.max(np.abs(x - self.running_mean) / (np.sqrt(self.running_var) + 1e-6))),
                affected_features=affected,
                change_type=change_type,
            )
            
            # Reset detectors
            self._reset_detectors()
            self.last_change_idx = time_idx
            
            return event
        
        return None
    
    def _cusum_update(self, x: np.ndarray) -> Optional[np.ndarray]:
        """CUSUM update and detection."""
        std = np.sqrt(self.running_var) + 1e-6
        z = (x - self.running_mean) / std
        
        # Update CUSUM statistics
        self.cusum_pos = np.maximum(0, self.cusum_pos + z - 0.5)
        self.cusum_neg = np.maximum(0, self.cusum_neg - z - 0.5)
        
        # Check for change
        if np.any(self.cusum_pos > self.cusum_threshold) or \
           np.any(self.cusum_neg > self.cusum_threshold):
            return z
        
        return None
    
    def _page_hinkley_update(self, x: np.ndarray) -> bool:
        """Page-Hinkley test update."""
        # Use mean of features
        x_mean = np.mean(x)
        
        self.ph_sum += x_mean - self.running_mean.mean() - 0.005
        self.ph_min = min(self.ph_min, self.ph_sum)
        
        return (self.ph_sum - self.ph_min) > self.ph_threshold
    
    def _identify_affected_features(self, x: np.ndarray) -> list[int]:
        """Identify which features changed most."""
        std = np.sqrt(self.running_var) + 1e-6
        z = np.abs(x - self.running_mean) / std
        
        # Features with z-score > 2
        affected = np.where(z > 2.0)[0].tolist()
        
        # If none, return top 3
        if not affected:
            affected = np.argsort(z)[-3:].tolist()
        
        return affected
    
    def _classify_change(self) -> str:
        """Classify type of change."""
        if len(self.history) < 10:
            return "abrupt"
        
        # Check if change was gradual (trend in last N samples)
        recent = np.array(list(self.history))[-10:]
        if len(recent) < 10:
            return "abrupt"
        
        # Simple linear trend detection
        x_range = np.arange(len(recent))
        correlations = []
        for i in range(recent.shape[1]):
            corr = np.corrcoef(x_range, recent[:, i])[0, 1]
            correlations.append(abs(corr))
        
        if np.mean(correlations) > 0.7:
            return "gradual"
        
        return "abrupt"
    
    def _reset_detectors(self) -> None:
        """Reset detector state after change point."""
        self.cusum_pos = np.zeros(self.n_features)
        self.cusum_neg = np.zeros(self.n_features)
        self.ph_sum = 0.0
        self.ph_min = 0.0


class ChangePointAwareBaseline:
    """Personal baseline with change-point detection and multiple banks.
    
    Maintains multiple baseline banks and switches between them
    based on detected regime changes.
    """
    
    def __init__(
        self,
        n_features: int,
        alpha: float = 0.03,
        beta: float = 0.02,
        safe_risk_threshold: float = 0.30,
        clip_z: float = 6.0,
        init_samples: int = 56,
        max_banks: int = 5,
        bank_merge_threshold: float = 0.5,
        change_sensitivity: float = 4.0,
    ):
        """Initialize change-point aware baseline.
        
        Args:
            n_features: Number of features
            alpha: EMA rate for mean
            beta: EMA rate for variance
            safe_risk_threshold: Max risk for baseline update
            clip_z: Z-score clipping value
            init_samples: Samples for initialization
            max_banks: Maximum number of baseline banks
            bank_merge_threshold: Threshold for merging similar banks
            change_sensitivity: Sensitivity for change detection
        """
        self.n_features = n_features
        self.alpha = alpha
        self.beta = beta
        self.safe_risk_threshold = safe_risk_threshold
        self.clip_z = clip_z
        self.init_samples = init_samples
        self.max_banks = max_banks
        self.bank_merge_threshold = bank_merge_threshold
        
        # Banks
        self.banks: list[BaselineBank] = []
        self.active_bank_idx: int = -1
        
        # Change point detector
        self.detector = ChangePointDetector(
            n_features=n_features,
            cusum_threshold=change_sensitivity,
        )
        
        # Initialization buffer
        self.init_buffer: list[np.ndarray] = []
        self.is_initialized = False
        
        # History
        self.change_history: list[ChangePointEvent] = []
        self.n_updates = 0
    
    def add_init_sample(
        self,
        features: np.ndarray,
        quality: float = 1.0,
    ) -> bool:
        """Add sample to initialization buffer.
        
        Args:
            features: Feature vector
            quality: Quality score
            
        Returns:
            True if just initialized
        """
        if quality < 0.3:
            return False
        
        self.init_buffer.append(features.copy())
        
        if len(self.init_buffer) >= self.init_samples:
            self._initialize_first_bank()
            return True
        
        return False
    
    def _initialize_first_bank(self) -> None:
        """Initialize first baseline bank."""
        samples = np.array(self.init_buffer)
        
        mu = np.median(samples, axis=0)
        mad = np.median(np.abs(samples - mu), axis=0)
        sigma = mad * 1.4826 + 1e-6
        
        bank = BaselineBank(
            mu=mu,
            sigma=sigma,
            start_idx=0,
            n_updates=len(samples),
            is_active=True,
        )
        
        self.banks.append(bank)
        self.active_bank_idx = 0
        self.is_initialized = True
        self.init_buffer = []
    
    def compute_drift(self, features: np.ndarray) -> np.ndarray:
        """Compute drift z-scores.
        
        Args:
            features: Feature vector
            
        Returns:
            Z-score drift vector
        """
        if not self.is_initialized:
            return np.zeros(self.n_features)
        
        bank = self.banks[self.active_bank_idx]
        
        z = (features - bank.mu) / (bank.sigma + 1e-6)
        z = np.clip(z, -self.clip_z, self.clip_z)
        
        return z
    
    def update(
        self,
        features: np.ndarray,
        quality: float,
        predicted_risk: float,
        time_idx: int,
        is_labeled_sick: bool = False,
    ) -> tuple[bool, Optional[ChangePointEvent]]:
        """Update baseline with new sample.
        
        Args:
            features: Feature vector
            quality: Quality score
            predicted_risk: Current risk prediction
            time_idx: Time index
            is_labeled_sick: Whether labeled as sick
            
        Returns:
            Tuple of (was_updated, change_event if detected)
        """
        if not self.is_initialized:
            init_done = self.add_init_sample(features, quality)
            return init_done, None
        
        # Check for change point
        change_event = self.detector.update(features, time_idx)
        
        if change_event is not None:
            # Handle change point
            self._handle_change_point(change_event, features, time_idx)
            self.change_history.append(change_event)
            return False, change_event
        
        # Check if safe to update
        if is_labeled_sick:
            return False, None
        if predicted_risk >= self.safe_risk_threshold:
            return False, None
        if quality < 0.3:
            return False, None
        
        # Update active bank
        bank = self.banks[self.active_bank_idx]
        if not bank.is_active:
            return False, None
        
        self._update_bank(bank, features)
        self.n_updates += 1
        
        return True, None
    
    def _update_bank(self, bank: BaselineBank, features: np.ndarray) -> None:
        """Update a baseline bank with EMA."""
        # Robust update for mean
        diff = features - bank.mu
        huber_diff = np.clip(diff, -2 * bank.sigma, 2 * bank.sigma)
        bank.mu = (1 - self.alpha) * bank.mu + self.alpha * (bank.mu + huber_diff)
        
        # Robust update for scale
        abs_diff = np.abs(features - bank.mu)
        robust_scale = np.clip(abs_diff, 0, 3 * bank.sigma)
        target_sigma = robust_scale * 1.4826 + 1e-6
        bank.sigma = (1 - self.beta) * bank.sigma + self.beta * target_sigma
        
        bank.n_updates += 1
    
    def _handle_change_point(
        self,
        event: ChangePointEvent,
        features: np.ndarray,
        time_idx: int,
    ) -> None:
        """Handle detected change point."""
        # Freeze current bank
        if self.active_bank_idx >= 0:
            self.banks[self.active_bank_idx].is_active = False
        
        # Check if we can reuse an old bank
        reuse_idx = self._find_similar_bank(features)
        
        if reuse_idx is not None:
            # Reactivate similar bank
            self.banks[reuse_idx].is_active = True
            self.active_bank_idx = reuse_idx
        else:
            # Create new bank
            self._create_new_bank(features, time_idx)
        
        # Merge or prune banks if needed
        self._manage_banks()
    
    def _find_similar_bank(self, features: np.ndarray) -> Optional[int]:
        """Find existing bank similar to current features."""
        best_idx = None
        best_distance = float('inf')
        
        for i, bank in enumerate(self.banks):
            if i == self.active_bank_idx:
                continue
            
            # Mahalanobis-like distance
            z = (features - bank.mu) / (bank.sigma + 1e-6)
            distance = np.mean(np.abs(z))
            
            if distance < self.bank_merge_threshold and distance < best_distance:
                best_distance = distance
                best_idx = i
        
        return best_idx
    
    def _create_new_bank(self, features: np.ndarray, time_idx: int) -> None:
        """Create a new baseline bank."""
        # Initialize with current features as starting point
        # Use larger initial sigma for uncertainty
        bank = BaselineBank(
            mu=features.copy(),
            sigma=np.ones(self.n_features) * 0.5,
            start_idx=time_idx,
            n_updates=1,
            is_active=True,
            confidence=0.5,  # Lower confidence for new bank
        )
        
        self.banks.append(bank)
        self.active_bank_idx = len(self.banks) - 1
    
    def _manage_banks(self) -> None:
        """Manage bank count - merge or prune if needed."""
        if len(self.banks) <= self.max_banks:
            return
        
        # Find least used inactive bank
        min_updates = float('inf')
        min_idx = None
        
        for i, bank in enumerate(self.banks):
            if not bank.is_active and bank.n_updates < min_updates:
                min_updates = bank.n_updates
                min_idx = i
        
        if min_idx is not None:
            del self.banks[min_idx]
            if self.active_bank_idx > min_idx:
                self.active_bank_idx -= 1
    
    def get_baseline(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Get current baseline (mu, sigma)."""
        if not self.is_initialized or self.active_bank_idx < 0:
            return None
        
        bank = self.banks[self.active_bank_idx]
        return bank.mu.copy(), bank.sigma.copy()
    
    def get_uncertainty_boost(self) -> float:
        """Get uncertainty boost factor based on bank confidence."""
        if not self.is_initialized or self.active_bank_idx < 0:
            return 2.0  # High uncertainty if not initialized
        
        bank = self.banks[self.active_bank_idx]
        
        # Lower confidence = higher boost
        base_boost = 2.0 - bank.confidence
        
        # Fewer updates = higher boost
        update_factor = min(1.0, bank.n_updates / 50)
        
        return base_boost * (2.0 - update_factor)
    
    def get_state_summary(self) -> dict:
        """Get summary of baseline state."""
        return {
            "n_banks": len(self.banks),
            "active_bank_idx": self.active_bank_idx,
            "n_changes": len(self.change_history),
            "is_initialized": self.is_initialized,
            "n_updates": self.n_updates,
            "banks": [
                {
                    "start_idx": b.start_idx,
                    "n_updates": b.n_updates,
                    "is_active": b.is_active,
                    "confidence": b.confidence,
                }
                for b in self.banks
            ],
        }


class MultiModalityChangePointBaseline:
    """Change-point aware baseline for multiple modalities."""
    
    def __init__(
        self,
        modality_dims: dict[str, int],
        alpha: float = 0.03,
        beta: float = 0.02,
        safe_risk_threshold: float = 0.30,
        sync_change_detection: bool = True,
    ):
        """Initialize multi-modality baseline.
        
        Args:
            modality_dims: Dict mapping modality to feature dimension
            alpha: EMA rate for mean
            beta: EMA rate for variance
            safe_risk_threshold: Max risk for updates
            sync_change_detection: Whether to sync changes across modalities
        """
        self.modality_dims = modality_dims
        self.modalities = list(modality_dims.keys())
        self.sync_change_detection = sync_change_detection
        
        # Per-modality baselines
        self.baselines = {
            mod: ChangePointAwareBaseline(
                n_features=dim,
                alpha=alpha,
                beta=beta,
                safe_risk_threshold=safe_risk_threshold,
            )
            for mod, dim in modality_dims.items()
        }
        
        # Global change tracking
        self.global_changes: list[ChangePointEvent] = []
    
    def compute_drift_batch(
        self,
        features_dict: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Compute drift for multiple modalities.
        
        Args:
            features_dict: Dict mapping modality to features
            
        Returns:
            Dict mapping modality to drift z-scores
        """
        return {
            mod: self.baselines[mod].compute_drift(features_dict[mod])
            for mod in self.modalities
            if mod in features_dict
        }
    
    def update(
        self,
        features_dict: dict[str, np.ndarray],
        quality_dict: dict[str, float],
        predicted_risk: float,
        time_idx: int,
        is_labeled_sick: bool = False,
    ) -> dict[str, Optional[ChangePointEvent]]:
        """Update all baselines.
        
        Args:
            features_dict: Dict mapping modality to features
            quality_dict: Dict mapping modality to quality
            predicted_risk: Current risk prediction
            time_idx: Time index
            is_labeled_sick: Whether labeled as sick
            
        Returns:
            Dict mapping modality to change event if detected
        """
        events = {}
        any_change = False
        
        for mod in self.modalities:
            if mod not in features_dict:
                events[mod] = None
                continue
            
            _, event = self.baselines[mod].update(
                features=features_dict[mod],
                quality=quality_dict.get(mod, 1.0),
                predicted_risk=predicted_risk,
                time_idx=time_idx,
                is_labeled_sick=is_labeled_sick,
            )
            
            events[mod] = event
            if event is not None:
                any_change = True
        
        # If syncing and any modality changed, record global change
        if self.sync_change_detection and any_change:
            # Use first detected change as representative
            for mod, event in events.items():
                if event is not None:
                    self.global_changes.append(event)
                    break
        
        return events
    
    def get_uncertainty_boost(self) -> dict[str, float]:
        """Get uncertainty boosts for all modalities."""
        return {
            mod: baseline.get_uncertainty_boost()
            for mod, baseline in self.baselines.items()
        }
    
    def is_in_transition(self) -> bool:
        """Check if any modality is in transition period."""
        return any(
            baseline.get_uncertainty_boost() > 1.5
            for baseline in self.baselines.values()
        )

