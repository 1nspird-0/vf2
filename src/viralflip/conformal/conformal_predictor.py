"""Conformal Prediction for calibrated uncertainty.

Conformal methods provide coverage-guaranteed uncertainty:
- P(y ∈ C(x)) ≥ 1 - α for any distribution

This is especially valuable for health alerts where we need:
1. Reliable uncertainty bounds
2. Coverage guarantees under distribution shift
3. Change-point aware calibration

Reference: Conformal Prediction for Time-series Forecasting with Change Points (OpenReview 2024)
"""

from dataclasses import dataclass
from typing import Optional, Union
import math

import numpy as np
import torch
import torch.nn as nn


@dataclass
class ConformalRiskSet:
    """Conformal prediction set for risk."""
    
    # Point prediction
    risk: float
    
    # Conformal interval
    lower: float
    upper: float
    
    # Coverage level (1 - alpha)
    coverage: float
    
    # Nonconformity score
    score: float
    
    # Whether to alert (risk exceeds threshold with confidence)
    should_alert: bool
    
    # Confidence in alert decision
    alert_confidence: float
    
    def is_safe_to_alert(self, threshold: float) -> bool:
        """Check if safe to alert (lower bound exceeds threshold)."""
        return self.lower > threshold


class ConformalPredictor:
    """Standard split conformal predictor for regression/risk scores.
    
    Uses calibration set to compute quantile of nonconformity scores,
    then applies to test predictions for prediction intervals.
    """
    
    def __init__(
        self,
        alpha: float = 0.1,
        score_fn: str = "absolute",
    ):
        """Initialize conformal predictor.
        
        Args:
            alpha: Miscoverage rate (0.1 = 90% coverage)
            score_fn: Nonconformity score type ('absolute', 'quantile')
        """
        self.alpha = alpha
        self.score_fn = score_fn
        
        # Calibration data
        self.calibration_scores: Optional[np.ndarray] = None
        self.quantile: Optional[float] = None
        
        self._is_calibrated = False
    
    def calibrate(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> float:
        """Calibrate on held-out data.
        
        Args:
            predictions: Model predictions, shape (n_samples,)
            targets: True targets, shape (n_samples,)
            weights: Optional sample weights
            
        Returns:
            Calibration quantile
        """
        # Compute nonconformity scores
        scores = self._compute_scores(predictions, targets)
        
        if weights is not None:
            # Weighted quantile
            sorted_idx = np.argsort(scores)
            sorted_scores = scores[sorted_idx]
            sorted_weights = weights[sorted_idx]
            
            cumsum = np.cumsum(sorted_weights)
            cumsum /= cumsum[-1]
            
            quantile_idx = np.searchsorted(cumsum, 1 - self.alpha)
            self.quantile = sorted_scores[min(quantile_idx, len(sorted_scores) - 1)]
        else:
            # Standard quantile with finite sample correction
            n = len(scores)
            q = np.ceil((n + 1) * (1 - self.alpha)) / n
            q = min(q, 1.0)
            self.quantile = np.quantile(scores, q)
        
        self.calibration_scores = scores
        self._is_calibrated = True
        
        return self.quantile
    
    def predict(
        self,
        predictions: np.ndarray,
        return_scores: bool = False,
    ) -> Union[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Generate prediction intervals.
        
        Args:
            predictions: Model predictions, shape (n_samples,)
            return_scores: Whether to return nonconformity scores
            
        Returns:
            lower: Lower bounds
            upper: Upper bounds
            scores: (optional) Nonconformity scores
        """
        if not self._is_calibrated:
            raise RuntimeError("Predictor not calibrated. Call calibrate() first.")
        
        lower = predictions - self.quantile
        upper = predictions + self.quantile
        
        # Clamp to valid probability range
        lower = np.clip(lower, 0, 1)
        upper = np.clip(upper, 0, 1)
        
        if return_scores:
            # For test time, score is just the prediction itself
            scores = np.abs(predictions - 0.5)  # Distance from uncertain
            return lower, upper, scores
        
        return lower, upper
    
    def predict_single(
        self,
        prediction: float,
        alert_threshold: float = 0.3,
    ) -> ConformalRiskSet:
        """Generate conformal risk set for single prediction.
        
        Args:
            prediction: Risk prediction (0-1)
            alert_threshold: Threshold for alerting
            
        Returns:
            ConformalRiskSet with bounds and alert decision
        """
        if not self._is_calibrated:
            raise RuntimeError("Predictor not calibrated.")
        
        lower = max(0, prediction - self.quantile)
        upper = min(1, prediction + self.quantile)
        
        # Alert if lower bound exceeds threshold
        should_alert = lower > alert_threshold
        
        # Confidence based on how much lower bound exceeds threshold
        if should_alert:
            alert_confidence = min(1.0, (lower - alert_threshold) / (1 - alert_threshold))
        else:
            alert_confidence = 0.0
        
        return ConformalRiskSet(
            risk=prediction,
            lower=lower,
            upper=upper,
            coverage=1 - self.alpha,
            score=abs(prediction - 0.5),
            should_alert=should_alert,
            alert_confidence=alert_confidence,
        )
    
    def _compute_scores(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> np.ndarray:
        """Compute nonconformity scores."""
        if self.score_fn == "absolute":
            return np.abs(predictions - targets)
        elif self.score_fn == "quantile":
            # Asymmetric score for quantile regression
            return np.maximum(
                self.alpha * (predictions - targets),
                (1 - self.alpha) * (targets - predictions)
            )
        else:
            raise ValueError(f"Unknown score function: {self.score_fn}")
    
    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated


class AdaptiveConformalPredictor:
    """Adaptive Conformal Inference (ACI) for time series.
    
    Updates calibration online to maintain coverage under distribution shift.
    Uses a learning rate to adapt quantile based on recent coverage.
    """
    
    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.05,
        initial_quantile: float = 0.5,
        min_quantile: float = 0.01,
        max_quantile: float = 2.0,
    ):
        """Initialize adaptive conformal predictor.
        
        Args:
            alpha: Target miscoverage rate
            gamma: Learning rate for quantile updates
            initial_quantile: Initial quantile value
            min_quantile: Minimum allowed quantile
            max_quantile: Maximum allowed quantile
        """
        self.alpha = alpha
        self.gamma = gamma
        self.quantile = initial_quantile
        self.min_quantile = min_quantile
        self.max_quantile = max_quantile
        
        # Tracking
        self.coverage_history = []
        self.quantile_history = [initial_quantile]
    
    def update(
        self,
        prediction: float,
        target: float,
    ) -> tuple[float, float]:
        """Update quantile based on observed target.
        
        Args:
            prediction: Model prediction
            target: Observed target
            
        Returns:
            Tuple of (new_lower, new_upper) bounds
        """
        # Check if target was covered
        lower = prediction - self.quantile
        upper = prediction + self.quantile
        covered = (target >= lower) and (target <= upper)
        
        self.coverage_history.append(covered)
        
        # Update quantile
        # If covered, decrease quantile; if not, increase
        if covered:
            self.quantile = self.quantile - self.gamma * self.alpha
        else:
            self.quantile = self.quantile + self.gamma * (1 - self.alpha)
        
        # Clamp to bounds
        self.quantile = np.clip(self.quantile, self.min_quantile, self.max_quantile)
        self.quantile_history.append(self.quantile)
        
        return lower, upper
    
    def predict(self, prediction: float) -> tuple[float, float]:
        """Generate prediction interval.
        
        Args:
            prediction: Model prediction
            
        Returns:
            Tuple of (lower, upper) bounds
        """
        lower = max(0, prediction - self.quantile)
        upper = min(1, prediction + self.quantile)
        return lower, upper
    
    def predict_single(
        self,
        prediction: float,
        alert_threshold: float = 0.3,
    ) -> ConformalRiskSet:
        """Generate conformal risk set."""
        lower, upper = self.predict(prediction)
        
        should_alert = lower > alert_threshold
        alert_confidence = max(0, (lower - alert_threshold)) if should_alert else 0
        
        return ConformalRiskSet(
            risk=prediction,
            lower=lower,
            upper=upper,
            coverage=1 - self.alpha,
            score=abs(prediction - 0.5),
            should_alert=should_alert,
            alert_confidence=alert_confidence,
        )
    
    def get_recent_coverage(self, window: int = 100) -> float:
        """Get coverage over recent predictions."""
        if not self.coverage_history:
            return 1 - self.alpha
        
        recent = self.coverage_history[-window:]
        return np.mean(recent)
    
    def reset(self, initial_quantile: Optional[float] = None) -> None:
        """Reset to initial state."""
        if initial_quantile is not None:
            self.quantile = initial_quantile
        else:
            self.quantile = self.quantile_history[0]
        
        self.coverage_history = []
        self.quantile_history = [self.quantile]


class ChangePointAwareConformal:
    """Conformal prediction with change-point detection.
    
    Detects distribution shifts (routine changes) and:
    1. Resets calibration after change points
    2. Uses shorter calibration windows after shifts
    3. Increases uncertainty during transition periods
    
    This is crucial for mobile health where routines change
    (new school schedule, travel, new phone, etc.)
    """
    
    def __init__(
        self,
        alpha: float = 0.1,
        base_window: int = 100,
        min_window: int = 20,
        change_threshold: float = 2.0,
        recovery_rate: float = 0.05,
        uncertainty_boost: float = 1.5,
    ):
        """Initialize change-point aware conformal predictor.
        
        Args:
            alpha: Target miscoverage rate
            base_window: Base calibration window size
            min_window: Minimum window after change point
            change_threshold: Z-score threshold for change detection
            recovery_rate: Rate at which window recovers after change
            uncertainty_boost: Multiplier for uncertainty after change
        """
        self.alpha = alpha
        self.base_window = base_window
        self.min_window = min_window
        self.change_threshold = change_threshold
        self.recovery_rate = recovery_rate
        self.uncertainty_boost = uncertainty_boost
        
        # State
        self.current_window = base_window
        self.current_boost = 1.0
        
        # History for change detection
        self.score_history = []
        self.prediction_history = []
        self.change_points = []
        
        # Calibration
        self.quantile = 0.5
    
    def detect_change(
        self,
        new_score: float,
    ) -> bool:
        """Detect if a change point occurred.
        
        Uses cumulative sum (CUSUM) style detection on nonconformity scores.
        
        Args:
            new_score: New nonconformity score
            
        Returns:
            True if change detected
        """
        if len(self.score_history) < self.min_window:
            return False
        
        # Compare to running statistics
        recent = np.array(self.score_history[-self.base_window:])
        mean = np.mean(recent)
        std = np.std(recent) + 1e-6
        
        z_score = abs(new_score - mean) / std
        
        return z_score > self.change_threshold
    
    def update(
        self,
        prediction: float,
        target: float,
    ) -> tuple[float, float, bool]:
        """Update with observed target.
        
        Args:
            prediction: Model prediction
            target: Observed target
            
        Returns:
            Tuple of (lower, upper, change_detected)
        """
        # Compute score
        score = abs(prediction - target)
        
        # Check for change point
        change_detected = self.detect_change(score)
        
        if change_detected:
            # Reset window and boost uncertainty
            self.current_window = self.min_window
            self.current_boost = self.uncertainty_boost
            self.change_points.append(len(self.score_history))
        else:
            # Gradually recover window and boost
            self.current_window = min(
                self.base_window,
                self.current_window + int(self.recovery_rate * self.base_window)
            )
            self.current_boost = max(1.0, self.current_boost - self.recovery_rate)
        
        # Update history
        self.score_history.append(score)
        self.prediction_history.append(prediction)
        
        # Recalibrate
        self._calibrate()
        
        # Generate interval
        lower, upper = self.predict(prediction)
        
        return lower, upper, change_detected
    
    def _calibrate(self) -> None:
        """Recalibrate quantile using current window."""
        if len(self.score_history) < 5:
            return
        
        scores = np.array(self.score_history[-self.current_window:])
        n = len(scores)
        
        # Conformal quantile with finite sample correction
        q = np.ceil((n + 1) * (1 - self.alpha)) / n
        q = min(q, 1.0)
        
        self.quantile = np.quantile(scores, q) * self.current_boost
    
    def predict(self, prediction: float) -> tuple[float, float]:
        """Generate prediction interval.
        
        Args:
            prediction: Model prediction
            
        Returns:
            Tuple of (lower, upper) bounds
        """
        lower = max(0, prediction - self.quantile)
        upper = min(1, prediction + self.quantile)
        return lower, upper
    
    def predict_single(
        self,
        prediction: float,
        alert_threshold: float = 0.3,
    ) -> ConformalRiskSet:
        """Generate conformal risk set."""
        lower, upper = self.predict(prediction)
        
        should_alert = lower > alert_threshold
        alert_confidence = max(0, (lower - alert_threshold)) if should_alert else 0
        
        return ConformalRiskSet(
            risk=prediction,
            lower=lower,
            upper=upper,
            coverage=1 - self.alpha,
            score=abs(prediction - 0.5),
            should_alert=should_alert,
            alert_confidence=alert_confidence,
        )
    
    def get_state(self) -> dict:
        """Get current state for serialization."""
        return {
            "quantile": self.quantile,
            "current_window": self.current_window,
            "current_boost": self.current_boost,
            "change_points": self.change_points,
            "n_samples": len(self.score_history),
        }
    
    def is_in_transition(self) -> bool:
        """Check if currently in transition period after change."""
        return self.current_boost > 1.0 or self.current_window < self.base_window


class MultiHorizonConformal:
    """Conformal predictor for multiple prediction horizons.
    
    Handles correlated predictions across horizons (24h, 48h, 72h).
    """
    
    def __init__(
        self,
        horizons: list[int],
        alpha: float = 0.1,
        use_changepoint: bool = True,
    ):
        """Initialize multi-horizon conformal predictor.
        
        Args:
            horizons: List of prediction horizons
            alpha: Miscoverage rate per horizon
            use_changepoint: Whether to use change-point aware calibration
        """
        self.horizons = horizons
        self.alpha = alpha
        
        # Per-horizon predictors
        if use_changepoint:
            self.predictors = {
                h: ChangePointAwareConformal(alpha=alpha)
                for h in horizons
            }
        else:
            self.predictors = {
                h: AdaptiveConformalPredictor(alpha=alpha)
                for h in horizons
            }
    
    def calibrate(
        self,
        predictions: dict[int, np.ndarray],
        targets: dict[int, np.ndarray],
    ) -> dict[int, float]:
        """Calibrate all horizons.
        
        Args:
            predictions: Dict mapping horizon to predictions
            targets: Dict mapping horizon to targets
            
        Returns:
            Dict mapping horizon to quantile
        """
        quantiles = {}
        
        for h in self.horizons:
            if h in predictions and h in targets:
                if hasattr(self.predictors[h], 'calibrate'):
                    quantiles[h] = self.predictors[h].calibrate(
                        predictions[h], targets[h]
                    )
        
        return quantiles
    
    def predict(
        self,
        predictions: dict[int, float],
        alert_threshold: float = 0.3,
    ) -> dict[int, ConformalRiskSet]:
        """Generate conformal risk sets for all horizons.
        
        Args:
            predictions: Dict mapping horizon to prediction
            alert_threshold: Alert threshold
            
        Returns:
            Dict mapping horizon to ConformalRiskSet
        """
        results = {}
        
        for h in self.horizons:
            if h in predictions:
                results[h] = self.predictors[h].predict_single(
                    predictions[h], alert_threshold
                )
        
        return results
    
    def should_alert(
        self,
        predictions: dict[int, float],
        threshold: float = 0.3,
        require_all: bool = False,
    ) -> tuple[bool, dict[int, bool]]:
        """Determine if alert should be triggered.
        
        Args:
            predictions: Dict mapping horizon to prediction
            threshold: Alert threshold
            require_all: If True, require all horizons to alert
            
        Returns:
            Tuple of (overall_alert, per_horizon_alerts)
        """
        risk_sets = self.predict(predictions, threshold)
        
        per_horizon = {
            h: rs.should_alert for h, rs in risk_sets.items()
        }
        
        if require_all:
            overall = all(per_horizon.values())
        else:
            overall = any(per_horizon.values())
        
        return overall, per_horizon

