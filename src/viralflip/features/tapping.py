"""Tapping test feature extraction.

Features extracted from finger tapping test:
- Total tap count
- Inter-tap interval (ITI) statistics: mean, std, CV
- Outlier rate (ITIs beyond baseline percentile)
- Fatigue slope (tap rate decrease over time)

Quality metrics:
- Completion flag (whether test was completed properly)
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import linregress


@dataclass
class TappingFeatures:
    """Container for tapping test features."""
    
    tap_count_total: float
    iti_mean: float
    iti_std: float
    iti_cv: float
    outlier_rate: float
    fatigue_slope: float
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([
            self.tap_count_total,
            self.iti_mean,
            self.iti_std,
            self.iti_cv,
            self.outlier_rate,
            self.fatigue_slope,
        ], dtype=np.float32)


@dataclass
class TappingQuality:
    """Container for tapping quality metrics."""
    
    completion_flag: float  # 1.0 if completed, 0.0 if not
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.completion_flag], dtype=np.float32)


class TappingFeatureExtractor:
    """Extract features from finger tapping tests."""
    
    def __init__(
        self,
        min_taps: int = 10,
        duration_sec: float = 10.0,
        outlier_percentile: float = 95.0,
    ):
        """Initialize tapping feature extractor.
        
        Args:
            min_taps: Minimum number of taps for valid test.
            duration_sec: Expected test duration in seconds.
            outlier_percentile: Percentile for defining ITI outliers.
        """
        self.min_taps = min_taps
        self.duration_sec = duration_sec
        self.outlier_percentile = outlier_percentile
        self._baseline_iti_p95: float | None = None
    
    def set_baseline(self, baseline_iti_p95: float) -> None:
        """Set baseline ITI percentile for outlier detection.
        
        Args:
            baseline_iti_p95: 95th percentile of ITI from baseline period.
        """
        self._baseline_iti_p95 = baseline_iti_p95
    
    def extract(
        self,
        tap_timestamps: np.ndarray,
        test_duration_sec: float | None = None,
    ) -> tuple[TappingFeatures, TappingQuality]:
        """Extract tapping features from tap timestamps.
        
        Args:
            tap_timestamps: Array of tap timestamps in seconds from test start.
            test_duration_sec: Actual test duration (if different from expected).
            
        Returns:
            Tuple of (TappingFeatures, TappingQuality).
        """
        if test_duration_sec is None:
            test_duration_sec = self.duration_sec
        
        # Check for valid test
        if len(tap_timestamps) < self.min_taps:
            return self._empty_features(), TappingQuality(completion_flag=0.0)
        
        # Sort timestamps
        tap_timestamps = np.sort(tap_timestamps)
        
        # Compute inter-tap intervals
        itis = np.diff(tap_timestamps)
        
        if len(itis) == 0:
            return self._empty_features(), TappingQuality(completion_flag=0.0)
        
        # Basic ITI statistics
        tap_count = len(tap_timestamps)
        iti_mean = np.mean(itis)
        iti_std = np.std(itis)
        iti_cv = iti_std / (iti_mean + 1e-10)
        
        # Outlier rate
        if self._baseline_iti_p95 is not None:
            outlier_threshold = self._baseline_iti_p95
        else:
            # Use current test's 95th percentile as fallback
            outlier_threshold = np.percentile(itis, self.outlier_percentile)
        
        outlier_rate = np.mean(itis > outlier_threshold)
        
        # Fatigue slope: tap rate over time
        fatigue_slope = self._compute_fatigue_slope(tap_timestamps, test_duration_sec)
        
        # Check completion
        completion_flag = 1.0 if (
            len(tap_timestamps) >= self.min_taps and
            tap_timestamps[-1] <= test_duration_sec * 1.5  # Allow some slack
        ) else 0.0
        
        features = TappingFeatures(
            tap_count_total=float(tap_count),
            iti_mean=float(iti_mean),
            iti_std=float(iti_std),
            iti_cv=float(iti_cv),
            outlier_rate=float(outlier_rate),
            fatigue_slope=float(fatigue_slope),
        )
        
        quality = TappingQuality(completion_flag=completion_flag)
        
        return features, quality
    
    def extract_from_itis(
        self,
        itis: np.ndarray,
        total_taps: int | None = None,
    ) -> tuple[TappingFeatures, TappingQuality]:
        """Extract features from pre-computed inter-tap intervals.
        
        Args:
            itis: Array of inter-tap intervals in seconds.
            total_taps: Total number of taps (if known; otherwise len(itis)+1).
            
        Returns:
            Tuple of (TappingFeatures, TappingQuality).
        """
        if len(itis) < self.min_taps - 1:
            return self._empty_features(), TappingQuality(completion_flag=0.0)
        
        tap_count = total_taps if total_taps is not None else len(itis) + 1
        
        iti_mean = np.mean(itis)
        iti_std = np.std(itis)
        iti_cv = iti_std / (iti_mean + 1e-10)
        
        # Outlier rate
        if self._baseline_iti_p95 is not None:
            outlier_threshold = self._baseline_iti_p95
        else:
            outlier_threshold = np.percentile(itis, self.outlier_percentile)
        outlier_rate = np.mean(itis > outlier_threshold)
        
        # Fatigue: compute from ITI trend
        if len(itis) > 2:
            x = np.arange(len(itis))
            slope, _, _, _, _ = linregress(x, itis)
            # Positive slope means ITIs increasing = slowing down
            fatigue_slope = slope
        else:
            fatigue_slope = 0.0
        
        features = TappingFeatures(
            tap_count_total=float(tap_count),
            iti_mean=float(iti_mean),
            iti_std=float(iti_std),
            iti_cv=float(iti_cv),
            outlier_rate=float(outlier_rate),
            fatigue_slope=float(fatigue_slope),
        )
        
        quality = TappingQuality(completion_flag=1.0)
        
        return features, quality
    
    def _compute_fatigue_slope(
        self,
        tap_timestamps: np.ndarray,
        duration_sec: float,
    ) -> float:
        """Compute fatigue slope from tap rate over time.
        
        Negative slope indicates fatigue (slowing down).
        """
        if len(tap_timestamps) < 4:
            return 0.0
        
        # Divide into windows
        n_windows = 5
        window_duration = duration_sec / n_windows
        
        window_rates = []
        for i in range(n_windows):
            window_start = i * window_duration
            window_end = (i + 1) * window_duration
            
            taps_in_window = np.sum(
                (tap_timestamps >= window_start) & (tap_timestamps < window_end)
            )
            rate = taps_in_window / window_duration
            window_rates.append(rate)
        
        window_rates = np.array(window_rates)
        
        # Linear regression
        x = np.arange(n_windows)
        slope, _, _, _, _ = linregress(x, window_rates)
        
        return float(slope)
    
    def _empty_features(self) -> TappingFeatures:
        """Return empty features."""
        return TappingFeatures(
            tap_count_total=0.0,
            iti_mean=0.0,
            iti_std=0.0,
            iti_cv=0.0,
            outlier_rate=0.0,
            fatigue_slope=0.0,
        )

