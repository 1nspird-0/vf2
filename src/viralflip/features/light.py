"""Ambient light sensor feature extraction.

Features extracted:
- Mean lux
- Lux standard deviation
- Evening lux mean (18:00-24:00)
- Circadian regularity score

Quality metrics:
- Light sensor uptime fraction
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class LightFeatures:
    """Container for ambient light features."""
    
    lux_mean: float
    lux_std: float
    lux_evening_mean: float
    circadian_regular_score: float
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([
            self.lux_mean,
            self.lux_std,
            self.lux_evening_mean,
            self.circadian_regular_score,
        ], dtype=np.float32)


@dataclass
class LightQuality:
    """Container for light quality metrics."""
    
    light_uptime_fraction: float
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.light_uptime_fraction], dtype=np.float32)


class LightFeatureExtractor:
    """Extract features from ambient light sensor data."""
    
    def __init__(
        self,
        evening_start_hour: int = 18,
        evening_end_hour: int = 24,
    ):
        """Initialize light feature extractor.
        
        Args:
            evening_start_hour: Start hour for evening period.
            evening_end_hour: End hour for evening period.
        """
        self.evening_start_hour = evening_start_hour
        self.evening_end_hour = evening_end_hour
        self._baseline_pattern: Optional[np.ndarray] = None
    
    def set_baseline_pattern(self, hourly_pattern: np.ndarray) -> None:
        """Set baseline circadian light pattern.
        
        Args:
            hourly_pattern: 24-element array of typical hourly lux values.
        """
        self._baseline_pattern = hourly_pattern
    
    def extract(
        self,
        lux_values: np.ndarray,
        timestamps: np.ndarray,
        bin_start_hour: int = 0,
        bin_duration_minutes: float = 360.0,
    ) -> tuple[LightFeatures, LightQuality]:
        """Extract light features.
        
        Args:
            lux_values: Array of lux readings.
            timestamps: Array of timestamps (minutes from bin start).
            bin_start_hour: Starting hour of the bin (0-23).
            bin_duration_minutes: Total bin duration in minutes.
            
        Returns:
            Tuple of (LightFeatures, LightQuality).
        """
        if len(lux_values) == 0:
            return self._empty_features(), LightQuality(light_uptime_fraction=0.0)
        
        # Compute uptime
        expected_readings = bin_duration_minutes  # Assume ~1 reading per minute
        uptime = min(len(lux_values) / expected_readings, 1.0)
        
        # Basic statistics
        lux_mean = np.mean(lux_values)
        lux_std = np.std(lux_values)
        
        # Evening lux (18:00-24:00)
        lux_evening = self._compute_evening_lux(
            lux_values, timestamps, bin_start_hour
        )
        
        # Circadian regularity
        circadian_score = self._compute_circadian_regularity(
            lux_values, timestamps, bin_start_hour
        )
        
        features = LightFeatures(
            lux_mean=float(lux_mean),
            lux_std=float(lux_std),
            lux_evening_mean=float(lux_evening),
            circadian_regular_score=float(circadian_score),
        )
        
        quality = LightQuality(light_uptime_fraction=float(uptime))
        
        return features, quality
    
    def extract_from_summary(
        self,
        lux_mean: float,
        lux_std: float,
        lux_evening: float,
        circadian_score: float,
        uptime_fraction: float,
    ) -> tuple[LightFeatures, LightQuality]:
        """Create features from pre-computed summary statistics.
        
        Args:
            lux_mean: Mean lux value.
            lux_std: Lux standard deviation.
            lux_evening: Evening mean lux.
            circadian_score: Circadian regularity score.
            uptime_fraction: Sensor uptime fraction.
            
        Returns:
            Tuple of (LightFeatures, LightQuality).
        """
        features = LightFeatures(
            lux_mean=float(lux_mean),
            lux_std=float(lux_std),
            lux_evening_mean=float(lux_evening),
            circadian_regular_score=float(circadian_score),
        )
        
        quality = LightQuality(light_uptime_fraction=float(uptime_fraction))
        
        return features, quality
    
    def _compute_evening_lux(
        self,
        lux_values: np.ndarray,
        timestamps: np.ndarray,
        bin_start_hour: int,
    ) -> float:
        """Compute mean lux during evening hours."""
        evening_values = []
        
        for lux, ts in zip(lux_values, timestamps):
            hour = (bin_start_hour + ts / 60) % 24
            if self.evening_start_hour <= hour < self.evening_end_hour:
                evening_values.append(lux)
        
        if len(evening_values) == 0:
            return 0.0
        
        return float(np.mean(evening_values))
    
    def _compute_circadian_regularity(
        self,
        lux_values: np.ndarray,
        timestamps: np.ndarray,
        bin_start_hour: int,
    ) -> float:
        """Compute circadian regularity score.
        
        Measures how stable the daily light pattern is compared to baseline.
        """
        if self._baseline_pattern is None:
            return 1.0  # No baseline, assume regular
        
        # Build hourly means for this bin
        hourly_values: dict[int, list] = {i: [] for i in range(24)}
        
        for lux, ts in zip(lux_values, timestamps):
            hour = int((bin_start_hour + ts / 60) % 24)
            hourly_values[hour].append(lux)
        
        # Compute hourly means
        current_pattern = []
        for hour in range(24):
            if hourly_values[hour]:
                current_pattern.append(np.mean(hourly_values[hour]))
            else:
                current_pattern.append(np.nan)
        
        current_pattern = np.array(current_pattern)
        
        # Compute correlation with baseline (ignoring NaN)
        valid = ~np.isnan(current_pattern)
        if np.sum(valid) < 3:
            return 1.0
        
        current_valid = current_pattern[valid]
        baseline_valid = self._baseline_pattern[valid]
        
        # Normalize and compute correlation
        if np.std(current_valid) < 1e-6 or np.std(baseline_valid) < 1e-6:
            return 1.0
        
        correlation = np.corrcoef(current_valid, baseline_valid)[0, 1]
        
        # Convert correlation to score (0 to 1)
        score = (correlation + 1) / 2
        
        return float(np.clip(score, 0, 1))
    
    def _empty_features(self) -> LightFeatures:
        """Return empty features."""
        return LightFeatures(
            lux_mean=0.0,
            lux_std=0.0,
            lux_evening_mean=0.0,
            circadian_regular_score=1.0,
        )

