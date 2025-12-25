"""Barometer feature extraction.

Features extracted:
- Mean pressure
- Pressure standard deviation
- Pressure slope (linear fit)
- Pressure jump count (sudden changes)

Quality metrics:
- Barometer uptime fraction
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import linregress


@dataclass
class BaroFeatures:
    """Container for barometer features."""
    
    pressure_mean: float
    pressure_std: float
    pressure_slope: float
    pressure_jump_count: float
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([
            self.pressure_mean,
            self.pressure_std,
            self.pressure_slope,
            self.pressure_jump_count,
        ], dtype=np.float32)


@dataclass
class BaroQuality:
    """Container for barometer quality metrics."""
    
    baro_uptime_fraction: float
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.baro_uptime_fraction], dtype=np.float32)


class BaroFeatureExtractor:
    """Extract features from barometer data."""
    
    def __init__(
        self,
        jump_threshold_hpa: float = 2.0,
        min_readings: int = 10,
    ):
        """Initialize barometer feature extractor.
        
        Args:
            jump_threshold_hpa: Threshold for pressure jump detection (hPa).
            min_readings: Minimum readings for valid extraction.
        """
        self.jump_threshold = jump_threshold_hpa
        self.min_readings = min_readings
    
    def extract(
        self,
        pressure_values: np.ndarray,
        timestamps: np.ndarray,
        bin_duration_minutes: float = 360.0,
    ) -> tuple[BaroFeatures, BaroQuality]:
        """Extract barometer features.
        
        Args:
            pressure_values: Array of pressure readings (hPa).
            timestamps: Array of timestamps (minutes from bin start).
            bin_duration_minutes: Total bin duration in minutes.
            
        Returns:
            Tuple of (BaroFeatures, BaroQuality).
        """
        if len(pressure_values) < self.min_readings:
            uptime = len(pressure_values) / (bin_duration_minutes / 5)  # Assume 5-min interval
            return self._empty_features(), BaroQuality(baro_uptime_fraction=uptime)
        
        # Compute uptime
        expected_readings = bin_duration_minutes / 5  # Assume ~5 min reading interval
        uptime = min(len(pressure_values) / expected_readings, 1.0)
        
        # Basic statistics
        pressure_mean = np.mean(pressure_values)
        pressure_std = np.std(pressure_values)
        
        # Pressure slope (linear regression)
        slope, _, _, _, _ = linregress(timestamps, pressure_values)
        pressure_slope = slope * 60  # Convert to hPa per hour
        
        # Pressure jump count
        pressure_diffs = np.abs(np.diff(pressure_values))
        jump_count = np.sum(pressure_diffs > self.jump_threshold)
        
        features = BaroFeatures(
            pressure_mean=float(pressure_mean),
            pressure_std=float(pressure_std),
            pressure_slope=float(pressure_slope),
            pressure_jump_count=float(jump_count),
        )
        
        quality = BaroQuality(baro_uptime_fraction=float(uptime))
        
        return features, quality
    
    def extract_from_summary(
        self,
        pressure_mean: float,
        pressure_std: float,
        pressure_slope: float,
        jump_count: int,
        uptime_fraction: float,
    ) -> tuple[BaroFeatures, BaroQuality]:
        """Create features from pre-computed summary statistics.
        
        Args:
            pressure_mean: Mean pressure (hPa).
            pressure_std: Pressure standard deviation.
            pressure_slope: Pressure slope (hPa/hour).
            jump_count: Number of pressure jumps.
            uptime_fraction: Sensor uptime fraction.
            
        Returns:
            Tuple of (BaroFeatures, BaroQuality).
        """
        features = BaroFeatures(
            pressure_mean=float(pressure_mean),
            pressure_std=float(pressure_std),
            pressure_slope=float(pressure_slope),
            pressure_jump_count=float(jump_count),
        )
        
        quality = BaroQuality(baro_uptime_fraction=float(uptime_fraction))
        
        return features, quality
    
    def _empty_features(self) -> BaroFeatures:
        """Return empty features."""
        return BaroFeatures(
            pressure_mean=0.0,
            pressure_std=0.0,
            pressure_slope=0.0,
            pressure_jump_count=0.0,
        )

