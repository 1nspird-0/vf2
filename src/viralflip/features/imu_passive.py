"""Passive IMU feature extraction from accelerometer/gyroscope data.

Features extracted:
- Activity minutes (low/medium/high intensity)
- Fragmentation index (number of activity bouts)
- Nighttime restlessness proxy
- Tremor band power (4-12 Hz during still periods)
- Sleep duration proxy

Quality metrics:
- IMU uptime fraction
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import signal


@dataclass
class IMUPassiveFeatures:
    """Container for passive IMU features."""
    
    activity_minutes_low: float
    activity_minutes_med: float
    activity_minutes_high: float
    fragmentation_index: float
    restlessness_night_proxy: float
    tremor_band_power: float
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([
            self.activity_minutes_low,
            self.activity_minutes_med,
            self.activity_minutes_high,
            self.fragmentation_index,
            self.restlessness_night_proxy,
            self.tremor_band_power,
        ], dtype=np.float32)


@dataclass
class IMUPassiveQuality:
    """Container for IMU quality metrics."""
    
    imu_uptime_fraction: float
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.imu_uptime_fraction], dtype=np.float32)


class IMUPassiveFeatureExtractor:
    """Extract features from passive IMU data."""
    
    def __init__(
        self,
        sample_rate: float = 50.0,
        activity_thresholds: tuple[float, float, float] = (0.1, 0.5, 1.5),
        window_sec: float = 10.0,
        tremor_band: tuple[float, float] = (4.0, 12.0),
        still_threshold: float = 0.05,
    ):
        """Initialize IMU feature extractor.
        
        Args:
            sample_rate: IMU sample rate in Hz.
            activity_thresholds: Thresholds for low/med/high activity (m/s^2).
            window_sec: Window size for feature computation in seconds.
            tremor_band: Frequency band for tremor detection (Hz).
            still_threshold: Threshold for detecting still periods (m/s^2).
        """
        self.sample_rate = sample_rate
        self.activity_thresholds = activity_thresholds
        self.window_size = int(window_sec * sample_rate)
        self.tremor_band = tremor_band
        self.still_threshold = still_threshold
    
    def extract(
        self,
        accel_x: np.ndarray,
        accel_y: np.ndarray,
        accel_z: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        is_night: bool = False,
        bin_duration_minutes: float = 360.0,
    ) -> tuple[IMUPassiveFeatures, IMUPassiveQuality]:
        """Extract passive IMU features.
        
        Args:
            accel_x: Accelerometer x-axis (m/s^2).
            accel_y: Accelerometer y-axis (m/s^2).
            accel_z: Accelerometer z-axis (m/s^2).
            timestamps: Optional timestamps (for uptime calculation).
            is_night: Whether this is a nighttime bin.
            bin_duration_minutes: Total bin duration in minutes.
            
        Returns:
            Tuple of (IMUPassiveFeatures, IMUPassiveQuality).
        """
        if len(accel_x) < self.window_size:
            uptime = len(accel_x) / (bin_duration_minutes * 60 * self.sample_rate)
            return self._empty_features(), IMUPassiveQuality(imu_uptime_fraction=uptime)
        
        # Compute acceleration magnitude minus gravity
        accel_mag = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
        accel_mag_detrend = np.abs(accel_mag - np.mean(accel_mag))
        
        # Window-based activity levels
        n_windows = len(accel_mag_detrend) // self.window_size
        window_activities = []
        
        for i in range(n_windows):
            start = i * self.window_size
            end = start + self.window_size
            window_std = np.std(accel_mag_detrend[start:end])
            window_activities.append(window_std)
        
        window_activities = np.array(window_activities)
        window_duration_min = (self.window_size / self.sample_rate) / 60
        
        # Activity minutes
        low_mask = (window_activities >= self.activity_thresholds[0]) & \
                   (window_activities < self.activity_thresholds[1])
        med_mask = (window_activities >= self.activity_thresholds[1]) & \
                   (window_activities < self.activity_thresholds[2])
        high_mask = window_activities >= self.activity_thresholds[2]
        
        activity_minutes_low = np.sum(low_mask) * window_duration_min
        activity_minutes_med = np.sum(med_mask) * window_duration_min
        activity_minutes_high = np.sum(high_mask) * window_duration_min
        
        # Fragmentation index (number of activity bouts)
        active_mask = window_activities >= self.activity_thresholds[0]
        fragmentation_index = self._count_bouts(active_mask)
        
        # Restlessness (for night bins)
        if is_night:
            restlessness = self._compute_restlessness(accel_mag_detrend)
        else:
            restlessness = 0.0
        
        # Tremor band power (during still periods)
        tremor_power = self._compute_tremor_power(accel_mag_detrend)
        
        # Compute uptime
        actual_samples = len(accel_x)
        expected_samples = bin_duration_minutes * 60 * self.sample_rate
        uptime = min(actual_samples / expected_samples, 1.0)
        
        features = IMUPassiveFeatures(
            activity_minutes_low=float(activity_minutes_low),
            activity_minutes_med=float(activity_minutes_med),
            activity_minutes_high=float(activity_minutes_high),
            fragmentation_index=float(fragmentation_index),
            restlessness_night_proxy=float(restlessness),
            tremor_band_power=float(tremor_power),
        )
        
        quality = IMUPassiveQuality(imu_uptime_fraction=float(uptime))
        
        return features, quality
    
    def extract_from_summary(
        self,
        activity_minutes: dict[str, float],
        fragmentation: float,
        restlessness: float,
        tremor_power: float,
        uptime_fraction: float,
    ) -> tuple[IMUPassiveFeatures, IMUPassiveQuality]:
        """Extract features from pre-computed summary statistics.
        
        Args:
            activity_minutes: Dict with 'low', 'med', 'high' keys.
            fragmentation: Number of activity bouts.
            restlessness: Nighttime restlessness score.
            tremor_power: Tremor band power.
            uptime_fraction: IMU uptime fraction.
            
        Returns:
            Tuple of (IMUPassiveFeatures, IMUPassiveQuality).
        """
        features = IMUPassiveFeatures(
            activity_minutes_low=float(activity_minutes.get('low', 0.0)),
            activity_minutes_med=float(activity_minutes.get('med', 0.0)),
            activity_minutes_high=float(activity_minutes.get('high', 0.0)),
            fragmentation_index=float(fragmentation),
            restlessness_night_proxy=float(restlessness),
            tremor_band_power=float(tremor_power),
        )
        
        quality = IMUPassiveQuality(imu_uptime_fraction=float(uptime_fraction))
        
        return features, quality
    
    def _count_bouts(self, active_mask: np.ndarray) -> int:
        """Count number of activity bouts (consecutive active windows)."""
        if len(active_mask) == 0:
            return 0
        
        # Count transitions from inactive to active
        transitions = np.diff(active_mask.astype(int))
        bout_starts = np.sum(transitions == 1)
        
        # Add 1 if starts with active
        if active_mask[0]:
            bout_starts += 1
        
        return int(bout_starts)
    
    def _compute_restlessness(self, accel_detrend: np.ndarray) -> float:
        """Compute restlessness proxy for nighttime data.
        
        Based on variance of movement during supposedly still periods.
        """
        # Find still periods
        window_vars = []
        for i in range(0, len(accel_detrend) - self.window_size, self.window_size):
            window = accel_detrend[i:i + self.window_size]
            window_vars.append(np.var(window))
        
        if len(window_vars) == 0:
            return 0.0
        
        window_vars = np.array(window_vars)
        
        # Still periods are those below threshold
        still_mask = window_vars < self.still_threshold ** 2
        
        if np.sum(still_mask) == 0:
            return float(np.mean(window_vars))
        
        # Restlessness = variance during "still" periods
        still_vars = window_vars[still_mask]
        restlessness = np.mean(still_vars) * 1000  # Scale for visibility
        
        return float(restlessness)
    
    def _compute_tremor_power(self, accel_detrend: np.ndarray) -> float:
        """Compute power in tremor frequency band during still periods."""
        # Find still windows
        still_segments = []
        
        for i in range(0, len(accel_detrend) - self.window_size, self.window_size):
            window = accel_detrend[i:i + self.window_size]
            if np.std(window) < self.still_threshold:
                still_segments.append(window)
        
        if len(still_segments) == 0:
            return 0.0
        
        # Compute power spectral density for still segments
        tremor_powers = []
        
        for segment in still_segments:
            freqs, psd = signal.welch(
                segment,
                fs=self.sample_rate,
                nperseg=min(len(segment), 256),
            )
            
            # Extract tremor band
            tremor_mask = (freqs >= self.tremor_band[0]) & (freqs <= self.tremor_band[1])
            if np.any(tremor_mask):
                tremor_power = np.mean(psd[tremor_mask])
                tremor_powers.append(tremor_power)
        
        if len(tremor_powers) == 0:
            return 0.0
        
        return float(np.mean(tremor_powers))
    
    def _empty_features(self) -> IMUPassiveFeatures:
        """Return empty features."""
        return IMUPassiveFeatures(
            activity_minutes_low=0.0,
            activity_minutes_med=0.0,
            activity_minutes_high=0.0,
            fragmentation_index=0.0,
            restlessness_night_proxy=0.0,
            tremor_band_power=0.0,
        )

