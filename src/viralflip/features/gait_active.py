"""Active gait test feature extraction from IMU data.

Features extracted from walking test:
- Cadence (steps per minute)
- Step interval statistics: mean, std, CV
- Jerk (smoothness): mean and std of acceleration derivative
- Sway proxy (lateral acceleration variance)
- Regularity score (autocorrelation peak strength)

Quality metrics:
- Steps detected threshold met
- Placement confidence
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import signal
from scipy.stats import linregress


@dataclass
class GaitActiveFeatures:
    """Container for active gait features."""
    
    cadence_spm: float
    step_interval_mean: float
    step_interval_std: float
    step_interval_cv: float
    jerk_mean: float
    jerk_std: float
    sway_proxy: float
    regularity_score: float
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([
            self.cadence_spm,
            self.step_interval_mean,
            self.step_interval_std,
            self.step_interval_cv,
            self.jerk_mean,
            self.jerk_std,
            self.sway_proxy,
            self.regularity_score,
        ], dtype=np.float32)


@dataclass
class GaitQuality:
    """Container for gait quality metrics."""
    
    steps_detected_ok: float  # 1.0 if enough steps, 0.0 otherwise
    placement_flag: float     # 1.0 if placement seems correct
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.steps_detected_ok, self.placement_flag], dtype=np.float32)


class GaitActiveFeatureExtractor:
    """Extract features from active walking tests."""
    
    def __init__(
        self,
        sample_rate: float = 100.0,
        min_steps: int = 10,
        cadence_bandpass: tuple[float, float] = (0.5, 3.0),
        expected_cadence_range: tuple[float, float] = (40, 140),
    ):
        """Initialize gait feature extractor.
        
        Args:
            sample_rate: IMU sample rate in Hz.
            min_steps: Minimum steps for valid gait analysis.
            cadence_bandpass: Bandpass filter range for cadence detection (Hz).
            expected_cadence_range: Expected cadence range in steps per minute.
        """
        self.sample_rate = sample_rate
        self.min_steps = min_steps
        self.cadence_bandpass = cadence_bandpass
        self.expected_cadence_range = expected_cadence_range
    
    def extract(
        self,
        accel_x: np.ndarray,
        accel_y: np.ndarray,
        accel_z: np.ndarray,
        gyro_x: Optional[np.ndarray] = None,
        gyro_y: Optional[np.ndarray] = None,
        gyro_z: Optional[np.ndarray] = None,
    ) -> tuple[GaitActiveFeatures, GaitQuality]:
        """Extract gait features from IMU data.
        
        Assumes phone is in pocket during walking.
        
        Args:
            accel_x: Accelerometer x-axis (m/s^2).
            accel_y: Accelerometer y-axis (m/s^2).
            accel_z: Accelerometer z-axis (m/s^2).
            gyro_x: Optional gyroscope x-axis (rad/s).
            gyro_y: Optional gyroscope y-axis (rad/s).
            gyro_z: Optional gyroscope z-axis (rad/s).
            
        Returns:
            Tuple of (GaitActiveFeatures, GaitQuality).
        """
        # Check minimum data length
        min_samples = int(2 * self.sample_rate)  # At least 2 seconds
        if len(accel_x) < min_samples:
            return self._empty_features(), GaitQuality(
                steps_detected_ok=0.0, placement_flag=0.0
            )
        
        # Compute acceleration magnitude
        accel_mag = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
        
        # Remove gravity (subtract mean)
        accel_mag_detrend = accel_mag - np.mean(accel_mag)
        
        # Bandpass filter for walking frequency
        accel_filtered = self._bandpass_filter(
            accel_mag_detrend,
            self.cadence_bandpass[0],
            self.cadence_bandpass[1],
        )
        
        # Detect steps
        step_indices = self._detect_steps(accel_filtered)
        n_steps = len(step_indices)
        
        if n_steps < self.min_steps:
            return self._empty_features(), GaitQuality(
                steps_detected_ok=0.0, placement_flag=0.5
            )
        
        # Compute step intervals
        step_intervals = np.diff(step_indices) / self.sample_rate
        
        # Cadence
        duration_sec = len(accel_x) / self.sample_rate
        cadence_spm = (n_steps / duration_sec) * 60
        
        # Step interval statistics
        step_interval_mean = np.mean(step_intervals)
        step_interval_std = np.std(step_intervals)
        step_interval_cv = step_interval_std / (step_interval_mean + 1e-10)
        
        # Jerk (derivative of acceleration)
        jerk = np.diff(accel_mag) * self.sample_rate
        jerk_mean = np.mean(np.abs(jerk))
        jerk_std = np.std(jerk)
        
        # Sway proxy: variance of lateral acceleration
        # Estimate vertical axis as the one with highest mean (gravity)
        means = [np.mean(accel_x), np.mean(accel_y), np.mean(accel_z)]
        vertical_idx = np.argmax(np.abs(means))
        
        if vertical_idx == 0:
            lateral = np.sqrt(accel_y**2 + accel_z**2)
        elif vertical_idx == 1:
            lateral = np.sqrt(accel_x**2 + accel_z**2)
        else:
            lateral = np.sqrt(accel_x**2 + accel_y**2)
        
        sway_proxy = np.var(lateral - np.mean(lateral))
        
        # Regularity score from autocorrelation
        regularity_score = self._compute_regularity(accel_filtered)
        
        # Quality assessment
        steps_ok = 1.0 if n_steps >= self.min_steps else 0.0
        placement_flag = self._assess_placement(accel_x, accel_y, accel_z)
        
        features = GaitActiveFeatures(
            cadence_spm=float(cadence_spm),
            step_interval_mean=float(step_interval_mean),
            step_interval_std=float(step_interval_std),
            step_interval_cv=float(step_interval_cv),
            jerk_mean=float(jerk_mean),
            jerk_std=float(jerk_std),
            sway_proxy=float(sway_proxy),
            regularity_score=float(regularity_score),
        )
        
        quality = GaitQuality(
            steps_detected_ok=steps_ok,
            placement_flag=placement_flag,
        )
        
        return features, quality
    
    def _bandpass_filter(
        self,
        data: np.ndarray,
        low_freq: float,
        high_freq: float,
        order: int = 4,
    ) -> np.ndarray:
        """Apply bandpass Butterworth filter."""
        nyquist = self.sample_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # Clamp to valid range
        low = max(0.001, min(low, 0.99))
        high = max(low + 0.001, min(high, 0.999))
        
        b, a = signal.butter(order, [low, high], btype='band')
        return signal.filtfilt(b, a, data)
    
    def _detect_steps(self, accel_filtered: np.ndarray) -> np.ndarray:
        """Detect step events from filtered acceleration."""
        # Find peaks (step impacts)
        min_step_samples = int(self.sample_rate * 60 / self.expected_cadence_range[1])
        
        # Adaptive threshold
        threshold = 0.3 * np.std(accel_filtered)
        
        peaks, properties = signal.find_peaks(
            accel_filtered,
            height=threshold,
            distance=min_step_samples,
        )
        
        return peaks
    
    def _compute_regularity(self, accel_filtered: np.ndarray) -> float:
        """Compute gait regularity from autocorrelation.
        
        Higher values indicate more regular/consistent gait.
        """
        # Autocorrelation
        acf = np.correlate(accel_filtered, accel_filtered, mode='full')
        acf = acf[len(acf) // 2:]
        acf = acf / (acf[0] + 1e-10)
        
        # Expected lag range for step period (based on cadence)
        min_lag = int(self.sample_rate * 60 / self.expected_cadence_range[1])
        max_lag = int(self.sample_rate * 60 / self.expected_cadence_range[0])
        max_lag = min(max_lag, len(acf) - 1)
        
        if max_lag <= min_lag:
            return 0.0
        
        # Find first major peak (step period)
        search_range = acf[min_lag:max_lag]
        if len(search_range) == 0:
            return 0.0
        
        # Peak strength is the regularity score
        peak_val = np.max(search_range)
        
        return float(np.clip(peak_val, 0, 1))
    
    def _assess_placement(
        self,
        accel_x: np.ndarray,
        accel_y: np.ndarray,
        accel_z: np.ndarray,
    ) -> float:
        """Assess if phone placement is appropriate for gait analysis.
        
        Returns confidence score between 0 and 1.
        """
        # Check if gravity is primarily on one axis (phone not tumbling)
        means = np.array([np.mean(accel_x), np.mean(accel_y), np.mean(accel_z)])
        gravity_mag = np.linalg.norm(means)
        
        # Expected gravity ~9.8 m/s^2
        if 8.0 < gravity_mag < 12.0:
            gravity_ok = 1.0
        else:
            gravity_ok = 0.5
        
        # Check stability (low variance in orientation)
        orientation_var = (
            np.var(accel_x / gravity_mag) +
            np.var(accel_y / gravity_mag) +
            np.var(accel_z / gravity_mag)
        )
        
        stability = np.clip(1.0 - orientation_var / 0.5, 0, 1)
        
        return float((gravity_ok + stability) / 2)
    
    def _empty_features(self) -> GaitActiveFeatures:
        """Return empty features."""
        return GaitActiveFeatures(
            cadence_spm=0.0,
            step_interval_mean=0.0,
            step_interval_std=0.0,
            step_interval_cv=0.0,
            jerk_mean=0.0,
            jerk_std=0.0,
            sway_proxy=0.0,
            regularity_score=0.0,
        )

