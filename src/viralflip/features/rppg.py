"""Remote photoplethysmography (rPPG) feature extraction.

Features extracted:
- Heart rate mean
- Inter-beat interval (IBI) standard deviation
- HRV RMSSD proxy
- Pulse amplitude mean and CV

Quality metrics:
- Signal quality index (SQI)
- Motion artifact score
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import signal
from scipy.stats import linregress


@dataclass
class RPPGFeatures:
    """Container for rPPG features."""
    
    hr_mean: float
    ibi_std: float
    hrv_rmssd_proxy: float
    pulse_amp_mean: float
    pulse_amp_cv: float
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([
            self.hr_mean,
            self.ibi_std,
            self.hrv_rmssd_proxy,
            self.pulse_amp_mean,
            self.pulse_amp_cv,
        ], dtype=np.float32)


@dataclass
class RPPGQuality:
    """Container for rPPG quality metrics."""
    
    sqi: float  # Signal quality index
    motion_artifact_score: float
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.sqi, self.motion_artifact_score], dtype=np.float32)


class RPPGFeatureExtractor:
    """Extract rPPG features from camera signal.
    
    Supports two modes:
    1. Finger over rear camera (with flash) - more robust
    2. Face video from front camera - requires face detection
    
    This implementation focuses on the finger-over-camera method for robustness.
    """
    
    def __init__(
        self,
        sample_rate: float = 30.0,
        hr_bandpass: tuple[float, float] = (0.7, 3.0),  # 42-180 bpm
        min_quality: float = 0.3,
        method: str = "pos",  # "pos", "chrom", or "green"
    ):
        """Initialize rPPG feature extractor.
        
        Args:
            sample_rate: Video frame rate in Hz.
            hr_bandpass: Bandpass filter for heart rate (Hz).
            min_quality: Minimum SQI for valid extraction.
            method: rPPG extraction method ("pos", "chrom", "green").
        """
        self.sample_rate = sample_rate
        self.hr_bandpass = hr_bandpass
        self.min_quality = min_quality
        self.method = method
    
    def extract(
        self,
        rgb_signal: np.ndarray,
        motion_signal: Optional[np.ndarray] = None,
    ) -> tuple[RPPGFeatures, RPPGQuality]:
        """Extract rPPG features from RGB signal.
        
        Args:
            rgb_signal: RGB values over time, shape (n_frames, 3).
            motion_signal: Optional motion magnitude signal for quality.
            
        Returns:
            Tuple of (RPPGFeatures, RPPGQuality).
        """
        if len(rgb_signal) < int(5 * self.sample_rate):  # Need at least 5 seconds
            return self._empty_features(), RPPGQuality(sqi=0.0, motion_artifact_score=1.0)
        
        # Extract pulse signal using specified method
        if self.method == "green":
            pulse = self._extract_green(rgb_signal)
        elif self.method == "chrom":
            pulse = self._extract_chrom(rgb_signal)
        else:  # pos
            pulse = self._extract_pos(rgb_signal)
        
        # Bandpass filter
        pulse_filtered = self._bandpass_filter(pulse)
        
        # Compute quality metrics
        sqi = self._compute_sqi(pulse_filtered)
        motion_score = self._compute_motion_score(motion_signal) if motion_signal is not None else 0.5
        
        if sqi < self.min_quality:
            return self._empty_features(), RPPGQuality(sqi=sqi, motion_artifact_score=motion_score)
        
        # Detect peaks (heartbeats)
        peaks = self._detect_peaks(pulse_filtered)
        
        if len(peaks) < 3:
            return self._empty_features(), RPPGQuality(sqi=sqi, motion_artifact_score=motion_score)
        
        # Compute IBIs
        ibis = np.diff(peaks) / self.sample_rate  # in seconds
        
        # Filter physiologically implausible IBIs
        valid_mask = (ibis > 0.33) & (ibis < 1.5)  # 40-180 bpm
        ibis = ibis[valid_mask]
        
        if len(ibis) < 2:
            return self._empty_features(), RPPGQuality(sqi=sqi, motion_artifact_score=motion_score)
        
        # Heart rate
        hr_mean = 60.0 / np.mean(ibis)
        
        # IBI variability
        ibi_std = np.std(ibis) * 1000  # Convert to ms
        
        # RMSSD (root mean square of successive differences)
        ibi_diffs = np.diff(ibis) * 1000  # ms
        hrv_rmssd = np.sqrt(np.mean(ibi_diffs ** 2))
        
        # Pulse amplitude
        pulse_amps = []
        for i in range(len(peaks) - 1):
            segment = pulse_filtered[peaks[i]:peaks[i + 1]]
            if len(segment) > 0:
                amp = np.max(segment) - np.min(segment)
                pulse_amps.append(amp)
        
        if len(pulse_amps) > 0:
            pulse_amp_mean = np.mean(pulse_amps)
            pulse_amp_cv = np.std(pulse_amps) / (pulse_amp_mean + 1e-10)
        else:
            pulse_amp_mean = 0.0
            pulse_amp_cv = 0.0
        
        features = RPPGFeatures(
            hr_mean=float(hr_mean),
            ibi_std=float(ibi_std),
            hrv_rmssd_proxy=float(hrv_rmssd),
            pulse_amp_mean=float(pulse_amp_mean),
            pulse_amp_cv=float(pulse_amp_cv),
        )
        
        quality = RPPGQuality(sqi=float(sqi), motion_artifact_score=float(motion_score))
        
        return features, quality
    
    def extract_from_finger(
        self,
        intensity_signal: np.ndarray,
        motion_signal: Optional[np.ndarray] = None,
    ) -> tuple[RPPGFeatures, RPPGQuality]:
        """Extract features from finger-over-camera intensity signal.
        
        This is a simpler interface when only intensity (not full RGB) is available.
        
        Args:
            intensity_signal: Intensity values over time.
            motion_signal: Optional motion magnitude for quality.
            
        Returns:
            Tuple of (RPPGFeatures, RPPGQuality).
        """
        if len(intensity_signal) < int(5 * self.sample_rate):
            return self._empty_features(), RPPGQuality(sqi=0.0, motion_artifact_score=1.0)
        
        # Detrend
        pulse = self._detrend(intensity_signal)
        
        # Bandpass filter
        pulse_filtered = self._bandpass_filter(pulse)
        
        # Compute quality
        sqi = self._compute_sqi(pulse_filtered)
        motion_score = self._compute_motion_score(motion_signal) if motion_signal is not None else 0.5
        
        if sqi < self.min_quality:
            return self._empty_features(), RPPGQuality(sqi=sqi, motion_artifact_score=motion_score)
        
        # Detect peaks
        peaks = self._detect_peaks(pulse_filtered)
        
        if len(peaks) < 3:
            return self._empty_features(), RPPGQuality(sqi=sqi, motion_artifact_score=motion_score)
        
        # Compute IBIs
        ibis = np.diff(peaks) / self.sample_rate
        valid_mask = (ibis > 0.33) & (ibis < 1.5)
        ibis = ibis[valid_mask]
        
        if len(ibis) < 2:
            return self._empty_features(), RPPGQuality(sqi=sqi, motion_artifact_score=motion_score)
        
        hr_mean = 60.0 / np.mean(ibis)
        ibi_std = np.std(ibis) * 1000
        
        ibi_diffs = np.diff(ibis) * 1000
        hrv_rmssd = np.sqrt(np.mean(ibi_diffs ** 2)) if len(ibi_diffs) > 0 else 0.0
        
        pulse_amps = []
        for i in range(len(peaks) - 1):
            segment = pulse_filtered[peaks[i]:peaks[i + 1]]
            if len(segment) > 0:
                pulse_amps.append(np.max(segment) - np.min(segment))
        
        pulse_amp_mean = np.mean(pulse_amps) if pulse_amps else 0.0
        pulse_amp_cv = np.std(pulse_amps) / (pulse_amp_mean + 1e-10) if pulse_amps else 0.0
        
        features = RPPGFeatures(
            hr_mean=float(hr_mean),
            ibi_std=float(ibi_std),
            hrv_rmssd_proxy=float(hrv_rmssd),
            pulse_amp_mean=float(pulse_amp_mean),
            pulse_amp_cv=float(pulse_amp_cv),
        )
        
        quality = RPPGQuality(sqi=float(sqi), motion_artifact_score=float(motion_score))
        
        return features, quality
    
    def _extract_green(self, rgb: np.ndarray) -> np.ndarray:
        """Extract pulse using green channel."""
        green = rgb[:, 1]
        return self._detrend(green)
    
    def _extract_chrom(self, rgb: np.ndarray) -> np.ndarray:
        """Extract pulse using CHROM method.
        
        De Haan, G., & Jeanne, V. (2013). Robust pulse rate from 
        chrominance-based rPPG.
        """
        # Normalize each channel
        rgb_norm = rgb / (np.mean(rgb, axis=0, keepdims=True) + 1e-10)
        
        # CHROM signal
        xs = 3 * rgb_norm[:, 0] - 2 * rgb_norm[:, 1]
        ys = 1.5 * rgb_norm[:, 0] + rgb_norm[:, 1] - 1.5 * rgb_norm[:, 2]
        
        # Combine
        alpha = np.std(xs) / (np.std(ys) + 1e-10)
        pulse = xs - alpha * ys
        
        return self._detrend(pulse)
    
    def _extract_pos(self, rgb: np.ndarray) -> np.ndarray:
        """Extract pulse using POS method.
        
        Wang, W., et al. (2017). Algorithmic principles of remote PPG.
        """
        # Normalize
        rgb_norm = rgb / (np.mean(rgb, axis=0, keepdims=True) + 1e-10)
        
        # Project onto plane orthogonal to skin tone
        # POS: P = [0, 1, -1] and Q = [-2, 1, 1]
        s1 = rgb_norm[:, 1] - rgb_norm[:, 2]  # G - B
        s2 = -2 * rgb_norm[:, 0] + rgb_norm[:, 1] + rgb_norm[:, 2]  # -2R + G + B
        
        # Combine with optimal weighting
        alpha = np.std(s1) / (np.std(s2) + 1e-10)
        pulse = s1 + alpha * s2
        
        return self._detrend(pulse)
    
    def _detrend(self, signal_data: np.ndarray) -> np.ndarray:
        """Remove linear trend from signal."""
        x = np.arange(len(signal_data))
        slope, intercept, _, _, _ = linregress(x, signal_data)
        trend = slope * x + intercept
        return signal_data - trend
    
    def _bandpass_filter(self, data: np.ndarray, order: int = 4) -> np.ndarray:
        """Apply bandpass Butterworth filter."""
        nyquist = self.sample_rate / 2
        low = self.hr_bandpass[0] / nyquist
        high = self.hr_bandpass[1] / nyquist
        
        low = max(0.001, min(low, 0.99))
        high = max(low + 0.001, min(high, 0.999))
        
        b, a = signal.butter(order, [low, high], btype='band')
        return signal.filtfilt(b, a, data)
    
    def _compute_sqi(self, pulse: np.ndarray) -> float:
        """Compute signal quality index.
        
        Based on spectral peak strength in expected HR range.
        """
        # Power spectral density
        freqs, psd = signal.welch(pulse, fs=self.sample_rate, nperseg=min(len(pulse), 256))
        
        # Find peak in HR range
        hr_mask = (freqs >= self.hr_bandpass[0]) & (freqs <= self.hr_bandpass[1])
        
        if not np.any(hr_mask):
            return 0.0
        
        hr_psd = psd[hr_mask]
        peak_power = np.max(hr_psd)
        total_power = np.sum(psd) + 1e-10
        
        # SQI is ratio of peak to total power
        sqi = peak_power / total_power
        
        # Also check spectral entropy (lower is better)
        psd_norm = psd / total_power
        entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10))
        max_entropy = np.log(len(psd))
        entropy_score = 1 - (entropy / max_entropy)
        
        return float(np.clip((sqi + entropy_score) / 2, 0, 1))
    
    def _compute_motion_score(self, motion: np.ndarray) -> float:
        """Compute motion artifact score (higher = worse)."""
        if motion is None:
            return 0.5
        
        # Normalize motion
        motion_norm = (motion - np.min(motion)) / (np.max(motion) - np.min(motion) + 1e-10)
        
        # Score based on mean and variance
        motion_score = np.mean(motion_norm) + np.std(motion_norm)
        
        return float(np.clip(motion_score, 0, 1))
    
    def _detect_peaks(self, pulse: np.ndarray) -> np.ndarray:
        """Detect pulse peaks."""
        # Expected distance between peaks (based on HR range)
        min_distance = int(self.sample_rate * 60 / 180)  # 180 bpm max
        
        # Adaptive threshold
        threshold = 0.3 * np.std(pulse)
        
        peaks, _ = signal.find_peaks(
            pulse,
            height=threshold,
            distance=min_distance,
        )
        
        return peaks
    
    def _empty_features(self) -> RPPGFeatures:
        """Return empty features."""
        return RPPGFeatures(
            hr_mean=0.0,
            ibi_std=0.0,
            hrv_rmssd_proxy=0.0,
            pulse_amp_mean=0.0,
            pulse_amp_cv=0.0,
        )

