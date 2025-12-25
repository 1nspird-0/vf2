"""Cough event feature extraction.

Features extracted from cough event timestamps/counts:
- Total cough count
- Night/day cough counts
- Burstiness (variance/mean of hourly counts)
- Max hourly count
- Mean confidence of detections

Quality metrics:
- Microphone uptime fraction
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CoughFeatures:
    """Container for cough features."""
    
    cough_count_total: float
    cough_count_night: float
    cough_count_day: float
    cough_burstiness: float
    cough_max_hourly: float
    cough_conf_mean: float
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([
            self.cough_count_total,
            self.cough_count_night,
            self.cough_count_day,
            self.cough_burstiness,
            self.cough_max_hourly,
            self.cough_conf_mean,
        ], dtype=np.float32)


@dataclass
class CoughQuality:
    """Container for cough quality metrics."""
    
    mic_uptime_fraction: float
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.mic_uptime_fraction], dtype=np.float32)


class CoughFeatureExtractor:
    """Extract features from cough event detections.
    
    This extractor operates on pre-detected cough events (timestamps + confidences),
    not raw audio. The cough detector itself should run on-device and only output
    event metadata to preserve privacy.
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.5,
        night_start_hour: int = 0,
        night_end_hour: int = 6,
        bin_hours: int = 6,
    ):
        """Initialize cough feature extractor.
        
        Args:
            confidence_threshold: Minimum confidence to count a cough event.
            night_start_hour: Start hour for night period (0-23).
            night_end_hour: End hour for night period (0-23).
            bin_hours: Duration of time bin in hours.
        """
        self.confidence_threshold = confidence_threshold
        self.night_start_hour = night_start_hour
        self.night_end_hour = night_end_hour
        self.bin_hours = bin_hours
    
    def extract(
        self,
        event_timestamps: np.ndarray,
        event_confidences: np.ndarray,
        bin_start_hour: int,
        mic_uptime_minutes: float,
        bin_duration_minutes: float = 360.0,  # 6 hours
    ) -> tuple[CoughFeatures, CoughQuality]:
        """Extract cough features from event data.
        
        Args:
            event_timestamps: Array of cough event timestamps (minutes since bin start).
            event_confidences: Array of confidence scores for each event.
            bin_start_hour: Starting hour of the time bin (0-23).
            mic_uptime_minutes: Total minutes microphone was active.
            bin_duration_minutes: Total duration of bin in minutes.
            
        Returns:
            Tuple of (CoughFeatures, CoughQuality).
        """
        if len(event_timestamps) == 0 or len(event_confidences) == 0:
            return self._empty_features(), CoughQuality(
                mic_uptime_fraction=mic_uptime_minutes / bin_duration_minutes
            )
        
        # Filter by confidence
        mask = event_confidences >= self.confidence_threshold
        timestamps = event_timestamps[mask]
        confidences = event_confidences[mask]
        
        if len(timestamps) == 0:
            return self._empty_features(), CoughQuality(
                mic_uptime_fraction=mic_uptime_minutes / bin_duration_minutes
            )
        
        # Total count
        cough_count_total = len(timestamps)
        
        # Night/day split
        night_count, day_count = self._split_night_day(
            timestamps, bin_start_hour, bin_duration_minutes
        )
        
        # Hourly counts for burstiness
        hourly_counts = self._compute_hourly_counts(timestamps, bin_duration_minutes)
        
        # Burstiness (index of dispersion)
        mean_hourly = np.mean(hourly_counts) if len(hourly_counts) > 0 else 0
        var_hourly = np.var(hourly_counts) if len(hourly_counts) > 0 else 0
        burstiness = var_hourly / (mean_hourly + 1e-6)
        
        # Max hourly
        max_hourly = np.max(hourly_counts) if len(hourly_counts) > 0 else 0
        
        # Mean confidence
        conf_mean = np.mean(confidences)
        
        features = CoughFeatures(
            cough_count_total=float(cough_count_total),
            cough_count_night=float(night_count),
            cough_count_day=float(day_count),
            cough_burstiness=float(burstiness),
            cough_max_hourly=float(max_hourly),
            cough_conf_mean=float(conf_mean),
        )
        
        quality = CoughQuality(
            mic_uptime_fraction=mic_uptime_minutes / bin_duration_minutes
        )
        
        return features, quality
    
    def extract_from_hourly_counts(
        self,
        hourly_counts: np.ndarray,
        hourly_confidences: Optional[np.ndarray] = None,
        bin_start_hour: int = 0,
        mic_uptime_fraction: float = 1.0,
    ) -> tuple[CoughFeatures, CoughQuality]:
        """Extract features from pre-aggregated hourly counts.
        
        Args:
            hourly_counts: Array of cough counts per hour within the bin.
            hourly_confidences: Optional mean confidence per hour.
            bin_start_hour: Starting hour of the time bin (0-23).
            mic_uptime_fraction: Fraction of time microphone was active.
            
        Returns:
            Tuple of (CoughFeatures, CoughQuality).
        """
        if len(hourly_counts) == 0 or np.sum(hourly_counts) == 0:
            return self._empty_features(), CoughQuality(mic_uptime_fraction=mic_uptime_fraction)
        
        # Total count
        cough_count_total = np.sum(hourly_counts)
        
        # Night/day split based on hours
        night_count = 0
        day_count = 0
        for i, count in enumerate(hourly_counts):
            hour = (bin_start_hour + i) % 24
            if self.night_start_hour <= hour < self.night_end_hour:
                night_count += count
            else:
                day_count += count
        
        # Burstiness
        mean_hourly = np.mean(hourly_counts)
        var_hourly = np.var(hourly_counts)
        burstiness = var_hourly / (mean_hourly + 1e-6)
        
        # Max hourly
        max_hourly = np.max(hourly_counts)
        
        # Mean confidence
        if hourly_confidences is not None and len(hourly_confidences) > 0:
            # Weight by counts
            total = np.sum(hourly_counts)
            if total > 0:
                conf_mean = np.sum(hourly_counts * hourly_confidences) / total
            else:
                conf_mean = 0.0
        else:
            conf_mean = 1.0  # Assume high confidence if not provided
        
        features = CoughFeatures(
            cough_count_total=float(cough_count_total),
            cough_count_night=float(night_count),
            cough_count_day=float(day_count),
            cough_burstiness=float(burstiness),
            cough_max_hourly=float(max_hourly),
            cough_conf_mean=float(conf_mean),
        )
        
        quality = CoughQuality(mic_uptime_fraction=mic_uptime_fraction)
        
        return features, quality
    
    def _split_night_day(
        self,
        timestamps: np.ndarray,
        bin_start_hour: int,
        bin_duration_minutes: float,
    ) -> tuple[int, int]:
        """Split cough counts into night and day periods."""
        night_count = 0
        day_count = 0
        
        for ts in timestamps:
            # Convert timestamp to hour of day
            hour_offset = ts / 60.0  # minutes to hours
            hour = (bin_start_hour + hour_offset) % 24
            
            if self.night_start_hour <= hour < self.night_end_hour:
                night_count += 1
            else:
                day_count += 1
        
        return night_count, day_count
    
    def _compute_hourly_counts(
        self,
        timestamps: np.ndarray,
        bin_duration_minutes: float,
    ) -> np.ndarray:
        """Compute hourly cough counts within the bin."""
        n_hours = int(np.ceil(bin_duration_minutes / 60))
        hourly_counts = np.zeros(n_hours)
        
        for ts in timestamps:
            hour_idx = int(ts / 60)
            if 0 <= hour_idx < n_hours:
                hourly_counts[hour_idx] += 1
        
        return hourly_counts
    
    def _empty_features(self) -> CoughFeatures:
        """Return empty features."""
        return CoughFeatures(
            cough_count_total=0.0,
            cough_count_night=0.0,
            cough_count_day=0.0,
            cough_burstiness=0.0,
            cough_max_hourly=0.0,
            cough_conf_mean=0.0,
        )

