"""Screen event feature extraction.

Features extracted:
- Screen on count
- Screen on minutes
- First unlock time
- Last lock time  
- Longest screen off interval

Quality metrics:
- Screen event uptime fraction

These features serve as sleep proxies and behavioral confounds.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ScreenFeatures:
    """Container for screen event features."""
    
    screen_on_count: float
    screen_on_minutes: float
    first_unlock_time: float  # Minutes from bin start
    last_lock_time: float     # Minutes from bin start
    longest_screen_off_interval: float  # Minutes
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([
            self.screen_on_count,
            self.screen_on_minutes,
            self.first_unlock_time,
            self.last_lock_time,
            self.longest_screen_off_interval,
        ], dtype=np.float32)


@dataclass
class ScreenQuality:
    """Container for screen quality metrics."""
    
    screen_event_uptime_fraction: float
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.screen_event_uptime_fraction], dtype=np.float32)


class ScreenFeatureExtractor:
    """Extract features from screen on/off events."""
    
    def __init__(
        self,
        min_events: int = 2,
    ):
        """Initialize screen feature extractor.
        
        Args:
            min_events: Minimum events for valid extraction.
        """
        self.min_events = min_events
    
    def extract(
        self,
        event_times: np.ndarray,
        event_types: np.ndarray,
        bin_duration_minutes: float = 360.0,
    ) -> tuple[ScreenFeatures, ScreenQuality]:
        """Extract screen features from events.
        
        Args:
            event_times: Array of event timestamps (minutes from bin start).
            event_types: Array of event types (1 = screen_on/unlock, 0 = screen_off/lock).
            bin_duration_minutes: Total bin duration in minutes.
            
        Returns:
            Tuple of (ScreenFeatures, ScreenQuality).
        """
        if len(event_times) < self.min_events:
            return self._empty_features(bin_duration_minutes), ScreenQuality(
                screen_event_uptime_fraction=0.5
            )
        
        # Sort events by time
        sort_idx = np.argsort(event_times)
        event_times = event_times[sort_idx]
        event_types = event_types[sort_idx]
        
        # Count screen on events
        screen_on_count = np.sum(event_types == 1)
        
        # Compute screen on duration
        screen_on_minutes = self._compute_on_duration(
            event_times, event_types, bin_duration_minutes
        )
        
        # First unlock time
        on_events = event_times[event_types == 1]
        first_unlock = on_events[0] if len(on_events) > 0 else bin_duration_minutes
        
        # Last lock time
        off_events = event_times[event_types == 0]
        last_lock = off_events[-1] if len(off_events) > 0 else 0.0
        
        # Longest screen off interval
        longest_off = self._compute_longest_off(
            event_times, event_types, bin_duration_minutes
        )
        
        # Quality: assume full coverage if we have events
        uptime = 1.0 if len(event_times) >= self.min_events else 0.5
        
        features = ScreenFeatures(
            screen_on_count=float(screen_on_count),
            screen_on_minutes=float(screen_on_minutes),
            first_unlock_time=float(first_unlock),
            last_lock_time=float(last_lock),
            longest_screen_off_interval=float(longest_off),
        )
        
        quality = ScreenQuality(screen_event_uptime_fraction=float(uptime))
        
        return features, quality
    
    def extract_from_summary(
        self,
        screen_on_count: int,
        screen_on_minutes: float,
        first_unlock: float,
        last_lock: float,
        longest_off: float,
        uptime_fraction: float = 1.0,
    ) -> tuple[ScreenFeatures, ScreenQuality]:
        """Create features from pre-computed summary statistics.
        
        Args:
            screen_on_count: Number of screen on events.
            screen_on_minutes: Total screen on time.
            first_unlock: First unlock time (minutes from bin start).
            last_lock: Last lock time (minutes from bin start).
            longest_off: Longest screen off interval (minutes).
            uptime_fraction: Event tracking uptime.
            
        Returns:
            Tuple of (ScreenFeatures, ScreenQuality).
        """
        features = ScreenFeatures(
            screen_on_count=float(screen_on_count),
            screen_on_minutes=float(screen_on_minutes),
            first_unlock_time=float(first_unlock),
            last_lock_time=float(last_lock),
            longest_screen_off_interval=float(longest_off),
        )
        
        quality = ScreenQuality(screen_event_uptime_fraction=float(uptime_fraction))
        
        return features, quality
    
    def _compute_on_duration(
        self,
        event_times: np.ndarray,
        event_types: np.ndarray,
        bin_duration: float,
    ) -> float:
        """Compute total screen on duration."""
        on_duration = 0.0
        screen_on = False
        last_on_time = 0.0
        
        for time, event_type in zip(event_times, event_types):
            if event_type == 1 and not screen_on:  # Screen on
                screen_on = True
                last_on_time = time
            elif event_type == 0 and screen_on:  # Screen off
                on_duration += time - last_on_time
                screen_on = False
        
        # If still on at end of bin
        if screen_on:
            on_duration += bin_duration - last_on_time
        
        return on_duration
    
    def _compute_longest_off(
        self,
        event_times: np.ndarray,
        event_types: np.ndarray,
        bin_duration: float,
    ) -> float:
        """Compute longest screen off interval."""
        off_intervals = []
        
        # Time from start to first on
        on_events = event_times[event_types == 1]
        if len(on_events) > 0:
            off_intervals.append(on_events[0])
        
        # Intervals between consecutive off and on
        screen_on = False
        last_off_time = 0.0
        
        for time, event_type in zip(event_times, event_types):
            if event_type == 0:  # Screen off
                last_off_time = time
                screen_on = False
            elif event_type == 1 and not screen_on:  # Screen on after off
                if last_off_time > 0:
                    off_intervals.append(time - last_off_time)
                screen_on = True
        
        # Time from last off to end
        off_events = event_times[event_types == 0]
        if len(off_events) > 0:
            off_intervals.append(bin_duration - off_events[-1])
        
        if len(off_intervals) == 0:
            return bin_duration
        
        return max(off_intervals)
    
    def _empty_features(self, bin_duration: float) -> ScreenFeatures:
        """Return empty features."""
        return ScreenFeatures(
            screen_on_count=0.0,
            screen_on_minutes=0.0,
            first_unlock_time=bin_duration,
            last_lock_time=0.0,
            longest_screen_off_interval=bin_duration,
        )

