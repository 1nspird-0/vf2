"""Active sensing scheduler for coordinating data collection.

Decides when and which sensors to request based on:
- Current uncertainty
- User availability and burden
- Battery and resource constraints
- Time of day and context
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import time

import numpy as np

from viralflip.active.acquisition import AcquisitionFunction, SensorValue


class SensorPriority(Enum):
    """Priority levels for sensor collection."""
    CRITICAL = 4  # Collect ASAP (high uncertainty, important)
    HIGH = 3      # Collect soon
    MEDIUM = 2    # Collect if convenient
    LOW = 1       # Optional
    SKIP = 0      # Don't collect


@dataclass
class CollectionRequest:
    """Request for sensor data collection."""
    
    modality: str
    priority: SensorPriority
    
    # Timing
    requested_at: float  # Unix timestamp
    expires_at: float    # Request expiration
    
    # Value estimates
    expected_value: SensorValue
    
    # Context
    reason: str = ""
    
    # Status
    completed: bool = False
    completed_at: Optional[float] = None
    actual_quality: Optional[float] = None


@dataclass
class SchedulerConfig:
    """Configuration for active sensing scheduler."""
    
    # Thresholds
    high_uncertainty_threshold: float = 0.4
    medium_uncertainty_threshold: float = 0.25
    
    # Request parameters
    request_expiry_seconds: float = 3600.0  # 1 hour
    max_pending_requests: int = 3
    min_request_interval_seconds: float = 300.0  # 5 minutes
    
    # Burden limits
    max_daily_burden: float = 5.0
    max_hourly_requests: int = 2
    
    # Time windows (24h format)
    collection_start_hour: int = 7
    collection_end_hour: int = 22
    
    # Sensor priorities (which sensors can be requested)
    requestable_sensors: list[str] = field(default_factory=lambda: [
        "rppg", "voice", "gait_active", "cough", "tap"
    ])


class ActiveSensingScheduler:
    """Schedules active sensor data collection.
    
    Coordinates when to prompt users for additional sensor data
    based on current prediction uncertainty and expected value.
    """
    
    def __init__(
        self,
        acquisition_fn: AcquisitionFunction,
        config: SchedulerConfig = None,
    ):
        """Initialize scheduler.
        
        Args:
            acquisition_fn: Acquisition function for computing sensor values
            config: Scheduler configuration
        """
        self.acquisition_fn = acquisition_fn
        self.config = config or SchedulerConfig()
        
        # Pending requests
        self.pending_requests: list[CollectionRequest] = []
        
        # History
        self.completed_requests: list[CollectionRequest] = []
        self.request_times: list[float] = []  # For rate limiting
        
        # Daily burden tracking
        self.daily_burden = 0.0
        self.last_burden_reset = time.time()
    
    def evaluate_sensors(
        self,
        current_prediction: float,
        current_uncertainty: float,
        modality_importances: dict[str, float],
        available_modalities: list[str] = None,
        context: Optional[np.ndarray] = None,
    ) -> list[SensorValue]:
        """Evaluate value of each requestable sensor.
        
        Args:
            current_prediction: Current illness risk prediction
            current_uncertainty: Current prediction uncertainty (e.g., from conformal)
            modality_importances: Learned importance weights per modality
            available_modalities: Which sensors are available to request
            context: Optional context features
            
        Returns:
            List of SensorValues sorted by priority (highest first)
        """
        available = available_modalities or self.config.requestable_sensors
        
        values = []
        for modality in available:
            importance = modality_importances.get(modality, 0.5)
            
            value = self.acquisition_fn.compute_value(
                modality=modality,
                current_prediction=current_prediction,
                current_uncertainty=current_uncertainty,
                modality_importance=importance,
            )
            values.append(value)
        
        # Sort by priority score
        values.sort(key=lambda v: v.priority_score, reverse=True)
        
        return values
    
    def should_request_sensor(
        self,
        current_prediction: float,
        current_uncertainty: float,
        modality_importances: dict[str, float],
    ) -> Optional[CollectionRequest]:
        """Determine if any sensor should be requested now.
        
        Args:
            current_prediction: Current risk prediction
            current_uncertainty: Current uncertainty
            modality_importances: Modality importance weights
            
        Returns:
            CollectionRequest if sensor should be requested, else None
        """
        # Check constraints
        if not self._can_make_request():
            return None
        
        # Evaluate sensors
        values = self.evaluate_sensors(
            current_prediction,
            current_uncertainty,
            modality_importances,
        )
        
        if not values:
            return None
        
        # Get highest priority sensor
        best_value = values[0]
        
        # Check if worth requesting
        if not self.acquisition_fn.should_acquire(best_value):
            return None
        
        # Determine priority level
        priority = self._determine_priority(
            current_uncertainty,
            best_value.priority_score,
        )
        
        if priority == SensorPriority.SKIP:
            return None
        
        # Create request
        request = self._create_request(best_value, priority)
        
        # Track
        self.pending_requests.append(request)
        self.request_times.append(time.time())
        self.daily_burden += best_value.user_burden
        
        return request
    
    def _can_make_request(self) -> bool:
        """Check if we can make a new request."""
        now = time.time()
        
        # Reset daily burden if new day
        if now - self.last_burden_reset > 86400:  # 24 hours
            self.daily_burden = 0.0
            self.last_burden_reset = now
        
        # Check daily burden limit
        if self.daily_burden >= self.config.max_daily_burden:
            return False
        
        # Check pending request limit
        self._clean_expired_requests()
        if len(self.pending_requests) >= self.config.max_pending_requests:
            return False
        
        # Check hourly rate limit
        hour_ago = now - 3600
        recent_requests = sum(1 for t in self.request_times if t > hour_ago)
        if recent_requests >= self.config.max_hourly_requests:
            return False
        
        # Check minimum interval
        if self.request_times:
            last_request = max(self.request_times)
            if now - last_request < self.config.min_request_interval_seconds:
                return False
        
        # Check time of day
        import datetime
        current_hour = datetime.datetime.now().hour
        if not (self.config.collection_start_hour <= current_hour < self.config.collection_end_hour):
            return False
        
        return True
    
    def _determine_priority(
        self,
        uncertainty: float,
        priority_score: float,
    ) -> SensorPriority:
        """Determine request priority."""
        if uncertainty > self.config.high_uncertainty_threshold:
            if priority_score > 0.8:
                return SensorPriority.CRITICAL
            elif priority_score > 0.5:
                return SensorPriority.HIGH
            else:
                return SensorPriority.MEDIUM
        elif uncertainty > self.config.medium_uncertainty_threshold:
            if priority_score > 0.7:
                return SensorPriority.HIGH
            elif priority_score > 0.4:
                return SensorPriority.MEDIUM
            else:
                return SensorPriority.LOW
        else:
            if priority_score > 0.8:
                return SensorPriority.MEDIUM
            else:
                return SensorPriority.SKIP
    
    def _create_request(
        self,
        value: SensorValue,
        priority: SensorPriority,
    ) -> CollectionRequest:
        """Create a collection request."""
        now = time.time()
        
        return CollectionRequest(
            modality=value.modality,
            priority=priority,
            requested_at=now,
            expires_at=now + self.config.request_expiry_seconds,
            expected_value=value,
            reason=f"High uncertainty, expected info gain: {value.expected_info_gain:.2f}",
        )
    
    def _clean_expired_requests(self) -> None:
        """Remove expired pending requests."""
        now = time.time()
        self.pending_requests = [
            r for r in self.pending_requests
            if r.expires_at > now and not r.completed
        ]
    
    def complete_request(
        self,
        modality: str,
        quality: float,
    ) -> Optional[CollectionRequest]:
        """Mark a request as completed.
        
        Args:
            modality: Completed modality
            quality: Quality of collected data
            
        Returns:
            Completed request if found
        """
        now = time.time()
        
        for request in self.pending_requests:
            if request.modality == modality and not request.completed:
                request.completed = True
                request.completed_at = now
                request.actual_quality = quality
                
                self.completed_requests.append(request)
                self.pending_requests.remove(request)
                
                return request
        
        return None
    
    def get_pending_requests(
        self,
        min_priority: SensorPriority = SensorPriority.LOW,
    ) -> list[CollectionRequest]:
        """Get pending requests above minimum priority.
        
        Args:
            min_priority: Minimum priority to include
            
        Returns:
            List of pending requests
        """
        self._clean_expired_requests()
        
        return [
            r for r in self.pending_requests
            if r.priority.value >= min_priority.value
        ]
    
    def get_next_request(self) -> Optional[CollectionRequest]:
        """Get highest priority pending request."""
        self._clean_expired_requests()
        
        if not self.pending_requests:
            return None
        
        return max(self.pending_requests, key=lambda r: r.priority.value)
    
    def get_stats(self) -> dict:
        """Get scheduler statistics."""
        return {
            "pending_requests": len(self.pending_requests),
            "completed_today": sum(
                1 for r in self.completed_requests
                if time.time() - r.completed_at < 86400
            ),
            "daily_burden_used": self.daily_burden,
            "daily_burden_remaining": self.config.max_daily_burden - self.daily_burden,
        }


class AdaptiveScheduler(ActiveSensingScheduler):
    """Adaptive scheduler that learns from collection outcomes.
    
    Adjusts request strategies based on:
    - Historical compliance rates
    - Actual information gain vs expected
    - User behavior patterns
    """
    
    def __init__(
        self,
        acquisition_fn: AcquisitionFunction,
        config: SchedulerConfig = None,
        learning_rate: float = 0.1,
    ):
        super().__init__(acquisition_fn, config)
        
        self.learning_rate = learning_rate
        
        # Learning state
        self.modality_compliance: dict[str, float] = {}  # Modality -> compliance rate
        self.modality_info_gain: dict[str, float] = {}   # Modality -> avg info gain
        self.hour_compliance: dict[int, float] = {h: 0.5 for h in range(24)}
    
    def update_from_outcome(
        self,
        request: CollectionRequest,
        actual_info_gain: float,
        complied: bool,
    ) -> None:
        """Update learning state from collection outcome.
        
        Args:
            request: The completed (or ignored) request
            actual_info_gain: Actual information gain (or 0 if not collected)
            complied: Whether user provided the data
        """
        modality = request.modality
        hour = int(request.requested_at // 3600) % 24
        
        # Update compliance rates
        if modality not in self.modality_compliance:
            self.modality_compliance[modality] = 0.5
        
        self.modality_compliance[modality] = (
            (1 - self.learning_rate) * self.modality_compliance[modality] +
            self.learning_rate * float(complied)
        )
        
        self.hour_compliance[hour] = (
            (1 - self.learning_rate) * self.hour_compliance[hour] +
            self.learning_rate * float(complied)
        )
        
        # Update info gain estimates
        if complied:
            if modality not in self.modality_info_gain:
                self.modality_info_gain[modality] = actual_info_gain
            else:
                self.modality_info_gain[modality] = (
                    (1 - self.learning_rate) * self.modality_info_gain[modality] +
                    self.learning_rate * actual_info_gain
                )
    
    def _determine_priority(
        self,
        uncertainty: float,
        priority_score: float,
    ) -> SensorPriority:
        """Determine priority with compliance adjustment."""
        base_priority = super()._determine_priority(uncertainty, priority_score)
        
        # Adjust based on current hour compliance
        import datetime
        current_hour = datetime.datetime.now().hour
        hour_compliance = self.hour_compliance.get(current_hour, 0.5)
        
        # If compliance is low, lower priority (less likely to bother user)
        if hour_compliance < 0.3 and base_priority.value > SensorPriority.LOW.value:
            return SensorPriority(base_priority.value - 1)
        
        return base_priority

