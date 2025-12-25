"""User burden and compliance models for active sensing.

Models user behavior to optimize data collection requests:
- Compliance probability prediction
- Burden estimation
- Optimal timing
"""

from dataclasses import dataclass, field
from typing import Optional
import time

import numpy as np


@dataclass
class UserProfile:
    """User behavior profile."""
    
    user_id: str
    
    # Compliance patterns
    avg_compliance_rate: float = 0.5
    compliance_by_hour: dict[int, float] = field(default_factory=dict)
    compliance_by_modality: dict[str, float] = field(default_factory=dict)
    
    # Burden tolerance
    burden_tolerance: float = 0.5  # 0 = low tolerance, 1 = high
    
    # Response patterns
    avg_response_time_seconds: float = 300.0
    typical_active_hours: tuple[int, int] = (8, 22)
    
    # History
    n_requests: int = 0
    n_completed: int = 0


class UserBurdenModel:
    """Model user burden and tolerance.
    
    Estimates how burdensome data collection requests are
    and adapts to user preferences.
    """
    
    def __init__(
        self,
        base_burden_weights: dict[str, float] = None,
    ):
        """Initialize burden model.
        
        Args:
            base_burden_weights: Base burden weights per modality
        """
        self.base_weights = base_burden_weights or {
            "rppg": 0.8,        # High: requires camera access, stillness
            "gait_active": 0.7,  # High: requires walking
            "voice": 0.4,        # Medium: requires speaking
            "cough": 0.3,        # Low-Medium: quick
            "tap": 0.5,          # Medium: requires attention
            "light": 0.1,        # Low: passive
            "baro": 0.1,         # Low: passive
        }
        
        # User-specific adjustments
        self.user_adjustments: dict[str, dict[str, float]] = {}
    
    def estimate_burden(
        self,
        modality: str,
        user_id: str,
        context: Optional[dict] = None,
    ) -> float:
        """Estimate burden of collecting modality for user.
        
        Args:
            modality: Sensor modality
            user_id: User ID
            context: Optional context (time, activity, etc.)
            
        Returns:
            Burden estimate (0-1)
        """
        # Base burden
        base = self.base_weights.get(modality, 0.5)
        
        # User adjustment
        if user_id in self.user_adjustments:
            adjustment = self.user_adjustments[user_id].get(modality, 0.0)
            base = np.clip(base + adjustment, 0, 1)
        
        # Context adjustment
        if context is not None:
            base = self._apply_context_adjustment(base, context)
        
        return base
    
    def _apply_context_adjustment(
        self,
        base_burden: float,
        context: dict,
    ) -> float:
        """Adjust burden based on context."""
        adjusted = base_burden
        
        # Time of day
        if "hour" in context:
            hour = context["hour"]
            # Higher burden early morning and late evening
            if hour < 7 or hour > 21:
                adjusted *= 1.3
            # Lower burden during typical active hours
            elif 9 <= hour <= 18:
                adjusted *= 0.9
        
        # Activity level
        if "is_moving" in context and context["is_moving"]:
            # gait_active less burdensome if already walking
            if adjusted > 0.5:
                adjusted *= 0.8
        
        return np.clip(adjusted, 0, 1)
    
    def update_from_feedback(
        self,
        user_id: str,
        modality: str,
        completed: bool,
        response_time: Optional[float] = None,
    ) -> None:
        """Update model from user feedback.
        
        Args:
            user_id: User ID
            modality: Requested modality
            completed: Whether user completed request
            response_time: Time to respond (seconds)
        """
        if user_id not in self.user_adjustments:
            self.user_adjustments[user_id] = {}
        
        current = self.user_adjustments[user_id].get(modality, 0.0)
        
        # Adjust based on completion
        if not completed:
            # Increase perceived burden
            adjustment = 0.1
        else:
            # Decrease perceived burden
            adjustment = -0.05
            
            # Response time indicates burden
            if response_time is not None:
                if response_time > 600:  # > 10 minutes
                    adjustment += 0.05  # Still burdensome
                elif response_time < 60:  # < 1 minute
                    adjustment -= 0.05  # Easy
        
        self.user_adjustments[user_id][modality] = np.clip(
            current + adjustment, -0.5, 0.5
        )
    
    def get_least_burdensome(
        self,
        modalities: list[str],
        user_id: str,
        context: Optional[dict] = None,
    ) -> str:
        """Get least burdensome modality from list.
        
        Args:
            modalities: Available modalities
            user_id: User ID
            context: Optional context
            
        Returns:
            Least burdensome modality
        """
        burdens = {
            m: self.estimate_burden(m, user_id, context)
            for m in modalities
        }
        
        return min(burdens, key=burdens.get)


class CompliancePredictor:
    """Predict probability of user complying with collection request.
    
    Uses historical data and context to estimate compliance probability.
    """
    
    def __init__(
        self,
        default_compliance: float = 0.5,
        learning_rate: float = 0.1,
    ):
        """Initialize compliance predictor.
        
        Args:
            default_compliance: Default compliance probability
            learning_rate: Learning rate for updates
        """
        self.default_compliance = default_compliance
        self.learning_rate = learning_rate
        
        # User profiles
        self.profiles: dict[str, UserProfile] = {}
        
        # Global patterns
        self.global_hour_compliance = {h: 0.5 for h in range(24)}
        self.global_modality_compliance: dict[str, float] = {}
    
    def get_or_create_profile(self, user_id: str) -> UserProfile:
        """Get or create user profile."""
        if user_id not in self.profiles:
            self.profiles[user_id] = UserProfile(user_id=user_id)
        return self.profiles[user_id]
    
    def predict_compliance(
        self,
        user_id: str,
        modality: str,
        hour: int,
        context: Optional[dict] = None,
    ) -> float:
        """Predict compliance probability.
        
        Args:
            user_id: User ID
            modality: Requested modality
            hour: Hour of day (0-23)
            context: Optional context features
            
        Returns:
            Compliance probability (0-1)
        """
        profile = self.get_or_create_profile(user_id)
        
        # Start with user's base rate
        prob = profile.avg_compliance_rate
        
        # Adjust for hour
        if hour in profile.compliance_by_hour:
            hour_rate = profile.compliance_by_hour[hour]
            prob = 0.5 * prob + 0.5 * hour_rate
        elif hour in self.global_hour_compliance:
            global_hour = self.global_hour_compliance[hour]
            prob = 0.7 * prob + 0.3 * global_hour
        
        # Adjust for modality
        if modality in profile.compliance_by_modality:
            mod_rate = profile.compliance_by_modality[modality]
            prob = 0.5 * prob + 0.5 * mod_rate
        elif modality in self.global_modality_compliance:
            global_mod = self.global_modality_compliance[modality]
            prob = 0.7 * prob + 0.3 * global_mod
        
        # Context adjustments
        if context is not None:
            prob = self._apply_context_adjustment(prob, context)
        
        return np.clip(prob, 0.01, 0.99)
    
    def _apply_context_adjustment(
        self,
        base_prob: float,
        context: dict,
    ) -> float:
        """Adjust probability based on context."""
        adjusted = base_prob
        
        # Recent requests
        if "n_recent_requests" in context:
            n_recent = context["n_recent_requests"]
            # Fatigue effect
            if n_recent > 0:
                fatigue_factor = 0.9 ** n_recent
                adjusted *= fatigue_factor
        
        # Request priority
        if "priority" in context:
            if context["priority"] >= 3:  # High/Critical
                adjusted = min(adjusted * 1.2, 0.99)
        
        return adjusted
    
    def update(
        self,
        user_id: str,
        modality: str,
        hour: int,
        complied: bool,
    ) -> None:
        """Update model with observation.
        
        Args:
            user_id: User ID
            modality: Requested modality
            hour: Hour of day
            complied: Whether user complied
        """
        profile = self.get_or_create_profile(user_id)
        value = float(complied)
        
        # Update user profile
        profile.n_requests += 1
        if complied:
            profile.n_completed += 1
        
        profile.avg_compliance_rate = (
            (1 - self.learning_rate) * profile.avg_compliance_rate +
            self.learning_rate * value
        )
        
        # Update hour-specific
        if hour not in profile.compliance_by_hour:
            profile.compliance_by_hour[hour] = self.default_compliance
        
        profile.compliance_by_hour[hour] = (
            (1 - self.learning_rate) * profile.compliance_by_hour[hour] +
            self.learning_rate * value
        )
        
        # Update modality-specific
        if modality not in profile.compliance_by_modality:
            profile.compliance_by_modality[modality] = self.default_compliance
        
        profile.compliance_by_modality[modality] = (
            (1 - self.learning_rate) * profile.compliance_by_modality[modality] +
            self.learning_rate * value
        )
        
        # Update global patterns
        self.global_hour_compliance[hour] = (
            (1 - self.learning_rate * 0.1) * self.global_hour_compliance[hour] +
            self.learning_rate * 0.1 * value
        )
        
        if modality not in self.global_modality_compliance:
            self.global_modality_compliance[modality] = self.default_compliance
        
        self.global_modality_compliance[modality] = (
            (1 - self.learning_rate * 0.1) * self.global_modality_compliance[modality] +
            self.learning_rate * 0.1 * value
        )
    
    def get_best_time(
        self,
        user_id: str,
        modality: str,
        min_hour: int = 8,
        max_hour: int = 21,
    ) -> int:
        """Get best hour to request sensor from user.
        
        Args:
            user_id: User ID
            modality: Modality to request
            min_hour: Earliest hour
            max_hour: Latest hour
            
        Returns:
            Best hour (0-23)
        """
        best_hour = min_hour
        best_prob = 0.0
        
        for hour in range(min_hour, max_hour + 1):
            prob = self.predict_compliance(user_id, modality, hour)
            if prob > best_prob:
                best_prob = prob
                best_hour = hour
        
        return best_hour

