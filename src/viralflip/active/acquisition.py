"""Acquisition functions for active sensing.

Determines when to request additional sensor data based on
expected information gain and current uncertainty.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import math

import numpy as np
import torch


@dataclass
class SensorValue:
    """Estimated value of collecting a sensor."""
    
    modality: str
    expected_info_gain: float
    uncertainty_reduction: float
    priority_score: float
    
    # Cost/burden factors
    user_burden: float
    battery_cost: float
    time_required: float  # seconds
    
    # Net value (gain - cost)
    net_value: float


class AcquisitionFunction(ABC):
    """Base class for acquisition functions."""
    
    @abstractmethod
    def compute_value(
        self,
        modality: str,
        current_prediction: float,
        current_uncertainty: float,
        modality_importance: float,
        **kwargs,
    ) -> SensorValue:
        """Compute value of acquiring a sensor.
        
        Args:
            modality: Sensor modality to evaluate
            current_prediction: Current risk prediction
            current_uncertainty: Current prediction uncertainty
            modality_importance: Learned importance of this modality
            
        Returns:
            SensorValue with acquisition value
        """
        pass
    
    @abstractmethod
    def should_acquire(
        self,
        value: SensorValue,
        threshold: float,
    ) -> bool:
        """Determine if sensor should be acquired.
        
        Args:
            value: Computed sensor value
            threshold: Acquisition threshold
            
        Returns:
            True if sensor should be requested
        """
        pass


class ExpectedInformationGain(AcquisitionFunction):
    """Acquisition based on expected information gain.
    
    Estimates how much the sensor will reduce prediction uncertainty,
    using learned modality-specific noise models.
    """
    
    def __init__(
        self,
        modality_noise: dict[str, float],
        modality_burden: dict[str, float],
        gain_scale: float = 1.0,
    ):
        """Initialize EIG acquisition.
        
        Args:
            modality_noise: Expected noise variance per modality
            modality_burden: User burden score per modality (0-1)
            gain_scale: Scaling factor for information gain
        """
        self.modality_noise = modality_noise
        self.modality_burden = modality_burden
        self.gain_scale = gain_scale
        
        # Sensor costs
        self.battery_costs = {
            "rppg": 0.3,
            "gait_active": 0.2,
            "voice": 0.1,
            "cough": 0.05,
            "tap": 0.05,
        }
        
        self.time_required = {
            "rppg": 30.0,  # seconds
            "gait_active": 20.0,
            "voice": 10.0,
            "cough": 5.0,
            "tap": 10.0,
        }
    
    def compute_value(
        self,
        modality: str,
        current_prediction: float,
        current_uncertainty: float,
        modality_importance: float,
        **kwargs,
    ) -> SensorValue:
        """Compute expected information gain."""
        # Get modality-specific parameters
        noise_var = self.modality_noise.get(modality, 0.5)
        burden = self.modality_burden.get(modality, 0.5)
        battery = self.battery_costs.get(modality, 0.1)
        time_req = self.time_required.get(modality, 10.0)
        
        # Compute expected uncertainty reduction
        # Using Bayesian update formula for variance reduction
        prior_var = current_uncertainty ** 2
        posterior_var = 1.0 / (1.0 / prior_var + 1.0 / noise_var)
        uncertainty_reduction = math.sqrt(prior_var) - math.sqrt(posterior_var)
        
        # Information gain (reduction in entropy)
        # H(prior) - H(posterior) ≈ 0.5 * log(prior_var / posterior_var)
        info_gain = 0.5 * math.log(prior_var / (posterior_var + 1e-8) + 1e-8)
        info_gain = max(0, info_gain) * self.gain_scale
        
        # Weight by modality importance
        weighted_gain = info_gain * modality_importance
        
        # Priority score (higher = more valuable)
        # Balance information gain against burden
        priority = weighted_gain / (burden + 0.1)
        
        # Net value (gain minus cost)
        cost = burden + 0.1 * battery + 0.01 * time_req
        net_value = weighted_gain - cost
        
        return SensorValue(
            modality=modality,
            expected_info_gain=info_gain,
            uncertainty_reduction=uncertainty_reduction,
            priority_score=priority,
            user_burden=burden,
            battery_cost=battery,
            time_required=time_req,
            net_value=net_value,
        )
    
    def should_acquire(
        self,
        value: SensorValue,
        threshold: float = 0.5,
    ) -> bool:
        """Acquire if net value exceeds threshold."""
        return value.net_value > threshold


class UncertaintyThreshold(AcquisitionFunction):
    """Simple threshold-based acquisition.
    
    Request sensor if current uncertainty is above threshold
    and sensor can help reduce it.
    """
    
    def __init__(
        self,
        uncertainty_threshold: float = 0.3,
        modality_burden: dict[str, float] = None,
    ):
        """Initialize uncertainty threshold acquisition.
        
        Args:
            uncertainty_threshold: Uncertainty level to trigger acquisition
            modality_burden: User burden per modality
        """
        self.uncertainty_threshold = uncertainty_threshold
        self.modality_burden = modality_burden or {}
    
    def compute_value(
        self,
        modality: str,
        current_prediction: float,
        current_uncertainty: float,
        modality_importance: float,
        **kwargs,
    ) -> SensorValue:
        """Compute value based on uncertainty."""
        burden = self.modality_burden.get(modality, 0.5)
        
        # Simple: value is proportional to uncertainty * importance
        info_gain = current_uncertainty * modality_importance
        priority = info_gain / (burden + 0.1)
        net_value = info_gain - burden
        
        return SensorValue(
            modality=modality,
            expected_info_gain=info_gain,
            uncertainty_reduction=current_uncertainty * 0.5,  # Rough estimate
            priority_score=priority,
            user_burden=burden,
            battery_cost=0.1,
            time_required=10.0,
            net_value=net_value,
        )
    
    def should_acquire(
        self,
        value: SensorValue,
        threshold: float = 0.5,
    ) -> bool:
        """Acquire if uncertainty is high enough."""
        return value.expected_info_gain > threshold


class BayesianOptimalDesign(AcquisitionFunction):
    """Bayesian optimal experimental design for sensor selection.
    
    Uses mutual information to select which sensors would be
    most informative for the current state.
    """
    
    def __init__(
        self,
        modality_models: dict[str, any],  # Trained noise models per modality
        n_samples: int = 100,
    ):
        """Initialize Bayesian optimal design.
        
        Args:
            modality_models: Noise/prediction models per modality
            n_samples: Number of Monte Carlo samples
        """
        self.modality_models = modality_models
        self.n_samples = n_samples
        
        self.modality_burden = {
            "rppg": 0.8,
            "gait_active": 0.7,
            "voice": 0.3,
            "cough": 0.2,
            "tap": 0.4,
            "light": 0.1,
            "baro": 0.1,
            "gps": 0.1,
            "imu_passive": 0.05,
            "screen": 0.0,
        }
    
    def compute_value(
        self,
        modality: str,
        current_prediction: float,
        current_uncertainty: float,
        modality_importance: float,
        current_state: Optional[np.ndarray] = None,
        **kwargs,
    ) -> SensorValue:
        """Compute value using Bayesian optimal design."""
        burden = self.modality_burden.get(modality, 0.5)
        
        # Estimate mutual information via Monte Carlo
        info_gain = self._estimate_mutual_information(
            modality,
            current_prediction,
            current_uncertainty,
            current_state,
        )
        
        # Weight by importance
        weighted_gain = info_gain * modality_importance
        
        # Priority and net value
        priority = weighted_gain / (burden + 0.1)
        net_value = weighted_gain - burden
        
        # Estimate uncertainty reduction
        uncertainty_reduction = current_uncertainty * min(1.0, info_gain)
        
        return SensorValue(
            modality=modality,
            expected_info_gain=info_gain,
            uncertainty_reduction=uncertainty_reduction,
            priority_score=priority,
            user_burden=burden,
            battery_cost=0.1,
            time_required=10.0,
            net_value=net_value,
        )
    
    def _estimate_mutual_information(
        self,
        modality: str,
        current_pred: float,
        current_unc: float,
        current_state: Optional[np.ndarray],
    ) -> float:
        """Estimate mutual information between sensor and illness state."""
        # Simple approximation using uncertainty
        # Real implementation would use learned models
        
        if modality not in self.modality_models:
            # Use heuristic based on uncertainty
            return current_unc * 0.5
        
        # Monte Carlo estimation
        # Sample illness states from current belief
        illness_samples = np.random.normal(
            current_pred, current_unc, self.n_samples
        )
        illness_samples = np.clip(illness_samples, 0, 1)
        
        # Compute expected entropy reduction
        # H(Y) - E[H(Y|X)]
        prior_entropy = -np.mean(
            illness_samples * np.log(illness_samples + 1e-8) +
            (1 - illness_samples) * np.log(1 - illness_samples + 1e-8)
        )
        
        # Approximate posterior entropy (would use model in real implementation)
        posterior_uncertainty = current_unc * 0.7  # Rough estimate
        posterior_samples = np.clip(
            illness_samples + np.random.normal(0, posterior_uncertainty, self.n_samples),
            0, 1
        )
        posterior_entropy = -np.mean(
            posterior_samples * np.log(posterior_samples + 1e-8) +
            (1 - posterior_samples) * np.log(1 - posterior_samples + 1e-8)
        )
        
        mutual_info = max(0, prior_entropy - posterior_entropy)
        return mutual_info
    
    def should_acquire(
        self,
        value: SensorValue,
        threshold: float = 0.3,
    ) -> bool:
        """Acquire if mutual information exceeds threshold."""
        return value.net_value > threshold


class ContextualBandit(AcquisitionFunction):
    """Contextual bandit for learning optimal acquisition strategy.
    
    Learns from experience which sensors are most valuable in
    different contexts (time of day, recent data, etc.)
    """
    
    def __init__(
        self,
        modalities: list[str],
        context_dim: int = 10,
        learning_rate: float = 0.01,
    ):
        """Initialize contextual bandit.
        
        Args:
            modalities: List of available modalities
            context_dim: Dimension of context features
            learning_rate: Learning rate for updates
        """
        self.modalities = modalities
        self.context_dim = context_dim
        self.learning_rate = learning_rate
        
        # Linear model parameters per modality
        self.weights = {
            m: np.zeros(context_dim) for m in modalities
        }
        self.counts = {m: 0 for m in modalities}
        
        # Burden (fixed)
        self.modality_burden = {
            "rppg": 0.8,
            "gait_active": 0.7,
            "voice": 0.3,
            "cough": 0.2,
            "tap": 0.4,
        }
    
    def compute_value(
        self,
        modality: str,
        current_prediction: float,
        current_uncertainty: float,
        modality_importance: float,
        context: Optional[np.ndarray] = None,
        **kwargs,
    ) -> SensorValue:
        """Compute value using learned model."""
        if context is None:
            context = np.zeros(self.context_dim)
        
        burden = self.modality_burden.get(modality, 0.5)
        
        # Predicted value from linear model
        if modality in self.weights:
            predicted_value = np.dot(self.weights[modality], context)
            # Add UCB exploration bonus
            exploration = 0.5 * np.sqrt(np.log(sum(self.counts.values()) + 1) / (self.counts[modality] + 1))
            ucb_value = predicted_value + exploration
        else:
            ucb_value = 0.5  # Default
        
        info_gain = max(0, ucb_value) * modality_importance
        priority = info_gain / (burden + 0.1)
        net_value = info_gain - burden
        
        return SensorValue(
            modality=modality,
            expected_info_gain=info_gain,
            uncertainty_reduction=current_uncertainty * 0.3,
            priority_score=priority,
            user_burden=burden,
            battery_cost=0.1,
            time_required=10.0,
            net_value=net_value,
        )
    
    def update(
        self,
        modality: str,
        context: np.ndarray,
        reward: float,
    ) -> None:
        """Update model with observed reward.
        
        Args:
            modality: Acquired modality
            context: Context when acquired
            reward: Observed reward (e.g., uncertainty reduction)
        """
        if modality not in self.weights:
            return
        
        # Gradient update
        predicted = np.dot(self.weights[modality], context)
        error = reward - predicted
        self.weights[modality] += self.learning_rate * error * context
        self.counts[modality] += 1
    
    def should_acquire(
        self,
        value: SensorValue,
        threshold: float = 0.3,
    ) -> bool:
        """Acquire based on UCB value."""
        return value.net_value > threshold

