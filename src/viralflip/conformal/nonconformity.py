"""Nonconformity score functions for conformal prediction.

Different score functions capture different aspects of prediction quality
and are suited for different tasks.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch


class NonconformityScore(ABC):
    """Abstract base class for nonconformity scores."""
    
    @abstractmethod
    def compute(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> np.ndarray:
        """Compute nonconformity scores.
        
        Args:
            predictions: Model predictions
            targets: True targets
            
        Returns:
            Nonconformity scores
        """
        pass
    
    @abstractmethod
    def invert(
        self,
        predictions: np.ndarray,
        quantile: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Invert to get prediction interval.
        
        Args:
            predictions: Model predictions
            quantile: Calibration quantile
            
        Returns:
            Tuple of (lower, upper) bounds
        """
        pass


class AbsoluteResidual(NonconformityScore):
    """Simple absolute residual score: |y - ŷ|."""
    
    def compute(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> np.ndarray:
        return np.abs(predictions - targets)
    
    def invert(
        self,
        predictions: np.ndarray,
        quantile: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        lower = predictions - quantile
        upper = predictions + quantile
        return lower, upper


class QuantileScore(NonconformityScore):
    """Asymmetric quantile score for quantile regression.
    
    score = max(α(ŷ - y), (1-α)(y - ŷ))
    
    Produces asymmetric intervals when combined with quantile predictions.
    """
    
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
    
    def compute(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> np.ndarray:
        return np.maximum(
            self.alpha * (predictions - targets),
            (1 - self.alpha) * (targets - predictions)
        )
    
    def invert(
        self,
        predictions: np.ndarray,
        quantile: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        # For quantile score, interval is asymmetric
        lower = predictions - quantile / (1 - self.alpha)
        upper = predictions + quantile / self.alpha
        return lower, upper


class AdaptiveScore(NonconformityScore):
    """Adaptive score that scales with prediction uncertainty.
    
    score = |y - ŷ| / σ(x)
    
    where σ(x) is the model's estimated uncertainty.
    Produces intervals that are wider when model is uncertain.
    """
    
    def __init__(self, min_sigma: float = 0.01):
        self.min_sigma = min_sigma
    
    def compute(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        sigmas: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if sigmas is None:
            # Estimate sigma from prediction (for probabilities)
            sigmas = np.sqrt(predictions * (1 - predictions)) + self.min_sigma
        
        return np.abs(predictions - targets) / np.maximum(sigmas, self.min_sigma)
    
    def invert(
        self,
        predictions: np.ndarray,
        quantile: float,
        sigmas: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if sigmas is None:
            sigmas = np.sqrt(predictions * (1 - predictions)) + self.min_sigma
        
        margin = quantile * sigmas
        lower = predictions - margin
        upper = predictions + margin
        return lower, upper


class LocallyAdaptiveScore(NonconformityScore):
    """Locally adaptive score using k-nearest neighbors.
    
    Adjusts intervals based on difficulty of local region in feature space.
    """
    
    def __init__(
        self,
        k: int = 20,
        min_scale: float = 0.1,
        max_scale: float = 3.0,
    ):
        self.k = k
        self.min_scale = min_scale
        self.max_scale = max_scale
        
        # Calibration data
        self.cal_features = None
        self.cal_residuals = None
    
    def fit(
        self,
        features: np.ndarray,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        """Fit on calibration data.
        
        Args:
            features: Feature matrix
            predictions: Model predictions
            targets: True targets
        """
        self.cal_features = features
        self.cal_residuals = np.abs(predictions - targets)
    
    def compute(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        residuals = np.abs(predictions - targets)
        
        if features is None or self.cal_features is None:
            return residuals
        
        # Get local scale from k-nearest neighbors
        scales = self._get_local_scales(features)
        return residuals / scales
    
    def invert(
        self,
        predictions: np.ndarray,
        quantile: float,
        features: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if features is None or self.cal_features is None:
            lower = predictions - quantile
            upper = predictions + quantile
            return lower, upper
        
        scales = self._get_local_scales(features)
        margin = quantile * scales
        
        lower = predictions - margin
        upper = predictions + margin
        return lower, upper
    
    def _get_local_scales(self, features: np.ndarray) -> np.ndarray:
        """Get local difficulty scales using k-NN."""
        from sklearn.neighbors import NearestNeighbors
        
        nn = NearestNeighbors(n_neighbors=min(self.k, len(self.cal_features)))
        nn.fit(self.cal_features)
        
        distances, indices = nn.kneighbors(features)
        
        # Local scale = mean absolute residual of neighbors
        scales = np.array([
            np.mean(self.cal_residuals[idx]) for idx in indices
        ])
        
        return np.clip(scales, self.min_scale, self.max_scale)


class TemporalScore(NonconformityScore):
    """Score that accounts for temporal autocorrelation.
    
    Uses a weighted average of recent residuals to adjust for
    persistent prediction errors.
    """
    
    def __init__(
        self,
        decay: float = 0.9,
        window: int = 10,
    ):
        self.decay = decay
        self.window = window
        
        # Running estimate
        self.running_mean = 0.0
        self.running_var = 1.0
        self.n_updates = 0
    
    def update(self, residual: float) -> None:
        """Update running statistics with new residual."""
        if self.n_updates == 0:
            self.running_mean = residual
            self.running_var = residual ** 2
        else:
            self.running_mean = self.decay * self.running_mean + (1 - self.decay) * residual
            self.running_var = (
                self.decay * self.running_var + 
                (1 - self.decay) * (residual - self.running_mean) ** 2
            )
        
        self.n_updates += 1
    
    def compute(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> np.ndarray:
        residuals = np.abs(predictions - targets)
        
        # Normalize by running statistics
        scale = np.sqrt(self.running_var) + 0.01
        return (residuals - self.running_mean) / scale
    
    def invert(
        self,
        predictions: np.ndarray,
        quantile: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        scale = np.sqrt(self.running_var) + 0.01
        margin = self.running_mean + quantile * scale
        
        lower = predictions - margin
        upper = predictions + margin
        return lower, upper
    
    def reset(self) -> None:
        """Reset running statistics."""
        self.running_mean = 0.0
        self.running_var = 1.0
        self.n_updates = 0


class CQRScore(NonconformityScore):
    """Conformalized Quantile Regression (CQR) score.
    
    Uses lower and upper quantile predictions to form intervals.
    score = max(q_lo - y, y - q_hi)
    """
    
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
    
    def compute_from_quantiles(
        self,
        q_lower: np.ndarray,
        q_upper: np.ndarray,
        targets: np.ndarray,
    ) -> np.ndarray:
        """Compute CQR score from quantile predictions.
        
        Args:
            q_lower: Lower quantile predictions (α/2)
            q_upper: Upper quantile predictions (1 - α/2)
            targets: True targets
            
        Returns:
            CQR nonconformity scores
        """
        return np.maximum(q_lower - targets, targets - q_upper)
    
    def compute(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> np.ndarray:
        # Without explicit quantiles, use symmetric approach
        return np.abs(predictions - targets)
    
    def invert_from_quantiles(
        self,
        q_lower: np.ndarray,
        q_upper: np.ndarray,
        quantile: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Invert CQR to get calibrated intervals.
        
        Args:
            q_lower: Lower quantile predictions
            q_upper: Upper quantile predictions
            quantile: Calibration quantile
            
        Returns:
            Tuple of (lower, upper) bounds
        """
        lower = q_lower - quantile
        upper = q_upper + quantile
        return lower, upper
    
    def invert(
        self,
        predictions: np.ndarray,
        quantile: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        lower = predictions - quantile
        upper = predictions + quantile
        return lower, upper

