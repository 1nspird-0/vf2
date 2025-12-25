"""Calibration metrics and reliability curves.

Calibration is crucial for interpretable risk scores.
We measure:
- Expected Calibration Error (ECE)
- Brier Score
- Reliability curves
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CalibrationMetrics:
    """Container for calibration metrics."""
    
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    brier: float  # Brier Score
    
    # Reliability curve data
    bin_centers: np.ndarray
    bin_accuracies: np.ndarray
    bin_confidences: np.ndarray
    bin_counts: np.ndarray
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "ece": self.ece,
            "mce": self.mce,
            "brier": self.brier,
            "bin_centers": self.bin_centers.tolist(),
            "bin_accuracies": self.bin_accuracies.tolist(),
            "bin_confidences": self.bin_confidences.tolist(),
            "bin_counts": self.bin_counts.tolist(),
        }


def compute_ece(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error.
    
    ECE = Σ (|B_m| / n) * |acc(B_m) - conf(B_m)|
    
    Args:
        y_true: Binary labels.
        y_pred: Predicted probabilities.
        n_bins: Number of calibration bins.
        
    Returns:
        ECE value.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    ece = 0.0
    n = len(y_true)
    
    for i in range(n_bins):
        low = bin_boundaries[i]
        high = bin_boundaries[i + 1]
        
        # Samples in this bin
        in_bin = (y_pred >= low) & (y_pred < high)
        prop_in_bin = in_bin.sum() / n
        
        if in_bin.sum() > 0:
            accuracy = y_true[in_bin].mean()
            confidence = y_pred[in_bin].mean()
            ece += prop_in_bin * abs(accuracy - confidence)
    
    return ece


def compute_mce(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Maximum Calibration Error.
    
    Args:
        y_true: Binary labels.
        y_pred: Predicted probabilities.
        n_bins: Number of calibration bins.
        
    Returns:
        MCE value.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    mce = 0.0
    
    for i in range(n_bins):
        low = bin_boundaries[i]
        high = bin_boundaries[i + 1]
        
        in_bin = (y_pred >= low) & (y_pred < high)
        
        if in_bin.sum() > 0:
            accuracy = y_true[in_bin].mean()
            confidence = y_pred[in_bin].mean()
            mce = max(mce, abs(accuracy - confidence))
    
    return mce


def compute_brier_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Compute Brier Score.
    
    Brier = (1/n) * Σ (y_pred - y_true)^2
    
    Args:
        y_true: Binary labels.
        y_pred: Predicted probabilities.
        
    Returns:
        Brier score (lower is better).
    """
    return np.mean((y_pred - y_true) ** 2)


def reliability_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute reliability curve data.
    
    Args:
        y_true: Binary labels.
        y_pred: Predicted probabilities.
        n_bins: Number of bins.
        
    Returns:
        Tuple of (bin_centers, bin_accuracies, bin_confidences, bin_counts).
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    
    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)
    
    for i in range(n_bins):
        low = bin_boundaries[i]
        high = bin_boundaries[i + 1]
        
        in_bin = (y_pred >= low) & (y_pred < high)
        bin_counts[i] = in_bin.sum()
        
        if bin_counts[i] > 0:
            bin_accuracies[i] = y_true[in_bin].mean()
            bin_confidences[i] = y_pred[in_bin].mean()
        else:
            bin_accuracies[i] = np.nan
            bin_confidences[i] = np.nan
    
    return bin_centers, bin_accuracies, bin_confidences, bin_counts


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> CalibrationMetrics:
    """Compute all calibration metrics.
    
    Args:
        y_true: Binary labels.
        y_pred: Predicted probabilities.
        n_bins: Number of bins for binned metrics.
        
    Returns:
        CalibrationMetrics object.
    """
    ece = compute_ece(y_true, y_pred, n_bins)
    mce = compute_mce(y_true, y_pred, n_bins)
    brier = compute_brier_score(y_true, y_pred)
    
    centers, accs, confs, counts = reliability_curve(y_true, y_pred, n_bins)
    
    return CalibrationMetrics(
        ece=ece,
        mce=mce,
        brier=brier,
        bin_centers=centers,
        bin_accuracies=accs,
        bin_confidences=confs,
        bin_counts=counts,
    )

