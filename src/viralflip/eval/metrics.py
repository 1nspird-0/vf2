"""Evaluation metrics for illness onset prediction.

Primary metrics:
- AUPRC per horizon (24/48/72)
- AUROC per horizon
- Lead-time: fraction of episodes predicted early

Operational metrics:
- False alarms per user-week
- Sensitivity at fixed false alarm budgets
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


@dataclass
class EvaluationResults:
    """Container for evaluation results."""
    
    # Per-horizon metrics
    auprc: dict[int, float]
    auroc: dict[int, float]
    
    # Lead time metrics
    lead_time_frac: dict[int, float]  # Fraction with early warning
    mean_lead_time_hours: dict[int, float]  # Average lead time
    
    # Operational metrics
    false_alarms_per_week: dict[int, float]
    sensitivity_at_budget: dict[int, dict[float, float]]  # horizon -> {budget: sens}
    
    # Summary
    auprc_mean: float
    auroc_mean: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "auprc": self.auprc,
            "auroc": self.auroc,
            "lead_time_frac": self.lead_time_frac,
            "mean_lead_time_hours": self.mean_lead_time_hours,
            "false_alarms_per_week": self.false_alarms_per_week,
            "sensitivity_at_budget": self.sensitivity_at_budget,
            "auprc_mean": self.auprc_mean,
            "auroc_mean": self.auroc_mean,
        }


def compute_auprc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Compute Area Under Precision-Recall Curve.
    
    Args:
        y_true: Binary labels.
        y_pred: Predicted probabilities.
        
    Returns:
        AUPRC value.
    """
    n_positive = y_true.sum()
    if n_positive == 0:
        return 0.0
    if n_positive == len(y_true):
        return 1.0  # All positive samples, perfect precision at all thresholds
    return average_precision_score(y_true, y_pred)


def compute_auroc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Compute Area Under ROC Curve.
    
    Args:
        y_true: Binary labels.
        y_pred: Predicted probabilities.
        
    Returns:
        AUROC value.
    """
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return 0.5
    return roc_auc_score(y_true, y_pred)


def compute_lead_time(
    predictions: np.ndarray,
    onset_indices: list[int],
    threshold: float,
    bin_hours: int = 6,
    max_lookback_bins: int = 12,
) -> tuple[float, float]:
    """Compute lead time metrics.
    
    Args:
        predictions: Prediction array, shape (n_bins,).
        onset_indices: List of onset bin indices.
        threshold: Risk threshold for triggering alert.
        bin_hours: Hours per bin.
        max_lookback_bins: Maximum bins to look back for early warning.
        
    Returns:
        Tuple of (fraction_with_warning, mean_lead_time_hours).
    """
    if len(onset_indices) == 0:
        return 0.0, 0.0
    
    lead_times = []
    
    for onset_idx in onset_indices:
        # Look back from onset to find first threshold crossing
        lookback_start = max(0, onset_idx - max_lookback_bins)
        
        lead_time = None
        for t in range(lookback_start, onset_idx):
            if predictions[t] >= threshold:
                lead_time = (onset_idx - t) * bin_hours
                break
        
        if lead_time is not None:
            lead_times.append(lead_time)
    
    if len(lead_times) == 0:
        return 0.0, 0.0
    
    frac_with_warning = len(lead_times) / len(onset_indices)
    mean_lead_time = np.mean(lead_times)
    
    return frac_with_warning, mean_lead_time


def compute_false_alarm_rate(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    bins_per_week: int = 28,  # 6-hour bins
) -> float:
    """Compute false alarms per week.
    
    Args:
        predictions: Prediction array.
        labels: Label array.
        threshold: Alert threshold.
        bins_per_week: Number of bins per week.
        
    Returns:
        False alarms per week.
    """
    n_bins = len(predictions)
    n_weeks = n_bins / bins_per_week
    
    if n_weeks == 0:
        return 0.0
    
    # False alarms: predicted positive but true negative
    false_alarms = np.sum((predictions >= threshold) & (labels == 0))
    
    return false_alarms / n_weeks


def sensitivity_at_false_alarm_budget(
    predictions: np.ndarray,
    labels: np.ndarray,
    budget_per_week: float,
    bins_per_week: int = 28,
) -> float:
    """Compute sensitivity at a given false alarm budget.
    
    Args:
        predictions: Prediction array.
        labels: Label array.
        budget_per_week: Maximum acceptable false alarms per week.
        bins_per_week: Number of bins per week.
        
    Returns:
        Sensitivity at the budget.
    """
    n_bins = len(predictions)
    n_weeks = n_bins / bins_per_week
    
    if labels.sum() == 0:
        return 0.0
    
    # Maximum false alarms allowed
    max_false_alarms = int(budget_per_week * n_weeks)
    
    # Find threshold that gives at most max_false_alarms
    sorted_indices = np.argsort(-predictions)  # Descending
    
    false_alarm_count = 0
    true_positive_count = 0
    chosen_threshold = 1.0
    
    for idx in sorted_indices:
        if labels[idx] == 1:
            true_positive_count += 1
        else:
            false_alarm_count += 1
            if false_alarm_count > max_false_alarms:
                break
        chosen_threshold = predictions[idx]
    
    sensitivity = true_positive_count / labels.sum()
    
    return sensitivity


def evaluate_model(
    predictions: dict[int, np.ndarray],
    labels: dict[int, np.ndarray],
    onset_indices: Optional[list[int]] = None,
    horizons: list[int] = [24, 48, 72],
    thresholds: list[float] = [0.3],
    false_alarm_budgets: list[float] = [0.5, 1.0],
    bin_hours: int = 6,
) -> EvaluationResults:
    """Comprehensive model evaluation.
    
    Args:
        predictions: Dict mapping horizon to prediction array.
        labels: Dict mapping horizon to label array.
        onset_indices: Optional list of onset bin indices.
        horizons: List of prediction horizons.
        thresholds: Thresholds for lead time calculation.
        false_alarm_budgets: False alarm budgets per week.
        bin_hours: Hours per bin.
        
    Returns:
        EvaluationResults object.
    """
    auprc = {}
    auroc = {}
    lead_time_frac = {}
    mean_lead_time = {}
    false_alarms = {}
    sensitivity_budget = {}
    
    for horizon in horizons:
        if horizon not in predictions or horizon not in labels:
            continue
        
        pred = predictions[horizon]
        lab = labels[horizon]
        
        # Basic metrics
        auprc[horizon] = compute_auprc(lab, pred)
        auroc[horizon] = compute_auroc(lab, pred)
        
        # Lead time (using first threshold)
        if onset_indices is not None:
            frac, avg_lt = compute_lead_time(
                pred, onset_indices, thresholds[0], bin_hours
            )
            lead_time_frac[horizon] = frac
            mean_lead_time[horizon] = avg_lt
        else:
            lead_time_frac[horizon] = 0.0
            mean_lead_time[horizon] = 0.0
        
        # False alarm rate
        false_alarms[horizon] = compute_false_alarm_rate(
            pred, lab, thresholds[0]
        )
        
        # Sensitivity at budgets
        sensitivity_budget[horizon] = {}
        for budget in false_alarm_budgets:
            sens = sensitivity_at_false_alarm_budget(pred, lab, budget)
            sensitivity_budget[horizon][budget] = sens
    
    auprc_mean = np.mean(list(auprc.values())) if auprc else 0.0
    auroc_mean = np.mean(list(auroc.values())) if auroc else 0.0
    
    return EvaluationResults(
        auprc=auprc,
        auroc=auroc,
        lead_time_frac=lead_time_frac,
        mean_lead_time_hours=mean_lead_time,
        false_alarms_per_week=false_alarms,
        sensitivity_at_budget=sensitivity_budget,
        auprc_mean=auprc_mean,
        auroc_mean=auroc_mean,
    )

