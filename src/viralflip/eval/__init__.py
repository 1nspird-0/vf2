"""Evaluation modules for ViralFlip."""

from viralflip.eval.metrics import (
    compute_auprc,
    compute_auroc,
    compute_lead_time,
    compute_false_alarm_rate,
    evaluate_model,
)
from viralflip.eval.calibration import (
    compute_ece,
    compute_brier_score,
    reliability_curve,
    CalibrationMetrics,
)
from viralflip.eval.ablations import AblationRunner

__all__ = [
    "compute_auprc",
    "compute_auroc",
    "compute_lead_time",
    "compute_false_alarm_rate",
    "evaluate_model",
    "compute_ece",
    "compute_brier_score",
    "reliability_curve",
    "CalibrationMetrics",
    "AblationRunner",
]

