"""Conformal prediction for calibrated uncertainty.

This package implements conformal methods for coverage-guaranteed uncertainty
quantification, especially under change points (routine shifts).
"""

from viralflip.conformal.conformal_predictor import (
    ConformalPredictor,
    AdaptiveConformalPredictor,
    ChangePointAwareConformal,
    ConformalRiskSet,
)
from viralflip.conformal.nonconformity import (
    NonconformityScore,
    AbsoluteResidual,
    QuantileScore,
    AdaptiveScore,
)

__all__ = [
    "ConformalPredictor",
    "AdaptiveConformalPredictor",
    "ChangePointAwareConformal",
    "ConformalRiskSet",
    "NonconformityScore",
    "AbsoluteResidual",
    "QuantileScore",
    "AdaptiveScore",
]

