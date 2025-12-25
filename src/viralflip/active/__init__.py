"""Active Sensing for value-of-information triggered data collection.

This package implements active sensing strategies that choose when
to request high-burden sensor data (rPPG, active gait test) based
on uncertainty and expected information gain.
"""

from viralflip.active.acquisition import (
    AcquisitionFunction,
    ExpectedInformationGain,
    UncertaintyThreshold,
    BayesianOptimalDesign,
)
from viralflip.active.scheduler import (
    ActiveSensingScheduler,
    SensorPriority,
    CollectionRequest,
)
from viralflip.active.user_model import (
    UserBurdenModel,
    CompliancePredictor,
)

__all__ = [
    "AcquisitionFunction",
    "ExpectedInformationGain",
    "UncertaintyThreshold",
    "BayesianOptimalDesign",
    "ActiveSensingScheduler",
    "SensorPriority",
    "CollectionRequest",
    "UserBurdenModel",
    "CompliancePredictor",
]

