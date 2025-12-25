"""Shift-robust and confounder-invariant learning modules.

This package implements methods to learn features that remain predictive
across different "environments" (distributions), addressing spurious correlations.
"""

from viralflip.robust.irm import (
    IRMPenalty,
    IRMLoss,
    EnvironmentClassifier,
    BehaviorEnvironmentDetector,
)
from viralflip.robust.domain_adversarial import (
    GradientReversalLayer,
    DomainAdversarialLoss,
    DomainDiscriminator,
)

__all__ = [
    "IRMPenalty",
    "IRMLoss",
    "EnvironmentClassifier",
    "BehaviorEnvironmentDetector",
    "GradientReversalLayer",
    "DomainAdversarialLoss",
    "DomainDiscriminator",
]

