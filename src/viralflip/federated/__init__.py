"""Federated Learning infrastructure for privacy-preserving training.

This package implements federated learning for mobile health:
- Train across many users without centralizing raw data
- On-device personalization with global model updates
- Privacy-preserving aggregation methods
"""

from viralflip.federated.client import (
    FederatedClient,
    ClientUpdate,
    LocalTrainer,
)
from viralflip.federated.server import (
    FederatedServer,
    AggregationStrategy,
    FedAvg,
    FedProx,
    SecureAggregation,
)
from viralflip.federated.differential_privacy import (
    DPConfig,
    clip_gradients,
    add_noise,
    compute_privacy_budget,
)

__all__ = [
    "FederatedClient",
    "ClientUpdate",
    "LocalTrainer",
    "FederatedServer",
    "AggregationStrategy",
    "FedAvg",
    "FedProx",
    "SecureAggregation",
    "DPConfig",
    "clip_gradients",
    "add_noise",
    "compute_privacy_budget",
]

