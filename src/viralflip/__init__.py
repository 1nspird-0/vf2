"""
ViralFlip: Advanced Predictive Illness Onset Forecasting
=========================================================

This package implements state-of-the-art illness onset prediction using
passive and active phone sensor data with modern ML techniques:

Core Features:
- Feature extraction from voice, cough, tapping, gait, IMU, rPPG, GPS, and ambient sensors
- Personal Baseline Memory (PBM) for within-person normalization
- Behavior-Drift Debiasing (BDD) for confound removal
- Drift-Lattice Hazard Network (DLHN) for multi-horizon forecasting
- Interpretable explanations with counterfactual risk deltas

Advanced Features:
- Optional pretrained encoder for robust representations
- Virus type classification for illness characterization
- Conformal prediction for calibrated uncertainty with coverage guarantees
- Environment-invariant learning (IRM compatible)
- Change-point aware dynamic baselines for routine shift detection
- Federated learning infrastructure for privacy-preserving training
- Active sensing for value-of-information triggered data collection

IMPORTANT: This is NOT a medical diagnosis device. Output is an early warning
risk score with uncertainty, for research and education purposes only.
"""

__version__ = "0.3.0"
__author__ = "ViralFlip Team"

# Core model
from viralflip.model.viralflip import (
    ViralFlip,
    ViralFlipOutput,
    VirusClassifier,
    ConfidenceScorer,
    EncoderBackedDriftScore,
    # Backwards compatibility
    ViralFlipModel,
    ViralFlipX,
    ViralFlipXOutput,
)

# Baselines
from viralflip.baseline.pbm import PersonalBaselineMemory
from viralflip.baseline.changepoint import (
    ChangePointAwareBaseline,
    MultiModalityChangePointBaseline,
)

# Debiasing
from viralflip.debias.ridge import BehaviorDriftDebiaser

# Pretraining
from viralflip.pretrain.masked_autoencoder import (
    MaskedMultimodalAutoencoder,
    MultimodalTimeSeriesEncoder,
)

# Robust learning
from viralflip.robust.irm import IRMLoss, IRMPenalty, BehaviorEnvironmentDetector

# Conformal prediction
from viralflip.conformal.conformal_predictor import (
    ConformalPredictor,
    AdaptiveConformalPredictor,
    ChangePointAwareConformal,
    MultiHorizonConformal,
)

# Federated learning
from viralflip.federated.client import FederatedClient
from viralflip.federated.server import FederatedServer

# Active sensing
from viralflip.active.scheduler import ActiveSensingScheduler
from viralflip.active.acquisition import ExpectedInformationGain

__all__ = [
    # Core model
    "ViralFlip",
    "ViralFlipOutput",
    "VirusClassifier",
    "ConfidenceScorer",
    "EncoderBackedDriftScore",
    # Backwards compatibility
    "ViralFlipModel",
    "ViralFlipX",
    "ViralFlipXOutput",
    # Baselines
    "PersonalBaselineMemory",
    "ChangePointAwareBaseline",
    "MultiModalityChangePointBaseline",
    # Debiasing
    "BehaviorDriftDebiaser",
    # Pretraining
    "MaskedMultimodalAutoencoder",
    "MultimodalTimeSeriesEncoder",
    # Robust learning
    "IRMLoss",
    "IRMPenalty",
    "BehaviorEnvironmentDetector",
    # Conformal
    "ConformalPredictor",
    "AdaptiveConformalPredictor",
    "ChangePointAwareConformal",
    "MultiHorizonConformal",
    # Federated
    "FederatedClient",
    "FederatedServer",
    # Active sensing
    "ActiveSensingScheduler",
    "ExpectedInformationGain",
]
