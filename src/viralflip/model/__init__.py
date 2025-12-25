"""Model components for ViralFlip."""

from viralflip.model.drift_score import DriftScoreModule, MultiModalityDriftScore
from viralflip.model.lag_lattice import LagLatticeHazardModel
from viralflip.model.interactions import InteractionModule
from viralflip.model.personalization import PersonalizationLayer
from viralflip.model.viralflip import (
    ViralFlip,
    ViralFlipOutput,
    VirusClassifier,
    ConfidenceScorer,
    EncoderBackedDriftScore,
    # Backwards compatibility aliases
    ViralFlipModel,
    ViralFlipX,
    ViralFlipXOutput,
)
from viralflip.model.virus_types import (
    VirusType, NUM_VIRUS_CLASSES, VIRUS_NAMES, VIRUS_SHORT_NAMES,
    get_virus_name, get_virus_from_name, list_virus_types,
)

__all__ = [
    # Core components
    "DriftScoreModule",
    "MultiModalityDriftScore",
    "LagLatticeHazardModel",
    "InteractionModule",
    "PersonalizationLayer",
    # Main model
    "ViralFlip",
    "ViralFlipOutput",
    # Supporting modules
    "VirusClassifier",
    "ConfidenceScorer",
    "EncoderBackedDriftScore",
    # Backwards compatibility aliases
    "ViralFlipModel",
    "ViralFlipX",
    "ViralFlipXOutput",
    # Virus classification
    "VirusType",
    "NUM_VIRUS_CLASSES",
    "VIRUS_NAMES",
    "VIRUS_SHORT_NAMES",
    "get_virus_name",
    "get_virus_from_name",
    "list_virus_types",
]
