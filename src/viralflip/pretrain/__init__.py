"""Self-supervised pretraining modules for ViralFlip-X.

This package implements masked multimodal self-supervised pretraining
for learning robust representations from unlabeled sensor streams.
"""

from viralflip.pretrain.masked_autoencoder import (
    MaskedMultimodalAutoencoder,
    MultimodalTimeSeriesEncoder,
    TimeSeriesDecoder,
    PatchEmbedding,
    create_mask_strategy,
)
from viralflip.pretrain.pretrain_trainer import (
    PretrainTrainer,
    PretrainConfig,
)

__all__ = [
    "MaskedMultimodalAutoencoder",
    "MultimodalTimeSeriesEncoder",
    "TimeSeriesDecoder",
    "PatchEmbedding",
    "create_mask_strategy",
    "PretrainTrainer",
    "PretrainConfig",
]

