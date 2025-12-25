"""Data loading utilities for ViralFlip.

This module provides loaders for real health datasets with illness labels.
"""

# Note: Imports are deferred to avoid circular imports
# Use these directly when needed:
#   from viralflip.data.real_data_loader import RealDataLoader
#   from viralflip.data.dataset import ViralFlipDataset

__all__ = [
    "RealDataLoader",
    "load_health_datasets", 
    "create_training_data",
    "ViralFlipDataset",
    "collate_viralflip_batch",
]


def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    if name in ["RealDataLoader", "load_health_datasets", "create_training_data"]:
        from viralflip.data.real_data_loader import RealDataLoader, load_health_datasets, create_training_data
        return locals()[name]
    elif name in ["ViralFlipDataset", "collate_viralflip_batch"]:
        from viralflip.data.dataset import ViralFlipDataset, collate_viralflip_batch
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

