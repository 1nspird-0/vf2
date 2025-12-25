"""Seed management for reproducibility."""

import random
from typing import Optional

import numpy as np
import torch


_GLOBAL_SEED: Optional[int] = None


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Integer seed value.
    """
    global _GLOBAL_SEED
    _GLOBAL_SEED = seed
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For reproducibility (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_rng(seed: Optional[int] = None) -> np.random.Generator:
    """Get a numpy random generator with optional seed.
    
    Args:
        seed: Optional seed. If None, uses global seed or random.
        
    Returns:
        Numpy random Generator instance.
    """
    if seed is None:
        seed = _GLOBAL_SEED
    return np.random.default_rng(seed)


def get_global_seed() -> Optional[int]:
    """Get the globally set seed.
    
    Returns:
        Global seed value or None if not set.
    """
    return _GLOBAL_SEED

