"""I/O utilities for configuration and data handling."""

import pickle
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML configuration file.
    
    Args:
        config_path: Path to YAML configuration file.
        
    Returns:
        Configuration dictionary.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: dict[str, Any], config_path: str | Path) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary.
        config_path: Path to save YAML file.
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def load_pickle(path: str | Path) -> Any:
    """Load pickled object.
    
    Args:
        path: Path to pickle file.
        
    Returns:
        Unpickled object.
    """
    path = Path(path)
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj: Any, path: str | Path) -> None:
    """Save object to pickle file.
    
    Args:
        obj: Object to pickle.
        path: Path to save pickle file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def ensure_dir(path: str | Path) -> Path:
    """Ensure directory exists, creating if necessary.
    
    Args:
        path: Directory path.
        
    Returns:
        Path object for directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

