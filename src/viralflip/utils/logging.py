"""Logging utilities for ViralFlip."""

import logging
import sys
from pathlib import Path
from typing import Optional


_LOGGERS: dict[str, logging.Logger] = {}


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str | Path] = None,
    format_string: Optional[str] = None,
) -> None:
    """Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to log file.
        format_string: Optional custom format string.
    """
    if format_string is None:
        format_string = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with the given name.
    
    Args:
        name: Logger name (typically __name__).
        
    Returns:
        Logger instance.
    """
    if name not in _LOGGERS:
        _LOGGERS[name] = logging.getLogger(name)
    return _LOGGERS[name]

