"""
Logging Configuration

Centralized logging configuration for the SGP4 project.
All modules should use this logger for consistent, professional output.

Usage:
    from logging_config import get_logger
    
    logger = get_logger(__name__)
    logger.info("Satellite propagated successfully")
    logger.warning("TLE data is outdated")
    logger.error("SGP4 propagation failed")
"""

import logging
import sys
from typing import Optional

# Default logging format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> None:
    """
    Configure logging for the entire application.
    
    Parameters
    ----------
    level : int
        Logging level (e.g., logging.DEBUG, logging.INFO)
    log_file : str, optional
        Path to log file. If None, logs only to console.
    """
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        handlers=handlers
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Parameters
    ----------
    name : str
        Name of the logger (typically __name__)
        
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    return logging.getLogger(name)


# Configure default logging on module import
configure_logging()
