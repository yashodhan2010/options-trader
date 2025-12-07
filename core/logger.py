"""
Logging utility for the Options Trading Bot
"""
import logging
import sys
from logging.handlers import RotatingFileHandler
from config.settings import LOGGING_CONFIG, LOGS_DIR


def setup_logger(name: str = "OptionsTrader") -> logging.Logger:
    """
    Set up and return a configured logger instance.
    
    Args:
        name: Name for the logger
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOGGING_CONFIG["level"]))
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatters
    formatter = logging.Formatter(LOGGING_CONFIG["format"])
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    LOGS_DIR.mkdir(exist_ok=True)
    file_handler = RotatingFileHandler(
        LOGGING_CONFIG["file_path"],
        maxBytes=LOGGING_CONFIG["max_bytes"],
        backupCount=LOGGING_CONFIG["backup_count"],
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


# Create default logger
logger = setup_logger()
