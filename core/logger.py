"""
Logging utility for the Options Trading Bot
"""
import logging
import sys
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
    
    # File handler (append mode, no rotation)
    LOGS_DIR.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(
        LOGGING_CONFIG["file_path"],
        mode="a",
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def setup_trade_logger(name: str = "TradeLog") -> logging.Logger:
    """
    Set up a dedicated logger for trades only.
    Logs to logs/trades.log with a clean format for easy reading.
    
    Args:
        name: Name for the logger
        
    Returns:
        Configured trade logger instance
    """
    trade_logger = logging.getLogger(name)
    trade_logger.setLevel(logging.INFO)
    
    # Prevent duplicate handlers
    if trade_logger.handlers:
        return trade_logger
    
    # Simple format for trade logs
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # Trade log file handler
    LOGS_DIR.mkdir(exist_ok=True)
    trade_file = LOGS_DIR / "trades.log"
    
    file_handler = logging.FileHandler(
        trade_file,
        mode="a",
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    trade_logger.addHandler(file_handler)
    
    # Don't propagate to root logger
    trade_logger.propagate = False
    
    return trade_logger


# Create default logger
logger = setup_logger()

# Create trade logger
trade_logger = setup_trade_logger()
