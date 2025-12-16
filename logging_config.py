"""
Logging Configuration Module

Unified logging setup for the entire trading system.
All modules should import and use logger from this module.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

# Create logs directory if it doesn't exist
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Create log filename with timestamp
LOG_FILE = LOG_DIR / f"trading_system_{datetime.now().strftime('%Y%m%d')}.log"


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Setup and return a configured logger.
    
    Args:
        name: Logger name (typically __name__ of the module)
        level: Logging level (default: INFO)
    
    Returns:
        Configured logger instance
    
    Example:
        >>> from logging_config import setup_logger
        >>> logger = setup_logger(__name__)
        >>> logger.info("System started")
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        logger.handlers.clear()
    
    logger.setLevel(level)
    logger.propagate = False
    
    # Console handler - for stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    # File handler - for persistent logs
    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # More detailed in files
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


# Create default logger for immediate use
logger = setup_logger('trading_system')


if __name__ == "__main__":
    # Test logging configuration
    test_logger = setup_logger('test')
    
    test_logger.debug("This is a debug message")
    test_logger.info("This is an info message")
    test_logger.warning("This is a warning message")
    test_logger.error("This is an error message")
    
    print(f"\nLogs are being written to: {LOG_FILE}")
    print(f"Check the file for detailed logs.")
