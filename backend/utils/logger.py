"""
Logging configuration
"""
from loguru import logger
import sys
from pathlib import Path
from backend.config.settings import settings

def setup_logger():
    """Configure application logger"""
    # Remove default handler
    logger.remove()
    
    # If logging is disabled, return early with no handlers
    if not settings.enable_logging:
        return logger
    
    # Console handler with colors
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True
    )
    # File handler for all logs
    # logger.add(
    #     settings.logs_dir / "app_{time:YYYY-MM-DD}.log",
    #     rotation="1 day",
    #     retention="30 days",
    #     level="DEBUG",
    #     format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    # )
    # # Error file handler
    # logger.add(
    #     settings.logs_dir / "errors_{time:YYYY-MM-DD}.log",
    #     rotation="1 day",
    #     retention="90 days",
    #     level="ERROR",
    #     format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}\n{exception}"
    # )
    return logger
app_logger = setup_logger()