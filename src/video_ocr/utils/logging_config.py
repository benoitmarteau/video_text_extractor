"""Logging configuration for Video OCR."""

import logging
import sys
from typing import Optional

try:
    from rich.logging import RichHandler
    from rich.console import Console

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


_logger: Optional[logging.Logger] = None


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    rich_formatting: bool = True,
) -> logging.Logger:
    """
    Set up logging for the Video OCR application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path to write logs
        rich_formatting: Use rich console formatting if available

    Returns:
        Configured logger instance
    """
    global _logger

    logger = logging.getLogger("video_ocr")
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    if rich_formatting and RICH_AVAILABLE:
        console = Console(stderr=True)
        handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            show_time=True,
            show_path=False,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    logger.addHandler(handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(file_handler)

    _logger = logger
    return logger


def get_logger() -> logging.Logger:
    """
    Get the configured logger instance.

    Returns:
        Logger instance (creates default if not configured)
    """
    global _logger

    if _logger is None:
        _logger = setup_logging()

    return _logger
