"""Logging configuration for the application with console output."""

import logging

__all__ = ["logger"]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False

# Only add a new handler if none exist
if not logger.handlers:
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
