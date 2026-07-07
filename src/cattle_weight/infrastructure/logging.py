"""Centralized logging configuration.

Call :func:`setup_logging` once at application startup (in ``app.py``'s
lifespan). All other modules should use :func:`get_logger` to get a named
logger — never call ``logging.basicConfig`` in library code.
"""
from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Configure application-wide logging to stdout.

    Args:
        level: Logging level string (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``).
               Case-insensitive. Falls back to ``INFO`` if invalid.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Avoid duplicate handlers if setup_logging is called more than once
    if not root_logger.handlers:
        root_logger.addHandler(handler)

    # Suppress noisy third-party loggers in production
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Call :func:`setup_logging` first at app startup.

    Args:
        name: Logger name, conventionally ``__name__`` of the calling module.

    Returns:
        A standard :class:`logging.Logger` instance.
    """
    return logging.getLogger(name)
