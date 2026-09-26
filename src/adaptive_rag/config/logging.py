"""
config.logging
--------------
Consistent structured logging configuration for AdaptiveRAG.
"""

import logging
import sys


def setup_logging(level: str | int = "INFO") -> None:
    """Configure root logger with consistent formatting."""
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
