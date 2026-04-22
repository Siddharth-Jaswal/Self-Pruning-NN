"""Project logging configuration helpers."""

from __future__ import annotations

import logging
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def get_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    """Create or retrieve a configured project logger.

    Args:
        name: Logger name.
        log_file: Optional file path for debug logging output.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT)

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file is not None:
        resolved_log_file = Path(log_file)
        resolved_log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file_path = resolved_log_file.resolve()
        existing_file_handler = next(
            (
                handler
                for handler in logger.handlers
                if isinstance(handler, logging.FileHandler)
                and Path(handler.baseFilename) == log_file_path
            ),
            None,
        )
        if existing_file_handler is None:
            file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
