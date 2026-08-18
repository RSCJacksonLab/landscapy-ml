"""Logging configuration shared by landscapy-ml workflows."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def _open_file_handler(path: Path) -> logging.FileHandler:
    return logging.FileHandler(path)


def configure_logger(
    log_file: Optional[str] = None, log_level: str = "INFO"
) -> logging.Logger:
    """
    Configure the package-level logger consistent with landscapy.

    Parameters
    ----------
    log_file : str, optional
        Optional path to a log file. When ``None``, logs are emitted to stdout.
    log_level : str, default="INFO"
        Logging level to set on the package logger.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger("landscapyml")
    level = getattr(logging, str(log_level).upper(), logging.INFO)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    for handler in logger.handlers:
        handler.setLevel(level)
        handler.setFormatter(formatter)

    if log_file:
        path = Path(log_file).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        existing_file_handler = next(
            (
                handler
                for handler in logger.handlers
                if isinstance(handler, logging.FileHandler)
                and Path(handler.baseFilename).resolve() == path
            ),
            None,
        )
        if existing_file_handler is None:
            file_handler = _open_file_handler(path)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    else:
        has_console_handler = any(
            isinstance(handler, logging.StreamHandler)
            and not isinstance(handler, logging.FileHandler)
            for handler in logger.handlers
        )
        if not has_console_handler:
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(level)
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)
    return logger
