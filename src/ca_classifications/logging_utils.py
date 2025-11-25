from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def configure_logger(log_file: Optional[str] = None, log_level: str = "INFO") -> logging.Logger:
    """
    Configure a package-level logger similar to landscapy.

    If log_file is provided, logs are written there; otherwise, logs go to stdout.
    """
    logger = logging.getLogger("ca_classifications")
    level = getattr(logging, str(log_level).upper(), logging.INFO)
    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(path)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == fh.baseFilename for h in logger.handlers):
            logger.addHandler(fh)
    else:
        if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
            sh = logging.StreamHandler()
            sh.setLevel(level)
            sh.setFormatter(formatter)
            logger.addHandler(sh)
    return logger
