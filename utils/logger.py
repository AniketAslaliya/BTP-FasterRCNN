"""utils/logger.py — File + console logger."""

import logging
import sys
from pathlib import Path


def get_logger(name: str, log_dir: str, filename: str = "training.log") -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s — %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    # File handler
    fh = logging.FileHandler(Path(log_dir) / filename)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger
