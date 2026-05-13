"""
logger_setup.py — Centralised logging setup
============================================
Import once at the top of main_script.py.
All other modules call logging.getLogger(__name__) normally.
"""

import logging
import sys
from datetime import datetime

import config


def setup(level: str = None, log_to_file: bool = None) -> None:
    level       = level or config.LOG_LEVEL
    log_to_file = log_to_file if log_to_file is not None else config.LOG_TO_FILE

    numeric = getattr(logging, level.upper(), logging.INFO)
    fmt = logging.Formatter(
        fmt    = "%(asctime)s  %(levelname)-8s  %(name)-28s  %(message)s",
        datefmt= "%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(numeric)
    root.handlers.clear()

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(numeric)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    if log_to_file:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = config.LOGS_DIR / f"run_{ts}.log"
        fh   = logging.FileHandler(path, encoding="utf-8")
        fh.setLevel(numeric)
        fh.setFormatter(fmt)
        root.addHandler(fh)


setup()
