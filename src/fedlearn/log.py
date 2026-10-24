"""fedlearn.log — the single logging voice for the whole project.

Every module and script logs through :func:`get_logger`, so the terminal tells
one coherent progress story:

    >>> from fedlearn.log import get_logger
    >>> log = get_logger("data.download")
    >>> log.info("downloading Ames ...")

Stage banners (:func:`stage`) map one-to-one to the groups in
``docs/fedmol_plan_report.html``; ``timed`` wraps any block and reports how
long it took. Color is added when stderr is a TTY and disabled otherwise.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import time
from typing import Iterator

_RESET = "\033[0m"
_LEVEL_STYLE = {
    logging.DEBUG: ("\033[2m", "DEBUG"),
    logging.INFO: ("\033[32m", "INFO "),
    logging.WARNING: ("\033[33m", "WARN "),
    logging.ERROR: ("\033[31m", "ERROR"),
    logging.CRITICAL: ("\033[1;31m", "CRIT "),
}
_BOLD = "\033[1m"
_DIM = "\033[2m"
_BAR_WIDTH = 88


class _Formatter(logging.Formatter):
    """Timestamped formatter that colors the level tag only when on a TTY."""

    def __init__(self) -> None:
        # fmt is bare "%(message)s": we build the prefix ourselves below so it
        # never gets emitted twice (log-level tag included).
        super().__init__(fmt="%(message)s", datefmt="%H:%M:%S")
        self._color = sys.stderr.isatty() and os.environ.get("NO_COLOR") is None

    def format(self, record: logging.LogRecord) -> str:
        stamp = self.formatTime(record, self.datefmt)
        msg = super().format(record)
        style, tag = _LEVEL_STYLE.get(record.levelno, ("", "????"))
        if self._color:
            return f"{_DIM}{stamp}{_RESET} {style}{tag}{_RESET} | {msg}"
        return f"{stamp} {tag} | {msg}"


def get_logger(name: str) -> logging.Logger:
    """Return a project logger, creating the shared console handler once."""
    logger = logging.getLogger(f"fedlearn.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_Formatter())
        logger.addHandler(handler)
        logger.setLevel(os.environ.get("FEDMOL_LOG_LEVEL", "INFO").upper())
        logger.propagate = False
    return logger


def banner(log: logging.Logger, title: str, width: int = _BAR_WIDTH) -> None:
    """Big section divider — the visual spine of every run."""
    log.info("")
    log.info("=" * width)
    log.info(_BOLD + f"  {title}" + _RESET)
    log.info("=" * width)


def stage(log: logging.Logger, current: int, total: int, title: str) -> None:
    """Progress banner matching the plan's groups, e.g. stage(1, 5, ...)."""
    log.info("")
    log.info("─" * _BAR_WIDTH)
    log.info(f"  STAGE {current}/{total} — {title}")
    log.info("─" * _BAR_WIDTH)


def section(log: logging.Logger, title: str) -> None:
    """Medium divider for logical sub-steps inside a stage."""
    log.info("")
    log.info(f"  › {title}")
    log.info("  " + "-" * 76)


@contextlib.contextmanager
def timed(log: logging.Logger, what: str) -> Iterator[None]:
    """Context manager that logs start/end of a block with elapsed time.

    >>> with timed(log, "featurizing 6,000 molecules"):
    ...     pass
    """
    started = time.perf_counter()
    log.info(f"  … {what}")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - started
        if elapsed < 60:
            log.info(f"  ✔ {what} — {elapsed:.2f}s")
        else:
            log.info(f"  ✔ {what} — {elapsed/60:.1f} min")


def summarize(label: str, value: object, log: logging.Logger) -> None:
    """One-line key-value progress telegraph (counts, fractions, metrics)."""
    log.info(f"    · {label:<44} {value}")
