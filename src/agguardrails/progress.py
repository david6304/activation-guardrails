"""Lightweight logging + progress reporting for long-running pipeline stages.

Designed for SLURM batch logs (`tail -f slurm-*.out`): periodic, newline-based
progress lines with rate and ETA rather than carriage-return progress bars that
bloat log files. Reuse ``configure_logging`` and ``ProgressLogger`` across stages
so every long job reports how far along it is.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

LOGGER_NAME = "agguardrails"


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the shared pipeline logger (idempotent)."""

    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)s %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger


def format_seconds(seconds: float) -> str:
    if seconds != seconds or seconds in (float("inf"), float("-inf")):  # NaN/inf
        return "?"
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


class ProgressLogger:
    """Log progress periodically with throughput and ETA.

    Call the instance with the running completed count. It logs the first item
    (to confirm work started), then at most once per ``min_interval_s``, and
    always on the final item.
    """

    def __init__(
        self,
        total: int,
        *,
        logger: Optional[logging.Logger] = None,
        label: str = "progress",
        min_interval_s: float = 30.0,
    ) -> None:
        self.total = total
        self.logger = logger or logging.getLogger(LOGGER_NAME)
        self.label = label
        self.min_interval_s = min_interval_s
        self._start = time.monotonic()
        self._last_log = 0.0

    def __call__(self, done: int, total: Optional[int] = None) -> None:
        total = self.total if total is None else total
        now = time.monotonic()
        is_final = done >= total
        if not is_final and done != 1 and (now - self._last_log) < self.min_interval_s:
            return
        self._last_log = now
        elapsed = now - self._start
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (total - done) / rate if rate > 0 else float("inf")
        pct = 100.0 * done / total if total else 100.0
        self.logger.info(
            "%s %d/%d (%.1f%%) | %.2f/s | elapsed %s | ETA %s",
            self.label,
            done,
            total,
            pct,
            rate,
            format_seconds(elapsed),
            format_seconds(eta),
        )
