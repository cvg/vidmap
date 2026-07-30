"""Process-wide logging configuration for VidMap runtime output."""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TextIO

LOGGER_NAMESPACE = "vidmap"

_CONTEXT: ContextVar[tuple[tuple[str, str], ...]] = ContextVar("vidmap_log_context", default=())

_LEVEL_COLORS = {
    logging.DEBUG: "\033[36m",
    logging.INFO: "\033[32m",
    logging.WARNING: "\033[33m",
    logging.ERROR: "\033[31m",
    logging.CRITICAL: "\033[1;31m",
}
_COLOR_RESET = "\033[0m"


def verbosity_to_level(verbosity: int) -> int:
    """Translate the public integer verbosity contract to standard log levels."""
    if verbosity < 0:
        raise ValueError("verbosity must be nonnegative")
    if verbosity == 0:
        return logging.WARNING
    if verbosity == 1:
        return logging.INFO
    return logging.DEBUG


class _RuntimeHandler(logging.StreamHandler):
    """Marker type for the single handler owned by VidMap."""


class _RuntimeContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        values = _CONTEXT.get()
        record.vidmap_context = "" if not values else "[" + " ".join(f"{key}={value}" for key, value in values) + "] "
        return True


class _RuntimeFormatter(logging.Formatter):
    def __init__(self, *, color: bool) -> None:
        super().__init__(
            fmt="%(asctime)s | %(vidmap_level)s | %(name)s | %(vidmap_context)s%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.color = color

    def format(self, record: logging.LogRecord) -> str:
        level = f"{record.levelname:<8}"
        if self.color:
            if record.levelno in _LEVEL_COLORS:
                level = f"{_LEVEL_COLORS[record.levelno]}{level}{_COLOR_RESET}"
        record.vidmap_level = level
        try:
            return super().format(record)
        finally:
            del record.vidmap_level


def _color_enabled(stream: TextIO) -> bool:
    """Use color only on an interactive terminal that has not opted out."""
    return os.environ.get("NO_COLOR") is None and os.environ.get("TERM") != "dumb" and stream.isatty()


def configure_logging(verbosity: int, *, stream: TextIO | None = None) -> logging.Logger:
    """Configure the ``vidmap`` namespace once without modifying the root logger."""
    logger = logging.getLogger(LOGGER_NAMESPACE)
    handler = next((candidate for candidate in logger.handlers if isinstance(candidate, _RuntimeHandler)), None)
    if handler is None:
        handler = _RuntimeHandler(sys.stderr if stream is None else stream)
        handler.addFilter(_RuntimeContextFilter())
        logger.addHandler(handler)
    elif stream is not None and isinstance(handler, logging.StreamHandler):
        handler.setStream(stream)

    handler.setFormatter(_RuntimeFormatter(color=_color_enabled(handler.stream)))
    handler.setLevel(logging.DEBUG)
    logger.setLevel(verbosity_to_level(verbosity))
    logger.disabled = False
    logger.propagate = False
    return logger


def progress_bars_enabled() -> bool:
    """Show interactive progress for lifecycle output attached to a terminal."""
    return logging.getLogger(LOGGER_NAMESPACE).isEnabledFor(logging.INFO) and sys.stderr.isatty()


@contextmanager
def log_context(**values: object) -> Iterator[None]:
    """Bind stable target fields to every VidMap record in the current context."""
    merged = dict(_CONTEXT.get())
    merged.update((key, str(value)) for key, value in values.items() if value is not None)
    token = _CONTEXT.set(tuple(merged.items()))
    try:
        yield
    finally:
        _CONTEXT.reset(token)
