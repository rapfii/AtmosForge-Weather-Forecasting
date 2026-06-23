"""Structured JSON logging for AtmosForge.

Provides a consistent logging setup with both console (rich) and
file (JSON) output formats for production-grade observability.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JSONFormatter(logging.Formatter):
    """Format log records as JSON lines for structured logging.

    Output format:
        {"timestamp": "...", "level": "INFO", "logger": "...", "message": "...", ...}
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as a JSON string.

        Args:
            record: The log record to format.

        Returns:
            JSON-formatted string.
        """
        log_entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Include exception info if present
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else "Unknown",
                "message": str(record.exc_info[1]),
            }

        # Include extra fields if provided
        extra_keys = set(record.__dict__.keys()) - set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys())
        for key in extra_keys:
            if key not in ("message", "msg"):
                log_entry[key] = getattr(record, key)

        return json.dumps(log_entry, default=str)


def setup_logger(
    name: str = "atmosforge",
    level: int = logging.INFO,
    log_dir: str | Path | None = None,
    console_output: bool = True,
    json_file: bool = True,
) -> logging.Logger:
    """Configure and return a structured logger.

    Args:
        name: Logger name (default: 'atmosforge').
        level: Logging level (default: INFO).
        log_dir: Directory for log files. If None, uses 'logs/'.
        console_output: Whether to output to console (default: True).
        json_file: Whether to write JSON log file (default: True).

    Returns:
        Configured logging.Logger instance.

    Example:
        >>> logger = setup_logger("atmosforge.training")
        >>> logger.info("Training started", extra={"model": "lstm", "epoch": 1})
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    # Console handler with readable format
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_format = logging.Formatter(
            fmt="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

    # JSON file handler for structured logging
    if json_file:
        log_path = Path(log_dir) if log_dir else Path("logs")
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            log_path / f"{name}.jsonl",
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)

    return logger
