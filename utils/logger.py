# utils/logger.py
# Structured JSON logger. Use this everywhere instead of print().

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

from config.settings import settings


class JSONFormatter(logging.Formatter):
    """
    JSON logger for production.
    """

    def format(
        self,
        record: logging.LogRecord,
    ) -> str:

        log = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "message": record.getMessage(),
        }

        if record.exc_info:
            log["exception"] = self.formatException(
                record.exc_info
            )

        return json.dumps(log)


class StandardFormatter(logging.Formatter):
    """
    Human-readable logger for development.
    """

    def __init__(self):

        super().__init__(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


def get_logger(
    name: str,
) -> logging.Logger:

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)

    if settings.log_json:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(StandardFormatter())

    logger.addHandler(handler)

    logger.setLevel(
        getattr(
            logging,
            settings.log_level.upper(),
            logging.INFO,
        )
    )

    logger.propagate = False

    return logger
