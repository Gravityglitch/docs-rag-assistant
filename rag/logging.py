"""structlog configuration: JSON lines in production, coloured console locally.

Request ids are bound with ``structlog.contextvars`` by the API middleware so every log
line emitted while handling a request carries ``request_id``.
"""

from __future__ import annotations

import logging
import sys

import structlog

_CONFIGURED = False


def _stderr_logger(*_: object) -> structlog.PrintLogger:
    # Resolve sys.stderr per logger so stream redirection (pytest capture, uvicorn) is honoured.
    return structlog.PrintLogger(file=sys.stderr)


def configure_logging(json_logs: bool = True, level: str = "INFO") -> None:
    global _CONFIGURED
    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer() if json_logs else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping().get(level.upper(), logging.INFO)
        ),
        logger_factory=_stderr_logger,
        cache_logger_on_first_use=False,
    )
    logging.basicConfig(level=level.upper(), stream=sys.stderr, format="%(message)s")
    _CONFIGURED = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    if not _CONFIGURED:
        configure_logging(json_logs=False)
    return structlog.get_logger(name)
