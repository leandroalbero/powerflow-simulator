"""Client-facing failure text. The traceback stays in the log."""

import logging

logger = logging.getLogger(__name__)


def public_error(fallback: str) -> str:
    """Call from an ``except`` block. Never return ``str(exc)`` to a client."""
    logger.exception("%s", fallback)
    return fallback
