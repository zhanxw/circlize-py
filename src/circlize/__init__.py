"""Compatibility alias package re-exporting everything from circlize_py."""

import circlize_py as _circlize_py
from circlize_py import *  # noqa: F401, F403

__version__ = _circlize_py.__version__
__all__ = list(getattr(_circlize_py, "__all__", []))
