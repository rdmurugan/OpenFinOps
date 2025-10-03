"""Custom exceptions for the Vizly library."""

from __future__ import annotations


class VizlyError(Exception):
    """Base exception for Vizly-related errors."""


class ThemeNotFoundError(VizlyError):
    """Raised when a requested theme key is not registered."""


class ChartValidationError(VizlyError):
    """Raised when chart inputs fail validation."""
