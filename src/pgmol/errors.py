"""Exception hierarchy for :mod:`pgmol`."""

from __future__ import annotations

__all__ = ["PGMolError", "SymmetryError", "XYZFormatError"]


class PGMolError(Exception):
    """Base class for every error raised by :mod:`pgmol`."""


class XYZFormatError(PGMolError, ValueError):
    """Raised when XYZ input is unreadable, truncated or inconsistent.

    Subclasses :class:`ValueError` so that callers who only guard against
    bad values keep working.
    """


class SymmetryError(PGMolError, ValueError):
    """Raised when symmetry analysis is given invalid parameters."""
