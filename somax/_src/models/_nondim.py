"""Shared helpers for the ``from_nondimensional`` model factories.

Every factory does the same three things: validate its dimensionless
inputs, turn them into the SI-shaped kwargs ``create()`` already
accepts, and hand back the unit :class:`~somax._src.core.scales.Scales`
the resulting model runs in. The pieces that are common to more than
one model live here so the mappings stay stated once.
"""

from __future__ import annotations

import math


def require_positive(context: str, **values: float) -> None:
    """Raise if any named value is not finite and strictly positive."""
    for name, value in values.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"{context}: {name} must be a finite positive number; got {value!r}."
            )


def require_non_negative(context: str, **values: float) -> None:
    """Raise if any named value is negative or not finite."""
    for name, value in values.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"{context}: {name} must be a finite non-negative number; "
                f"got {value!r}."
            )
