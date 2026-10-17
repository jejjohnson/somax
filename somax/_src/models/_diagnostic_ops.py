"""finitevolX Tier-B operators for model ``diagnose()`` methods.

The C-grid models compute their diagnostics with ``finitevolx.pipekit``
operators (``KineticEnergy``, ``RelativeVorticity``, ``PotentialVorticity``,
``TotalEnergy``, ``DomainIntegral``, …) instead of re-implementing them, so a
stagger or ghost-ring convention lives in one place.

``finitevolx.pipekit`` imports pipekit, so it is imported lazily — on the
first ``diagnose()`` call, not when a model module is imported.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

from jaxtyping import Array


def fvx_ops() -> ModuleType:
    """The ``finitevolx.pipekit`` module (imported on first use)."""
    import finitevolx.pipekit as fp

    return fp


def grid_spec(grid: Any) -> tuple[int, int, float, float]:
    """``(nx_interior, ny_interior, Lx, Ly)`` of a ``CartesianGrid2D``."""
    return fvx_ops().as_grid_spec(grid)


def masked(mask: Any, stagger: str, x: Array) -> Array:
    """Zero ``x`` on the dry points of ``stagger`` (no-op without a mask).

    Matches a mask-aware Tier-A operator, which zeroes its output on the
    dry points of its output stagger.
    """
    if mask is None:
        return x
    return fvx_ops().ApplyMask(mask=mask, stagger=stagger)(x)
