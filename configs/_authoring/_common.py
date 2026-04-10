"""Shared helpers for authored simulation configs.

Centralizes a few defaults so individual config modules stay short and
consistent. Each helper returns a plain dict (no hidden state).
"""

from __future__ import annotations

from typing import Any


# 1 year in seconds (Julian year, matches what the GFD literature uses).
YEAR_SECONDS: float = 365.25 * 86400.0


def default_timestepping(
    *,
    t1_seconds: float,
    dt: float = 600.0,
    save_interval_seconds: float = 86400.0,
    t0: float = 0.0,
) -> dict[str, Any]:
    """Build a ``timestepping`` block.

    Args:
        t1_seconds: Total integration window in seconds.
        dt: Initial time step (seconds). Constant stepper by default.
        save_interval_seconds: Spacing between saved snapshots.
        t0: Start time (seconds).
    """
    return {
        "t0": t0,
        "t1": float(t1_seconds),
        "dt": float(dt),
        "save_interval": float(save_interval_seconds),
    }


def default_debug(
    *,
    debug_nx: int = 32,
    debug_ny: int = 32,
    debug_t1_seconds: float = 86400.0,
    debug_save_interval_seconds: float = 600.0,
) -> dict[str, Any]:
    """Build a standard ``debug`` overrides block.

    The debug block is *only* applied when ``somax-sim run --debug`` is
    passed; otherwise it is ignored. The defaults here give a small,
    short, frequent-snapshot smoke run that finishes in seconds.

    Args:
        debug_nx: Override for ``testcase.grid.nx``.
        debug_ny: Override for ``testcase.grid.ny``.
        debug_t1_seconds: Override for ``timestepping.t1`` (default: 1 day).
        debug_save_interval_seconds: Override for ``timestepping.save_interval``
            (default: every 10 minutes — frequent snapshots).
    """
    return {
        "testcase": {"grid": {"nx": debug_nx, "ny": debug_ny}},
        "timestepping": {
            "t1": float(debug_t1_seconds),
            "save_interval": float(debug_save_interval_seconds),
        },
    }


def output_full() -> dict[str, Any]:
    """Output block: write everything (snapshots + metrics + final state)."""
    return {"write_snapshots": True, "write_metrics": True}


def output_spinup() -> dict[str, Any]:
    """Output block for spinup runs: only the final state matters."""
    return {"write_snapshots": False, "write_metrics": False}
