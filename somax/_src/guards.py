"""In-JIT fail-fast tripwires for somax model computations.

These are tiny, JIT/``vmap``/``grad``-safe predicates that return their
input array unchanged so they drop transparently into a tendency
computation::

    h = guard_positive(h, where="layer thickness h_k")  # raises if h <= 0

They are built on :func:`equinox.error_if`, which raises *immediately* when
the predicate is true (unlike ``jax.experimental.checkify``, which defers the
raise to the end of the JIT region — wrong for fail-fast, since the blown-up
chunk would run to completion first). Equinox is a base somax dependency, so
no extra install is needed.

The runtime cost of the check, and what happens on failure, is governed by the
``EQX_ON_ERROR`` environment variable (``breakpoint`` while debugging,
``nan``/``off`` to disable for validated production sweeps). See the Equinox
docs for ``equinox.error_if`` and ``equinox.debug``.

These guards are the *in-JIT* counterpart to the host-side checks in the
``somax-sim`` runner (``_assert_finite_state``) and the :mod:`somax.monitor`
chunk-boundary monitors: a guard halts **at** the offending step, before the
rest of a multi-thousand-step chunk runs.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array


def guard_finite(x: Array, *, where: str) -> Array:
    """Return ``x`` unchanged, raising in-JIT if it holds any NaN/Inf.

    Args:
        x: Array to check (returned unchanged).
        where: Human-readable location for the error message (e.g.
            ``"RHS"`` or ``"layer thickness h_k"``).

    Returns:
        ``x`` unchanged. Raises at trace/run time via
        :func:`equinox.error_if` when any element is non-finite.
    """
    return eqx.error_if(x, jnp.any(~jnp.isfinite(x)), f"non-finite state in {where}")


def guard_positive(x: Array, *, where: str, floor: float = 0.0) -> Array:
    """Return ``x`` unchanged, raising in-JIT if any element ``<= floor``.

    The canonical use is layer-thickness positivity in multilayer SWM: the
    PV ``q_k = (zeta_k + f) / h_k`` is singular as ``h_k -> 0+``, so a
    non-positive thickness makes the remaining computation meaningless
    (FAIL-HARD).

    Args:
        x: Array to check (returned unchanged).
        where: Human-readable location for the error message.
        floor: Exclusive lower bound; elements must be strictly greater.
            Defaults to ``0.0``.

    Returns:
        ``x`` unchanged. Raises via :func:`equinox.error_if` when any
        element is ``<= floor``.
    """
    return eqx.error_if(
        x, jnp.any(x <= floor), f"{where} went non-positive (<= {floor})"
    )


def guard_ceiling(x: Array, *, where: str, ceil: float) -> Array:
    """Return ``x`` unchanged, raising in-JIT if ``|x|`` exceeds ``ceil``.

    A magnitude tripwire for velocity-state models — an opt-in blow-up
    ceiling that halts an obviously-diverging run early.

    Args:
        x: Array to check (returned unchanged).
        where: Human-readable location for the error message.
        ceil: Maximum allowed absolute value.

    Returns:
        ``x`` unchanged. Raises via :func:`equinox.error_if` when any
        ``|element| > ceil``.
    """
    return eqx.error_if(x, jnp.any(jnp.abs(x) > ceil), f"|{where}| exceeded {ceil}")
