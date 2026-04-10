"""Optional preflight + postflight assertions for somax-sim runs.

Assertions are configured per-run via the ``assertions`` block of the
:class:`RunSpec`. Each entry maps an assertion *name* (a key in
:data:`PREFLIGHT_ASSERTIONS` or :data:`POSTFLIGHT_ASSERTIONS`) to its
parameters as a dict.

There are two phases:

- **Preflight** runs *before* integration starts. It takes ``(spec, model)``
  and is the right place for cheap consistency checks: CFL, parameter
  bounds, layer-count consistency, etc.
- **Postflight** runs *after* metrics are computed but *before* they are
  written to disk. It takes ``(spec, metrics)`` where ``metrics`` is the
  flat dict that will become ``metrics.json``. This is the right place
  for output-side validation: scalar bounds, conservation tolerances,
  expected ranges.

Both phases raise :class:`AssertionFailedError` on failure, which the
runner translates into a non-zero exit code (so DVC stages and CI fail).

Why pluggable?
--------------
Different models care about different invariants. Multilayer SWM has a
gravity-wave CFL; QG has an advection CFL; ODE models have neither.
Rather than hard-coding one CFL formula, we let users opt in to the
assertions that match their model.

Adding a new assertion
----------------------
Write a function with one of the two signatures and add it to the
matching registry below. Names must be unique across both registries.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np


if TYPE_CHECKING:
    from somax._src.cli.spec import RunSpec


# ----------------------------------------------------------------------
# Exception
# ----------------------------------------------------------------------


class AssertionFailedError(RuntimeError):
    """Raised when an opt-in preflight or postflight assertion fails."""


# ----------------------------------------------------------------------
# Preflight assertions: (spec, model) -> None
# ----------------------------------------------------------------------


def check_cfl(
    spec: RunSpec,
    model: Any,
    *,
    wave_speed_m_per_s: float,
    max_cfl: float = 0.5,
) -> None:
    """Pre-flight CFL check against a user-supplied wave speed.

    Computes ``CFL = wave_speed * dt / dx_min`` and raises if it exceeds
    ``max_cfl``. The wave speed must be supplied explicitly — we do not
    try to infer it from the model since the appropriate speed depends
    on which equations the model solves (gravity wave for SWM, internal
    gravity wave for stratified, advection for QG, ...).

    For a multilayer SWM, the relevant speed is the external mode
    ``c = sqrt(g * H_total)``. For a single-layer SWM with mean depth
    ``H0``, it's ``sqrt(g * H0)``. For barotropic QG, ``c`` is the
    maximum velocity (advection CFL).

    Args:
        spec: The validated, debug-merged RunSpec.
        model: The constructed model instance (used only for ``model.grid``).
        wave_speed_m_per_s: Fastest wave speed in the model, m/s.
        max_cfl: Maximum acceptable CFL number. Defaults to ``0.5``,
            a conservative bound that works for most explicit RK schemes.

    Raises:
        AssertionFailedError: If ``wave_speed * dt / dx_min > max_cfl``.
    """
    if wave_speed_m_per_s <= 0:
        raise AssertionFailedError(
            f"cfl: wave_speed_m_per_s must be > 0 (got {wave_speed_m_per_s})"
        )
    if max_cfl <= 0:
        raise AssertionFailedError(f"cfl: max_cfl must be > 0 (got {max_cfl})")

    grid = getattr(model, "grid", None)
    if grid is None:
        raise AssertionFailedError(
            f"cfl: model {type(model).__name__!r} has no .grid attribute; "
            f"cannot infer dx"
        )
    dx_min = float(min(grid.dx, grid.dy))
    dt = float(spec.timestepping.dt)
    cfl = wave_speed_m_per_s * dt / dx_min
    if cfl > max_cfl:
        dt_safe = max_cfl * dx_min / wave_speed_m_per_s
        raise AssertionFailedError(
            f"cfl check FAILED: CFL = {cfl:.3f} > max_cfl = {max_cfl}\n"
            f"  wave_speed = {wave_speed_m_per_s:.2f} m/s\n"
            f"  dt         = {dt:.4f} s\n"
            f"  dx_min     = {dx_min:.2f} m\n"
            f"  → maximum stable dt at this CFL: {dt_safe:.4f} s"
        )


PREFLIGHT_ASSERTIONS: dict[str, Callable[..., None]] = {
    "cfl": check_cfl,
}


# ----------------------------------------------------------------------
# Postflight assertions: (spec, metrics) -> None
# ----------------------------------------------------------------------


def check_bounded_metric(
    spec: RunSpec,
    metrics: dict[str, Any],
    *,
    name: str,
    min: float | None = None,
    max: float | None = None,
) -> None:
    """Post-flight check that a scalar metric falls in ``[min, max]``.

    Args:
        spec: The RunSpec (unused but in the signature for symmetry).
        metrics: The flat metrics dict the runner is about to write.
        name: Key in ``metrics`` to inspect. Must reference a numeric scalar.
        min: Optional lower bound (inclusive). Skipped if ``None``.
        max: Optional upper bound (inclusive). Skipped if ``None``.

    Raises:
        AssertionFailedError: If the metric is missing, non-numeric,
            non-finite, or out of range.
    """
    if name not in metrics:
        raise AssertionFailedError(
            f"bounded_metric: metric {name!r} not present in run output. "
            f"Available metrics: {sorted(metrics)}"
        )
    raw = metrics[name]
    try:
        value = float(np.asarray(raw))
    except (TypeError, ValueError) as exc:
        raise AssertionFailedError(
            f"bounded_metric: metric {name!r} is not a numeric scalar (got {raw!r})"
        ) from exc
    if not np.isfinite(value):
        raise AssertionFailedError(
            f"bounded_metric: metric {name!r} is non-finite ({value})"
        )
    if min is not None and value < min:
        raise AssertionFailedError(
            f"bounded_metric: {name} = {value} is below min = {min}"
        )
    if max is not None and value > max:
        raise AssertionFailedError(
            f"bounded_metric: {name} = {value} is above max = {max}"
        )


POSTFLIGHT_ASSERTIONS: dict[str, Callable[..., None]] = {
    "bounded_metric": check_bounded_metric,
}


# ----------------------------------------------------------------------
# Runner — dispatch over the assertions block
# ----------------------------------------------------------------------


def run_preflight(spec: RunSpec, model: Any) -> None:
    """Run every preflight assertion declared in ``spec.assertions``.

    Unknown assertion names are an error (catches typos in configs).

    Args:
        spec: Validated, debug-merged RunSpec.
        model: Constructed model instance.

    Raises:
        AssertionFailedError: If any assertion fails OR an unknown name
            is referenced.
    """
    for name, params in (spec.assertions or {}).items():
        if name in POSTFLIGHT_ASSERTIONS:
            # Postflight names are skipped at preflight time.
            continue
        check = PREFLIGHT_ASSERTIONS.get(name)
        if check is None:
            raise AssertionFailedError(
                f"unknown assertion {name!r}; available preflight: "
                f"{sorted(PREFLIGHT_ASSERTIONS)}; available postflight: "
                f"{sorted(POSTFLIGHT_ASSERTIONS)}"
            )
        check(spec, model, **(params or {}))


def run_postflight(spec: RunSpec, metrics: dict[str, Any]) -> None:
    """Run every postflight assertion declared in ``spec.assertions``.

    Unknown assertion names are an error (catches typos in configs).

    Args:
        spec: Validated, debug-merged RunSpec.
        metrics: The flat metrics dict that will be written to disk.

    Raises:
        AssertionFailedError: If any assertion fails OR an unknown name
            is referenced.
    """
    for name, params in (spec.assertions or {}).items():
        if name in PREFLIGHT_ASSERTIONS:
            continue
        check = POSTFLIGHT_ASSERTIONS.get(name)
        if check is None:
            raise AssertionFailedError(
                f"unknown assertion {name!r}; available preflight: "
                f"{sorted(PREFLIGHT_ASSERTIONS)}; available postflight: "
                f"{sorted(POSTFLIGHT_ASSERTIONS)}"
            )
        check(spec, metrics, **(params or {}))
