"""Resolution guards shared by the model factories and the CLI.

These live outside the CLI package on purpose. A
``from_nondimensional`` factory runs them by default, and the factories
are ordinary library API — a base ``pip install somax`` must be able to
call them without pulling in the optional ``sim`` dependency group that
the command-line interface needs.

They take ``(spec, model)`` so the CLI can register them directly as
preflight assertions; ``spec`` is unused and may be ``None``.
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import paramax


if TYPE_CHECKING:
    from somax._src.cli.spec import RunSpec

try:  # pragma: no cover - exercised by whichever install is in use
    from loguru import logger as _logger
except ModuleNotFoundError:  # pragma: no cover
    _logger = None


class AssertionFailedError(RuntimeError):
    """Raised when an opt-in preflight or postflight assertion fails."""


def _warn(template: str, *args: Any) -> None:
    """Emit a non-fatal guard warning.

    Routed through loguru when it is installed, so a CLI run keeps its
    single log stream; through :mod:`warnings` otherwise, so a base
    install still hears about a marginally resolved boundary layer.
    """
    if _logger is not None:
        _logger.warning(template, *args)
    else:
        warnings.warn(template.format(*args), UserWarning, stacklevel=3)


def _require_finite_positive(check: str, name: str, value: float) -> float:
    """Return ``value``, or fail the check if it is not usable.

    Taking ``abs`` would invent a boundary layer for a model that has
    none: a negative viscosity is anti-diffusive and a negative drag is
    anti-damping, and reporting either as "resolved" is worse than
    saying the check does not apply. Plain parameters can be passed
    straight to ``create``, so this is reachable.
    """
    value = float(value)
    if not math.isfinite(value):
        raise AssertionFailedError(
            f"{check}: {name} is {value}; the boundary-layer width is "
            f"undefined for a non-finite coefficient."
        )
    if value < 0.0:
        raise AssertionFailedError(
            f"{check}: {name} is {value:.4g}. A negative coefficient is "
            f"anti-diffusive rather than dissipative, so there is no "
            f"boundary layer to resolve — fix the model rather than the grid."
        )
    return value


def _require_finite_thresholds(check: str, **thresholds: float) -> None:
    """Reject a non-finite resolution threshold.

    These come straight from a config: ``munk_width: {n_cells_min:
    .nan}`` is valid YAML, ``yaml.safe_load`` produces a NaN, and
    ``run_preflight`` forwards it unchanged. Every comparison against
    NaN is false, so the guard would pass a layer resolved by no cells
    at all.

    Args:
        check: Caller name, for the error message.
        **thresholds: Named threshold values.

    Raises:
        AssertionFailedError: If any threshold is not finite.
    """
    for name, value in thresholds.items():
        if not math.isfinite(float(value)):
            raise AssertionFailedError(
                f"{check}: {name} is {float(value)}. A non-finite threshold "
                f"compares false against every ratio, so the check would "
                f"always pass."
            )


def _boundary_layer_inputs(model: Any, check: str) -> tuple[Any, float, float]:
    """Shared lookup for the western-boundary-layer guards.

    Returns ``(params, beta, dx)`` with constrained parameters already
    reconstituted, so a model carrying ``somax.positive`` wrappers is
    checked on its actual coefficient values.
    """
    model = paramax.unwrap(model)
    consts = getattr(model, "consts", None)
    grid = getattr(model, "grid", None)
    params = getattr(model, "params", None)
    if consts is None or grid is None or params is None:
        raise AssertionFailedError(
            f"{check}: model {type(model).__name__!r} lacks params/consts/grid; "
            f"this check applies to beta-plane QG models."
        )
    beta = getattr(consts, "beta", None)
    if beta is None:
        raise AssertionFailedError(
            f"{check}: model {type(model).__name__!r} does not expose "
            f"consts.beta; the boundary-layer width is undefined without it."
        )
    # ``abs`` is right for beta, unlike for the dissipative
    # coefficients: its sign is a hemisphere convention, and the layer
    # width depends only on its magnitude. Finiteness still has to be
    # tested separately — ``abs(nan)`` is NaN, the zero test below does
    # not catch it, and both width ratios then compare false against
    # their thresholds, so every guard would silently pass.
    if not math.isfinite(float(beta)):
        raise AssertionFailedError(
            f"{check}: consts.beta is {float(beta)}; the boundary-layer "
            f"width is undefined for a non-finite planetary gradient."
        )
    beta = abs(float(beta))
    if beta == 0.0:
        raise AssertionFailedError(
            f"{check}: consts.beta is zero. There is no western boundary "
            f"layer on an f-plane, so this check does not apply."
        )
    # The zonal spacing, not the finer of the two. Both layers are
    # normal to the western and eastern walls, so it is ``dx`` that has
    # to span them; on a grid with ``dy < dx`` — ``ny`` much larger
    # than ``nx`` — taking the minimum inflates both ratios and can
    # pass a western boundary current that is unresolved.
    return params, beta, float(grid.dx)


def check_munk_width(
    spec: RunSpec | None,
    model: Any,
    *,
    n_cells_min: float = 2.0,
    n_cells_warn: float = 4.0,
) -> None:
    """Pre-flight: the Munk (viscous) western boundary layer is resolved.

    The Munk layer has width ``delta_M = (nu / beta)**(1/3)``. It is the
    thinnest feature a viscous wind-driven gyre contains, so if the grid
    cannot span it the western boundary current is wrong no matter what
    the interior looks like. FAIL when ``delta_M/dx < n_cells_min``;
    WARN below ``n_cells_warn``.

    Works on dimensional and nondimensional models alike, since it
    compares two lengths taken from the same model.

    Args:
        spec: The validated RunSpec (unused; present for registry
            signature symmetry, and ``None`` when called from a
            ``from_nondimensional`` factory).
        model: The constructed model instance.
        n_cells_min: ``delta_M/dx`` below which to FAIL.
        n_cells_warn: ``delta_M/dx`` below which to WARN.

    Raises:
        AssertionFailedError: If the model lacks beta / viscosity, if
            viscosity is zero, or if the layer is unresolved.
    """
    del spec
    _require_finite_thresholds(
        "munk_width", n_cells_min=n_cells_min, n_cells_warn=n_cells_warn
    )
    params, beta, dx = _boundary_layer_inputs(model, "munk_width")
    nu = getattr(params, "lateral_viscosity", None)
    if nu is None:
        raise AssertionFailedError(
            "munk_width: model does not expose params.lateral_viscosity; "
            "cannot compute the Munk width."
        )
    nu = _require_finite_positive("munk_width", "lateral_viscosity", jnp.asarray(nu))
    if nu == 0.0:
        raise AssertionFailedError(
            "munk_width: lateral_viscosity is zero, so there is no Munk "
            "layer. Drop this check for an inviscid or Stommel-only run."
        )
    delta_m = (nu / beta) ** (1.0 / 3.0)
    ratio = delta_m / dx
    if ratio < n_cells_min:
        raise AssertionFailedError(
            f"munk_width check FAILED: delta_M/dx = {ratio:.2f} < "
            f"{n_cells_min}\n"
            f"  delta_M = {delta_m:.4g} (= (nu/beta)^(1/3), nu={nu:.4g}, "
            f"beta={beta:.4g})\n"
            f"  dx      = {dx:.4g}\n"
            f"  → the western boundary current is unresolved; refine the "
            f"grid or raise the viscosity "
            f"(nu >= {beta * (n_cells_min * dx) ** 3:.4g})."
        )
    if ratio < n_cells_warn:
        _warn(
            "Munk layer marginally resolved: delta_M/dx = {:.2f} "
            "(delta_M={:.4g}, dx={:.4g})",
            ratio,
            delta_m,
            dx,
        )


def check_stommel_width(
    spec: RunSpec | None,
    model: Any,
    *,
    n_cells_min: float = 1.0,
    n_cells_warn: float = 2.0,
) -> None:
    """Pre-flight: the Stommel (frictional) western boundary layer is resolved.

    The Stommel layer has width ``delta_S = kappa / beta``. The
    threshold is one cell rather than two: a drag-dominated boundary
    layer is a smoother structure than a viscous one, so a coarser
    representation is still meaningful.

    Args:
        spec: The validated RunSpec (unused; see :func:`check_munk_width`).
        model: The constructed model instance.
        n_cells_min: ``delta_S/dx`` below which to FAIL.
        n_cells_warn: ``delta_S/dx`` below which to WARN.

    Raises:
        AssertionFailedError: If the model lacks beta / drag, if drag is
            zero, or if the layer is unresolved.
    """
    del spec
    _require_finite_thresholds(
        "stommel_width", n_cells_min=n_cells_min, n_cells_warn=n_cells_warn
    )
    params, beta, dx = _boundary_layer_inputs(model, "stommel_width")
    kappa = getattr(params, "bottom_drag", None)
    if kappa is None:
        raise AssertionFailedError(
            "stommel_width: model does not expose params.bottom_drag; "
            "cannot compute the Stommel width."
        )
    kappa = _require_finite_positive("stommel_width", "bottom_drag", jnp.asarray(kappa))
    if kappa == 0.0:
        raise AssertionFailedError(
            "stommel_width: bottom_drag is zero, so there is no Stommel "
            "layer. Drop this check for a Munk-only run."
        )
    delta_s = kappa / beta
    ratio = delta_s / dx
    if ratio < n_cells_min:
        raise AssertionFailedError(
            f"stommel_width check FAILED: delta_S/dx = {ratio:.2f} < "
            f"{n_cells_min}\n"
            f"  delta_S = {delta_s:.4g} (= kappa/beta, kappa={kappa:.4g}, "
            f"beta={beta:.4g})\n"
            f"  dx      = {dx:.4g}\n"
            f"  → the frictional boundary layer is unresolved; refine the "
            f"grid or raise the drag "
            f"(kappa >= {beta * n_cells_min * dx:.4g})."
        )
    if ratio < n_cells_warn:
        _warn(
            "Stommel layer marginally resolved: delta_S/dx = {:.2f} "
            "(delta_S={:.4g}, dx={:.4g})",
            ratio,
            delta_s,
            dx,
        )
