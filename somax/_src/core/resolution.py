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
import numpy as np
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
    """Reject a resolution threshold that would disable its guard.

    These come straight from a config: ``munk_width: {n_cells_min:
    .nan}`` is valid YAML, ``yaml.safe_load`` produces a NaN, and
    ``run_preflight`` forwards it unchanged. Every comparison against
    NaN is false, so the guard would pass a layer resolved by no cells
    at all.

    A negative threshold has the same effect by a different route: a
    width ratio is never negative, so ``ratio < n_cells_min`` is
    always false and an explicitly configured assertion silently stops
    asserting anything. Zero is allowed — it is the warn-only setting.

    Args:
        check: Caller name, for the error message.
        **thresholds: Named threshold values.

    Raises:
        AssertionFailedError: If any threshold is not finite, or is
            negative.
    """
    for name, value in thresholds.items():
        number = float(value)
        if not math.isfinite(number):
            raise AssertionFailedError(
                f"{check}: {name} is {number}. A non-finite threshold "
                f"compares false against every ratio, so the check would "
                f"always pass."
            )
        if number < 0.0:
            raise AssertionFailedError(
                f"{check}: {name} is {number}. A width ratio is never "
                f"negative, so a negative threshold would disable the "
                f"check entirely; use 0 to warn only."
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


def check_deformation_radius(
    spec: RunSpec,
    model: Any,
    *,
    n_cells_min: float = 2.0,
    n_cells_warn: float = 4.0,
) -> None:
    """Pre-flight: the first baroclinic deformation radius is resolved.

    Resolving the first internal deformation radius ``L_d = sqrt(g'H)/f0``
    with at least ``n_cells_min`` grid cells is necessary for baroclinic
    instability and mesoscale eddies (Hallberg 2013). FAIL when
    ``L_d/dx < n_cells_min``; WARN when ``n_cells_min <= L_d/dx <
    n_cells_warn``.

    ``dx`` here is the coarser of the two spacings, since a deformation
    radius is isotropic and must be spanned in both directions.

    Requires a stratified model exposing ``model.strat.g_prime`` /
    ``model.strat.H`` and ``model.consts.f0`` (multilayer SWM, baroclinic /
    reparameterized QG). Raises for models without that structure so a typo'd
    config doesn't silently skip the check.

    Args:
        spec: The validated RunSpec (unused; present for signature symmetry).
        model: The constructed model instance.
        n_cells_min: Minimum ``L_d/dx`` below which to FAIL. Defaults to 2.0.
        n_cells_warn: ``L_d/dx`` below which to WARN. Defaults to 4.0.

    Raises:
        AssertionFailedError: If the model lacks stratification/Coriolis, or
            if ``L_d/dx < n_cells_min``.
    """
    strat = getattr(model, "strat", None)
    consts = getattr(model, "consts", None)
    grid = getattr(model, "grid", None)
    if strat is None or grid is None or consts is None:
        raise AssertionFailedError(
            f"deformation_radius: model {type(model).__name__!r} lacks "
            f"strat/consts/grid; this check applies to stratified models "
            f"(multilayer SWM, baroclinic/reparameterized QG)."
        )
    g_prime = getattr(strat, "g_prime", None)
    H = getattr(strat, "H", None)
    f0 = getattr(consts, "f0", None)
    if g_prime is None or H is None or f0 is None:
        raise AssertionFailedError(
            f"deformation_radius: model {type(model).__name__!r} does not "
            f"expose strat.g_prime / strat.H / consts.f0; cannot compute L_d."
        )
    f0_abs = abs(float(f0))
    if f0_abs == 0.0:
        raise AssertionFailedError(
            "deformation_radius: consts.f0 is zero; L_d is undefined on an "
            "f-plane with no rotation."
        )
    # Prefer the model's vertical-mode deformation radii when available: the
    # first internal radius comes from the vertical-mode eigenproblem, not
    # just one interface's sqrt(g'_k H_k)/f0 (which can substantially
    # overestimate the baroclinic radius for a thin upper layer). Fall back to
    # the per-interface estimate only when the modal transform is absent.
    modal = getattr(model, "modal", None)
    modal_radii = getattr(modal, "rossby_radii", None) if modal is not None else None
    if modal_radii is not None:
        radii = np.asarray(jnp.asarray(modal_radii))
        # The barotropic mode is infinite; keep only the finite internal modes.
        finite = radii[np.isfinite(radii)]
        if finite.size == 0:
            raise AssertionFailedError(
                "deformation_radius: model exposes no finite internal "
                "deformation radius (modal.rossby_radii are all non-finite)."
            )
        Ld = float(np.min(finite))
        source = "modal.rossby_radii"
    else:
        # Per-interface estimate sqrt(g'_k H_k)/f0; smallest internal mode.
        g_prime_arr = np.asarray(jnp.asarray(g_prime))
        H_arr = np.asarray(jnp.asarray(H))
        radii = np.sqrt(g_prime_arr * H_arr) / f0_abs
        internal = radii[1:] if radii.shape[0] > 1 else radii
        Ld = float(np.min(internal))
        source = "sqrt(g'H)/f0 estimate"
    # The *coarser* spacing: a deformation radius is an isotropic
    # length, so it has to be resolved in both directions, and taking
    # the finer one would pass an anisotropic grid that resolves it
    # along only one axis. (The Munk and Stommel guards take dx
    # instead — those layers are normal to the western wall.)
    dx_min = float(max(grid.dx, grid.dy))
    ratio = Ld / dx_min
    if ratio < n_cells_min:
        raise AssertionFailedError(
            f"deformation_radius check FAILED: L_d/dx = {ratio:.2f} < "
            f"{n_cells_min}\n"
            f"  L_d   = {Ld:.4g} (smallest internal deformation radius, "
            f"from {source})\n"
            f"  dx    = {dx_min:.4g}\n"
            f"  → eddies will be suppressed; refine the grid or pick an "
            f"eddy-permitting configuration."
        )
    if ratio < n_cells_warn:
        _warn(
            "deformation radius marginally resolved: L_d/dx = {:.2f} "
            "(L_d={:.4g}, dx={:.4g})",
            ratio,
            Ld,
            dx_min,
        )
