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

import jax.numpy as jnp
import numpy as np
import paramax
from loguru import logger


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
    dx_min = float(min(grid.dx, grid.dy))
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
        logger.warning(
            "deformation radius marginally resolved: L_d/dx = {:.2f} "
            "(L_d={:.4g}, dx={:.4g})",
            ratio,
            Ld,
            dx_min,
        )


def check_equatorial_deformation_radius(
    spec: RunSpec | None,
    model: Any,
    *,
    n_cells_min: float = 2.0,
    n_cells_warn: float = 4.0,
) -> None:
    """Pre-flight: the equatorial deformation radius is resolved.

    On a sphere the Coriolis parameter vanishes at the equator, so the
    mid-latitude radius ``sqrt(gH)/f0`` diverges there and says nothing
    useful. The finite scale that replaces it is the *equatorial*
    deformation radius

    ``L_eq = sqrt(c / beta_eq)``,  ``c = sqrt(g H)``,
    ``beta_eq = 2 Omega / a``,

    which is the trapping width of the equatorial waveguide — Kelvin,
    Yanai and equatorial Rossby waves all live inside it. A grid that
    cannot span it does not have an equatorial waveguide at all.

    In terms of the Burger number ``Bu = gH/(2 Omega a)**2`` this is
    just ``L_eq/a = Bu**(1/4)``, so the check is scale-free and reads
    the same on a nondimensional sphere as on Earth.

    The comparison uses the *widest* interior cell rather than the
    narrowest. Zonal cells are widest at the equator, which is exactly
    where the waveguide sits, so the coarsest cell is the one that has
    to resolve it.

    Requires a spherical shallow-water-shaped model exposing
    ``consts.gravity``, ``consts.H0``, ``consts.omega``,
    ``consts.radius`` and a spherical ``grid``. Raises for models
    without that structure so a typo'd config doesn't silently skip the
    check. Barotropic spherical QG is rigid-lid and has no gravity
    wave, so it is rejected rather than checked.

    Args:
        spec: The validated RunSpec (unused; present for registry
            signature symmetry, and ``None`` when called from a
            ``from_nondimensional`` factory).
        model: The constructed model instance.
        n_cells_min: ``L_eq/dx`` below which to FAIL.
        n_cells_warn: ``L_eq/dx`` below which to WARN.

    Raises:
        AssertionFailedError: If the model is not a spherical
            shallow-water model, if a required constant is
            non-positive, or if the waveguide is unresolved.
    """
    del spec
    consts = getattr(model, "consts", None)
    grid = getattr(model, "grid", None)
    if consts is None or grid is None:
        raise AssertionFailedError(
            f"equatorial_deformation_radius: model {type(model).__name__!r} "
            f"lacks consts/grid; this check applies to spherical "
            f"shallow-water models."
        )
    missing = [
        name
        for name in ("gravity", "H0", "omega", "radius")
        if getattr(consts, name, None) is None
    ]
    if missing or getattr(grid, "cos_lat_T", None) is None:
        raise AssertionFailedError(
            f"equatorial_deformation_radius: model {type(model).__name__!r} "
            f"does not expose consts.gravity / H0 / omega / radius on a "
            f"spherical grid (missing {missing or ['grid.cos_lat_T']}); "
            f"cannot compute L_eq. Barotropic QG is rigid-lid and has no "
            f"gravity-wave deformation radius — drop this check for it."
        )
    g = float(consts.gravity)
    depth = float(consts.H0)
    omega = abs(float(consts.omega))
    radius = float(consts.radius)
    for name, value in (("gravity", g), ("H0", depth), ("omega", omega)):
        if value <= 0.0:
            raise AssertionFailedError(
                f"equatorial_deformation_radius: consts.{name} is "
                f"{value:.4g}; L_eq is undefined without a gravity wave and "
                f"a rotating planet."
            )
    wave_speed = np.sqrt(g * depth)
    beta_eq = 2.0 * omega / radius
    Leq = float(np.sqrt(wave_speed / beta_eq))

    dx_max = _max_spherical_cell_width(grid)
    ratio = Leq / dx_max
    if ratio < n_cells_min:
        raise AssertionFailedError(
            f"equatorial_deformation_radius check FAILED: L_eq/dx = "
            f"{ratio:.2f} < {n_cells_min}\n"
            f"  L_eq = {Leq:.4g} (= sqrt(c/beta_eq), c={wave_speed:.4g}, "
            f"beta_eq={beta_eq:.4g})\n"
            f"  dx   = {dx_max:.4g} (widest interior cell)\n"
            f"  → the equatorial waveguide is unresolved; refine the grid "
            f"or raise the equivalent depth."
        )
    if ratio < n_cells_warn:
        logger.warning(
            "equatorial deformation radius marginally resolved: "
            "L_eq/dx = {:.2f} (L_eq={:.4g}, dx={:.4g})",
            ratio,
            Leq,
            dx_max,
        )


def _max_spherical_cell_width(grid: Any) -> float:
    """Widest interior cell of a spherical grid, in metres.

    Computed here rather than read off ``grid.max_cell_width``: that
    helper arrives with finitevolX#244 and somax still pins v0.0.41.
    The cosine is clamped at zero because in float32 ``cos(pi/2)`` is a
    small *negative* number, which would otherwise flip the sign of a
    polar cell width.
    """
    cos_lat = np.asarray(jnp.asarray(grid.cos_lat_T))[1:-1, 1:-1]
    dx = float(np.max(np.maximum(cos_lat, 0.0))) * float(grid.R) * float(grid.dlon)
    dy = float(grid.R) * float(grid.dlat)
    return max(dx, dy)


def check_pv_inversion(
    spec: RunSpec,
    model: Any,
    *,
    tol: float = 1e-6,
) -> None:
    """Pre-flight: the barotropic-QG PV-inversion round-trip closes to eps.

    Inverts the model's initial PV to a streamfunction and re-derives PV via
    the Laplacian, asserting a small relative residual. Catches a broken
    elliptic solver, wrong boundary stencil, or mis-staggered field *before*
    burning compute on the integration.

    Scoped to **barotropic** QG, where the PV is exactly the relative
    vorticity ``q = nabla^2 psi`` (a 2-D PV field). Baroclinic /
    reparameterized QG invert a modal Helmholtz operator
    ``q = nabla^2 psi - f0^2 A psi``; re-deriving with the bare Laplacian would
    drop the stretching term and the residual would be O(1) for a perfectly
    balanced state, so those models are rejected rather than checked with the
    wrong operator. Raises for non-QG models too, so a typo doesn't silently
    skip the check.

    Args:
        spec: The validated RunSpec (used only to rebuild the initial state).
        model: The constructed barotropic QG model instance.
        tol: Maximum allowed relative residual ``||L(psi) - q|| / ||q||``.

    Raises:
        AssertionFailedError: If the model is not a barotropic QG model, or if
            the round-trip residual exceeds ``tol``.
    """
    invert = getattr(model, "_invert_pv", None)
    diff = getattr(model, "diff", None)
    if invert is None or diff is None or not hasattr(diff, "laplacian"):
        raise AssertionFailedError(
            f"pv_inversion: model {type(model).__name__!r} has no _invert_pv / "
            f"diff.laplacian; this check applies to barotropic QG."
        )
    # Build the factory initial state for this scenario x model pair.
    from somax._src.cli._factories import build
    from somax._src.cli._run import _model_params, _scenario_params

    _model, state0 = build(
        spec.scenario.name,
        spec.model.name,
        scenario_params=_scenario_params(spec),
        model_params=_model_params(spec),
    )
    q = state0.q
    if q.ndim != 2:
        raise AssertionFailedError(
            f"pv_inversion: model {type(model).__name__!r} has a "
            f"{q.ndim}-D PV field; this check is scoped to barotropic QG "
            f"(2-D PV, q = laplacian(psi)). Baroclinic / reparameterized QG "
            f"invert a modal Helmholtz operator with a stretching term that "
            f"the bare Laplacian round-trip cannot reproduce."
        )
    psi = invert(q)
    q_hat = diff.laplacian(psi)
    # Compare on the interior (drop the one-cell ghost halo the BC owns).
    interior = (slice(1, -1), slice(1, -1))
    num = float(jnp.linalg.norm((q_hat - q)[interior]))
    den = float(jnp.linalg.norm(q[interior]))
    if den < 1e-30:
        # A zero initial PV trivially round-trips; nothing to assert.
        return
    residual = num / den
    if residual > tol:
        raise AssertionFailedError(
            f"pv_inversion check FAILED: relative residual {residual:.2e} > "
            f"tol {tol:.0e}\n"
            f"  ||laplacian(psi) - q|| / ||q|| over the interior.\n"
            f"  A clean elliptic solver should close this to ~machine eps; a "
            f"large residual points at a broken solver, wrong BC stencil, or "
            f"mis-staggered field."
        )


def check_static_stability(spec: RunSpec, model: Any) -> None:
    """Pre-flight: the stratification is statically stable (g' > 0).

    A layered model is statically stable when every interface reduced gravity
    is positive (``g'_k > 0`` ⇔ ``N^2 > 0`` ⇔ density increasing with depth).
    A non-positive interface reduced gravity is a convectively unstable /
    mis-ordered density profile.

    Applies to stratified models exposing ``model.strat.g_prime``; raises for
    models without it.

    Args:
        spec: The validated RunSpec (unused; signature symmetry).
        model: The constructed model instance.

    Raises:
        AssertionFailedError: If the model lacks stratification, or any
            internal interface reduced gravity is non-positive.
    """
    strat = getattr(model, "strat", None)
    g_prime = getattr(strat, "g_prime", None) if strat is not None else None
    if g_prime is None:
        raise AssertionFailedError(
            f"static_stability: model {type(model).__name__!r} has no "
            f"strat.g_prime; this check applies to stratified layered models."
        )
    g_prime_arr = np.asarray(jnp.asarray(g_prime))
    # g_prime[0] is the surface (full gravity); internal interfaces are [1:].
    internal = g_prime_arr[1:] if g_prime_arr.shape[0] > 1 else g_prime_arr
    if np.any(internal <= 0.0):
        bad = np.where(internal <= 0.0)[0] + 1
        raise AssertionFailedError(
            f"static_stability check FAILED: non-positive reduced gravity at "
            f"interface(s) {bad.tolist()} (g' = {internal.tolist()}).\n"
            f"  N^2 <= 0 implies a convectively unstable / mis-ordered density "
            f"profile. Order layers light-to-dense (top-to-bottom)."
        )


def _boundary_layer_inputs(model: Any, check: str) -> tuple[Any, float, float]:
    """Shared lookup for the western-boundary-layer guards.

    Returns ``(params, beta, dx_min)`` with constrained parameters
    already reconstituted, so a model carrying ``somax.positive``
    wrappers is checked on its actual coefficient values.
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
    beta = abs(float(beta))
    if beta == 0.0:
        raise AssertionFailedError(
            f"{check}: consts.beta is zero. There is no western boundary "
            f"layer on an f-plane, so this check does not apply."
        )
    return params, beta, float(min(grid.dx, grid.dy))


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
    params, beta, dx_min = _boundary_layer_inputs(model, "munk_width")
    nu = getattr(params, "lateral_viscosity", None)
    if nu is None:
        raise AssertionFailedError(
            "munk_width: model does not expose params.lateral_viscosity; "
            "cannot compute the Munk width."
        )
    nu = abs(float(jnp.asarray(nu)))
    if nu == 0.0:
        raise AssertionFailedError(
            "munk_width: lateral_viscosity is zero, so there is no Munk "
            "layer. Drop this check for an inviscid or Stommel-only run."
        )
    delta_m = (nu / beta) ** (1.0 / 3.0)
    ratio = delta_m / dx_min
    if ratio < n_cells_min:
        raise AssertionFailedError(
            f"munk_width check FAILED: delta_M/dx = {ratio:.2f} < "
            f"{n_cells_min}\n"
            f"  delta_M = {delta_m:.4g} (= (nu/beta)^(1/3), nu={nu:.4g}, "
            f"beta={beta:.4g})\n"
            f"  dx      = {dx_min:.4g}\n"
            f"  → the western boundary current is unresolved; refine the "
            f"grid or raise the viscosity "
            f"(nu >= {beta * (n_cells_min * dx_min) ** 3:.4g})."
        )
    if ratio < n_cells_warn:
        logger.warning(
            "Munk layer marginally resolved: delta_M/dx = {:.2f} "
            "(delta_M={:.4g}, dx={:.4g})",
            ratio,
            delta_m,
            dx_min,
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
    params, beta, dx_min = _boundary_layer_inputs(model, "stommel_width")
    kappa = getattr(params, "bottom_drag", None)
    if kappa is None:
        raise AssertionFailedError(
            "stommel_width: model does not expose params.bottom_drag; "
            "cannot compute the Stommel width."
        )
    kappa = abs(float(jnp.asarray(kappa)))
    if kappa == 0.0:
        raise AssertionFailedError(
            "stommel_width: bottom_drag is zero, so there is no Stommel "
            "layer. Drop this check for a Munk-only run."
        )
    delta_s = kappa / beta
    ratio = delta_s / dx_min
    if ratio < n_cells_min:
        raise AssertionFailedError(
            f"stommel_width check FAILED: delta_S/dx = {ratio:.2f} < "
            f"{n_cells_min}\n"
            f"  delta_S = {delta_s:.4g} (= kappa/beta, kappa={kappa:.4g}, "
            f"beta={beta:.4g})\n"
            f"  dx      = {dx_min:.4g}\n"
            f"  → the frictional boundary layer is unresolved; refine the "
            f"grid or raise the drag "
            f"(kappa >= {beta * n_cells_min * dx_min:.4g})."
        )
    if ratio < n_cells_warn:
        logger.warning(
            "Stommel layer marginally resolved: delta_S/dx = {:.2f} "
            "(delta_S={:.4g}, dx={:.4g})",
            ratio,
            delta_s,
            dx_min,
        )


PREFLIGHT_ASSERTIONS: dict[str, Callable[..., None]] = {
    "cfl": check_cfl,
    "deformation_radius": check_deformation_radius,
    "equatorial_deformation_radius": check_equatorial_deformation_radius,
    "munk_width": check_munk_width,
    "pv_inversion": check_pv_inversion,
    "stommel_width": check_stommel_width,
    "static_stability": check_static_stability,
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
