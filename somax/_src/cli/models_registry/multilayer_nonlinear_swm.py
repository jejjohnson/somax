"""Multilayer nonlinear shallow water model entry.

Phase 3 (#77) implements the ``double_gyre`` x ``multilayer_nonlinear_swm``
adapter — the port of the legacy ``baroclinic_instability_swm`` test
case. Uses :class:`somax.models.MultilayerShallowWater2D`.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from somax._src.cli.scenarios import ScenarioBundle

from ._types import BuiltModel, ModelEntry, SupportFlags


def _build(scenario: ScenarioBundle, params: dict[str, Any]) -> BuiltModel:
    from somax.models import MultilayerShallowWater2D, MultilayerSW2DState

    geometry = scenario.geometry
    if geometry.Lx is None or geometry.Ly is None:
        raise ValueError(
            "multilayer_nonlinear_swm requires a Cartesian scenario "
            "geometry with Lx/Ly."
        )
    consts = scenario.constants
    forcing = scenario.forcing_params
    stratification = dict(params.get("stratification", {}))
    model_params = dict(params.get("params", {}))

    H = tuple(float(v) for v in stratification["H"])
    g_prime = tuple(float(v) for v in stratification["g_prime"])
    if len(H) != len(g_prime):
        raise ValueError(
            f"multilayer_nonlinear_swm: len(H)={len(H)} but "
            f"len(g_prime)={len(g_prime)}; both must match n_layers."
        )

    model = MultilayerShallowWater2D.create(
        nx=geometry.nx,
        ny=geometry.ny,
        Lx=geometry.Lx,
        Ly=geometry.Ly,
        g=consts.g,
        f0=consts.f0,
        beta=consts.beta,
        n_layers=len(H),
        H=H,
        g_prime=g_prime,
        lateral_viscosity=float(model_params.get("lateral_viscosity", 0.0)),
        bottom_drag=float(model_params.get("bottom_drag", 0.0)),
        wind_amplitude=float(forcing.get("wind_amplitude", 0.0)),
        wind_profile=str(forcing.get("wind_profile", "doublegyre")),
        bc=str(model_params.get("bc", "periodic")),
        method=str(model_params.get("method", "upwind1")),
    )

    state0 = _initial_state(scenario, model, MultilayerSW2DState)
    return BuiltModel(model=model, state0=state0)


def _initial_state(scenario: ScenarioBundle, model: Any, state_cls: type) -> Any:
    """Build the initial state for the multilayer SWM."""
    grid = model.grid
    ny, nx = grid.Ny, grid.Nx
    nl = model.strat.H.shape[0]
    ic = scenario.initial_condition

    H_layers = model.strat.H[:, None, None]
    h_rest = jnp.broadcast_to(H_layers, (nl, ny, nx))

    if ic.type == "at_rest":
        return state_cls(
            h=h_rest,
            u=jnp.zeros((nl, ny, nx)),
            v=jnp.zeros((nl, ny, nx)),
        )
    if ic.type == "jet":
        ic_params = ic.params
        jet_speed = float(ic_params.get("jet_speed", 0.5))
        jet_width = float(ic_params.get("jet_width", 5.0e4))
        perturbation = float(ic_params.get("perturbation", 0.01))

        Lx = float(scenario.geometry.Lx)
        Ly = float(scenario.geometry.Ly)
        x = jnp.arange(nx) * grid.dx
        y = jnp.arange(ny) * grid.dy
        X, Y = jnp.meshgrid(x, y)
        y0 = Ly / 2.0
        jet_profile = jnp.exp(-0.5 * ((Y - y0) / jet_width) ** 2)

        # Opposite jets per layer → baroclinic shear. Works for any
        # n_layers; sign alternates so the vertical integral is zero for
        # even layer counts and small for odd.
        signs = jnp.asarray([1.0 if (i % 2 == 0) else -1.0 for i in range(nl)])
        u0 = signs[:, None, None] * jet_speed * jet_profile[None]
        v0 = jnp.broadcast_to(
            perturbation * jnp.sin(4.0 * jnp.pi * X / Lx)[None] * jet_profile[None],
            (nl, ny, nx),
        )
        return state_cls(h=h_rest, u=u0, v=v0)

    raise NotImplementedError(
        f"multilayer_nonlinear_swm: initial_condition.type={ic.type!r} not "
        "supported (expected 'at_rest' or 'jet')."
    )


MULTILAYER_NONLINEAR_SWM = ModelEntry(
    name="multilayer_nonlinear_swm",
    family="swm",
    layers="multi",
    coordinates="cartesian",
    supports=SupportFlags(masks=True, spherical=False, forcing=("tau_x", "tau_y")),
    build=_build,
)
