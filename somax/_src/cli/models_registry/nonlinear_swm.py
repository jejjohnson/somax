"""Nonlinear shallow water model entry.

Phase 3 (#77) implements the ``double_gyre`` x ``nonlinear_swm``
adapter — the port of the legacy ``barotropic_jet_instability`` test
case. Uses :class:`somax.models.NonlinearShallowWater2D`.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from somax._src.cli.scenarios import ScenarioBundle

from ._types import BuiltModel, ModelEntry, SupportFlags


def _build(scenario: ScenarioBundle, params: dict[str, Any]) -> BuiltModel:
    from somax.models import NonlinearShallowWater2D, NonlinearSW2DState

    geometry = scenario.geometry
    if geometry.Lx is None or geometry.Ly is None:
        raise ValueError(
            "nonlinear_swm requires a Cartesian scenario geometry with Lx/Ly."
        )
    consts = scenario.constants
    forcing = scenario.forcing_params
    model_params = dict(params.get("params", {}))
    H0 = float(model_params.get("H0", 1000.0))

    model = NonlinearShallowWater2D.create(
        nx=geometry.nx,
        ny=geometry.ny,
        Lx=geometry.Lx,
        Ly=geometry.Ly,
        g=consts.g,
        f0=consts.f0,
        beta=consts.beta,
        H0=H0,
        lateral_viscosity=float(model_params.get("lateral_viscosity", 0.0)),
        bottom_drag=float(model_params.get("bottom_drag", 0.0)),
        wind_amplitude=float(forcing.get("wind_amplitude", 0.0)),
        wind_profile=str(forcing.get("wind_profile", "doublegyre")),
        bc=str(model_params.get("bc", "periodic")),
        method=str(model_params.get("method", "upwind1")),
    )

    state0 = _initial_state(scenario, model, NonlinearSW2DState, H0=H0)
    return BuiltModel(model=model, state0=state0)


def _initial_state(
    scenario: ScenarioBundle, model: Any, state_cls: type, *, H0: float
) -> Any:
    """Build a :class:`NonlinearSW2DState` from ``scenario.initial_condition``."""
    grid = model.grid
    ny, nx = grid.Ny, grid.Nx
    ic = scenario.initial_condition

    if ic.type == "at_rest":
        return state_cls(
            h=jnp.full((ny, nx), H0),
            u=jnp.zeros((ny, nx)),
            v=jnp.zeros((ny, nx)),
        )
    if ic.type == "jet":
        ic_params = ic.params
        jet_speed = float(ic_params.get("jet_speed", 1.0))
        jet_width = float(ic_params.get("jet_width", 5.0e4))
        perturbation = float(ic_params.get("perturbation", 0.01))

        Lx = float(scenario.geometry.Lx)
        Ly = float(scenario.geometry.Ly)
        x = jnp.arange(nx) * grid.dx
        y = jnp.arange(ny) * grid.dy
        X, Y = jnp.meshgrid(x, y)
        y0 = Ly / 2.0
        u0 = jet_speed * jnp.exp(-0.5 * ((Y - y0) / jet_width) ** 2)
        v0 = (
            perturbation
            * jnp.sin(4.0 * jnp.pi * X / Lx)
            * jnp.exp(-0.5 * ((Y - y0) / jet_width) ** 2)
        )
        return state_cls(h=jnp.full_like(u0, H0), u=u0, v=v0)
    if ic.type == "gaussian_eddy":
        ic_params = ic.params
        amplitude = float(ic_params.get("amplitude", 1.0))
        sigma = float(ic_params.get("sigma", 5.0e4))
        Lx = float(scenario.geometry.Lx)
        Ly = float(scenario.geometry.Ly)
        x0 = float(ic_params.get("x0", Lx / 2.0))
        y0 = float(ic_params.get("y0", Ly / 2.0))
        x = jnp.arange(nx) * grid.dx
        y = jnp.arange(ny) * grid.dy
        X, Y = jnp.meshgrid(x, y)
        bump = amplitude * jnp.exp(
            -0.5 * (((X - x0) / sigma) ** 2 + ((Y - y0) / sigma) ** 2)
        )
        return state_cls(
            h=jnp.full_like(bump, H0) + bump,
            u=jnp.zeros_like(bump),
            v=jnp.zeros_like(bump),
        )

    raise NotImplementedError(
        f"nonlinear_swm: initial_condition.type={ic.type!r} not supported "
        "(expected 'at_rest', 'jet', or 'gaussian_eddy')."
    )


NONLINEAR_SWM = ModelEntry(
    name="nonlinear_swm",
    family="swm",
    layers=1,
    coordinates="cartesian",
    supports=SupportFlags(masks=True, spherical=False, forcing=("tau_x", "tau_y")),
    build=_build,
)
