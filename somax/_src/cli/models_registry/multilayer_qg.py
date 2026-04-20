"""Multilayer baroclinic quasi-geostrophic model entry.

Phase 3 (#77) implements the ``double_gyre`` x ``multilayer_qg``
adapter — the port of the legacy ``doublegyre_baroclinic_qg`` test
case. Uses :class:`somax.models.BaroclinicQG`.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from somax._src.cli.scenarios import ScenarioBundle

from ._types import BuiltModel, ModelEntry, SupportFlags


def _build(scenario: ScenarioBundle, params: dict[str, Any]) -> BuiltModel:
    from somax.models import BaroclinicQG, BaroclinicQGState

    geometry = scenario.geometry
    if geometry.Lx is None or geometry.Ly is None:
        raise ValueError(
            "multilayer_qg requires a Cartesian scenario geometry with Lx/Ly."
        )
    consts = scenario.constants
    forcing = scenario.forcing_params
    stratification = dict(params.get("stratification", {}))
    model_params = dict(params.get("params", {}))

    H = tuple(float(v) for v in stratification["H"])
    g_prime = tuple(float(v) for v in stratification["g_prime"])
    if len(H) != len(g_prime):
        raise ValueError(
            f"multilayer_qg: len(H)={len(H)} but len(g_prime)={len(g_prime)}; "
            "both must match n_layers."
        )

    model = BaroclinicQG.create(
        nx=geometry.nx,
        ny=geometry.ny,
        Lx=geometry.Lx,
        Ly=geometry.Ly,
        f0=consts.f0,
        beta=consts.beta,
        n_layers=len(H),
        H=H,
        g_prime=g_prime,
        lateral_viscosity=float(model_params.get("lateral_viscosity", 0.0)),
        bottom_drag=float(model_params.get("bottom_drag", 0.0)),
        wind_amplitude=float(forcing.get("wind_amplitude", 0.0)),
        wind_profile=str(forcing.get("wind_profile", "doublegyre")),
        poisson_bc=str(model_params.get("poisson_bc", "dst")),
    )

    ic = scenario.initial_condition
    if ic.type != "at_rest":
        raise NotImplementedError(
            f"multilayer_qg: initial_condition.type={ic.type!r} not supported "
            "(only 'at_rest' is wired up in Phase 3)."
        )
    nl = model.consts.n_layers
    state0 = BaroclinicQGState(q=jnp.zeros((nl, model.grid.Ny, model.grid.Nx)))
    return BuiltModel(model=model, state0=state0)


MULTILAYER_QG = ModelEntry(
    name="multilayer_qg",
    family="qg",
    layers="multi",
    coordinates="cartesian",
    supports=SupportFlags(masks=True, spherical=False, forcing=("tau_x", "tau_y")),
    build=_build,
)
