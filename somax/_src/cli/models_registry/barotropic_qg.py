"""Barotropic quasi-geostrophic model entry.

Phase 3 (#77) implements the ``double_gyre`` x ``barotropic_qg``
adapter — the port of the legacy ``doublegyre_qg`` test case. Uses
:class:`somax.models.BarotropicQG`.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from somax._src.cli.scenarios import ScenarioBundle

from ._types import BuiltModel, ModelEntry, SupportFlags


def _build(scenario: ScenarioBundle, params: dict[str, Any]) -> BuiltModel:
    from somax.models import BarotropicQG, BarotropicQGState

    geometry = scenario.geometry
    if geometry.Lx is None or geometry.Ly is None:
        raise ValueError(
            "barotropic_qg requires a Cartesian scenario geometry with Lx/Ly."
        )
    consts = scenario.constants
    forcing = scenario.forcing_params
    model_params = dict(params.get("params", {}))

    model = BarotropicQG.create(
        nx=geometry.nx,
        ny=geometry.ny,
        Lx=geometry.Lx,
        Ly=geometry.Ly,
        f0=consts.f0,
        beta=consts.beta,
        lateral_viscosity=float(model_params.get("lateral_viscosity", 0.0)),
        bottom_drag=float(model_params.get("bottom_drag", 0.0)),
        wind_amplitude=float(forcing.get("wind_amplitude", 0.0)),
        wind_profile=str(forcing.get("wind_profile", "doublegyre")),
    )

    ic = scenario.initial_condition
    if ic.type != "at_rest":
        raise NotImplementedError(
            f"barotropic_qg: initial_condition.type={ic.type!r} not supported "
            "(only 'at_rest' is wired up in Phase 3)."
        )
    state0 = BarotropicQGState(q=jnp.zeros((model.grid.Ny, model.grid.Nx)))
    return BuiltModel(model=model, state0=state0)


BAROTROPIC_QG = ModelEntry(
    name="barotropic_qg",
    family="qg",
    layers=1,
    coordinates="cartesian",
    supports=SupportFlags(masks=True, spherical=False, forcing=("tau_x", "tau_y")),
    build=_build,
)
