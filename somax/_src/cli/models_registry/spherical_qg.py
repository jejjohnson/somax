"""Spherical quasi-geostrophic model entry.

Wires :class:`somax.models.SphericalQG` (#73) into the registry.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from somax._src.cli.scenarios import ScenarioBundle

from ._types import BuiltModel, ModelEntry, SupportFlags
from .spherical_swm import _numerics, _require_spherical, _spherical_mask


def _build(scenario: ScenarioBundle, params: dict[str, Any]) -> BuiltModel:
    from somax.models import SphericalQG, SphericalQGState

    lon_bounds, lat_bounds = _require_spherical(scenario, "spherical_qg")
    geometry = scenario.geometry
    forcing = scenario.forcing_params
    model_params = dict(params.get("params", {}))

    model = SphericalQG.create(
        nx=geometry.nx,
        ny=geometry.ny,
        lon_range=lon_bounds,
        lat_range=lat_bounds,
        lateral_viscosity=float(model_params.get("lateral_viscosity", 0.0)),
        bottom_drag=float(model_params.get("bottom_drag", 0.0)),
        wind_amplitude=float(forcing.get("wind_amplitude", 0.0)),
        wind_profile=str(forcing.get("wind_profile", "zonal")),
        mask=_spherical_mask(scenario),
        **_numerics(params, "method", "cg_tol", "cg_max_steps"),
    )

    ic = scenario.initial_condition
    if ic.type != "at_rest":
        raise NotImplementedError(
            f"spherical_qg: initial_condition.type={ic.type!r} not supported "
            "(only 'at_rest')."
        )
    state0 = SphericalQGState(q=jnp.zeros((model.grid.Ny, model.grid.Nx)))
    return BuiltModel(model=model, state0=state0)


SPHERICAL_QG = ModelEntry(
    name="spherical_qg",
    family="qg",
    layers=1,
    coordinates="spherical",
    supports=SupportFlags(masks=True, spherical=True, forcing=("tau_x", "tau_y")),
    build=_build,
)
