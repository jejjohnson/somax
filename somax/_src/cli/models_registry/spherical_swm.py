"""Spherical shallow water model entry.

Wires :class:`somax.models.SphericalSWM` (#73) into the registry.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from somax._src.cli.scenarios import ScenarioBundle

from ._types import BuiltModel, ModelEntry, SupportFlags


def _require_spherical(scenario: ScenarioBundle, name: str) -> tuple:
    """Pull the lon/lat bounds a spherical model needs off the bundle."""
    geometry = scenario.geometry
    if geometry.lon_bounds is None or geometry.lat_bounds is None:
        raise ValueError(
            f"{name} requires a spherical scenario geometry with "
            f"lon_bounds / lat_bounds; got kind={geometry.kind!r}."
        )
    return geometry.lon_bounds, geometry.lat_bounds


def _build(scenario: ScenarioBundle, params: dict[str, Any]) -> BuiltModel:
    from somax.models import SphericalSWM, SphericalSWMState

    lon_bounds, lat_bounds = _require_spherical(scenario, "spherical_swm")
    geometry = scenario.geometry
    forcing = scenario.forcing_params
    model_params = dict(params.get("params", {}))
    stratification = dict(params.get("stratification", {}))

    model = SphericalSWM.create(
        nx=geometry.nx,
        ny=geometry.ny,
        lon_range=lon_bounds,
        lat_range=lat_bounds,
        H0=float(stratification.get("H0", 1000.0)),
        lateral_viscosity=float(model_params.get("lateral_viscosity", 0.0)),
        bottom_drag=float(model_params.get("bottom_drag", 0.0)),
        wind_amplitude=float(forcing.get("wind_amplitude", 0.0)),
        wind_profile=str(forcing.get("wind_profile", "zonal")),
        mask=geometry.mask,
    )

    ic = scenario.initial_condition
    if ic.type != "at_rest":
        raise NotImplementedError(
            f"spherical_swm: initial_condition.type={ic.type!r} not supported "
            "(only 'at_rest')."
        )
    shape = (model.grid.Ny, model.grid.Nx)
    state0 = SphericalSWMState(
        h=jnp.full(shape, model.consts.H0),
        u=jnp.zeros(shape),
        v=jnp.zeros(shape),
    )
    return BuiltModel(model=model, state0=state0)


SPHERICAL_SWM = ModelEntry(
    name="spherical_swm",
    family="swm",
    layers=1,
    coordinates="spherical",
    supports=SupportFlags(masks=True, spherical=True, forcing=("tau_x", "tau_y")),
    build=_build,
)
