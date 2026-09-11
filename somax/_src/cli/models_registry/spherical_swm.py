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


#: YAML keys accepted in a ``scenario.nondim`` block, mapped to the
#: factory's argument names. They match the factory one-for-one; the
#: mapping exists so an unknown key is rejected by name.
_NONDIM_KEYS = {
    "rossby": "rossby",
    "burger": "burger",
    "froude": "froude",
    "lamb": "lamb",
    "ekman": "ekman",
    "ekman_lateral": "ekman_lateral",
    "wind_hat": "wind_hat",
    "check_resolution": "check_resolution",
}


def _build_nondimensional(
    scenario: ScenarioBundle, nondim: dict[str, Any]
) -> BuiltModel:
    """Build from a ``scenario.nondim`` block instead of SI constants."""
    from somax.models import SphericalSWM, SphericalSWMState

    lon_bounds, lat_bounds = _require_spherical(scenario, "spherical_swm")
    unknown = sorted(set(nondim) - set(_NONDIM_KEYS))
    if unknown:
        raise ValueError(
            f"spherical_swm: unknown scenario.nondim key(s) {unknown}. "
            f"Accepted: {sorted(_NONDIM_KEYS)}."
        )
    if "rossby" not in nondim:
        raise ValueError(
            "spherical_swm: scenario.nondim requires 'rossby'; it has no "
            "sensible default."
        )
    if not {"burger", "froude", "lamb"} & set(nondim):
        raise ValueError(
            "spherical_swm: scenario.nondim requires exactly one of "
            "'burger', 'froude' or 'lamb' to set the stratification."
        )

    kwargs = {_NONDIM_KEYS[key]: value for key, value in nondim.items()}
    geometry = scenario.geometry
    model, _ = SphericalSWM.from_nondimensional(
        nx=geometry.nx,
        ny=geometry.ny,
        lon_range=lon_bounds,
        lat_range=lat_bounds,
        wind_profile=str(scenario.forcing_params.get("wind_profile", "zonal")),
        mask=geometry.mask,
        **kwargs,
    )

    ic = scenario.initial_condition
    if ic.type != "at_rest":
        raise NotImplementedError(
            f"spherical_swm: initial_condition.type={ic.type!r} not supported "
            "for a nondimensional build (only 'at_rest')."
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
    from_nondimensional=_build_nondimensional,
)
