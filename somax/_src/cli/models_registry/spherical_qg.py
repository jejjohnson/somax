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


#: YAML keys accepted in a ``scenario.nondim`` block. There is no
#: ``burger``: barotropic QG is rigid-lid, and no ``beta_hat``: on a
#: sphere the planetary vorticity gradient follows from the geometry.
_NONDIM_KEYS = {
    "rossby": "rossby",
    "ekman": "ekman",
    "ekman_lateral": "ekman_lateral",
    "wind_hat": "wind_hat",
}


def _build_nondimensional(
    scenario: ScenarioBundle, nondim: dict[str, Any]
) -> BuiltModel:
    """Build from a ``scenario.nondim`` block instead of SI constants."""
    from somax.models import SphericalQG, SphericalQGState

    lon_bounds, lat_bounds = _require_spherical(scenario, "spherical_qg")
    unknown = sorted(set(nondim) - set(_NONDIM_KEYS))
    if unknown:
        raise ValueError(
            f"spherical_qg: unknown scenario.nondim key(s) {unknown}. "
            f"Accepted: {sorted(_NONDIM_KEYS)}."
        )
    if "rossby" not in nondim:
        raise ValueError(
            "spherical_qg: scenario.nondim requires 'rossby'; it has no "
            "sensible default."
        )

    kwargs = {_NONDIM_KEYS[key]: value for key, value in nondim.items()}
    geometry = scenario.geometry
    model, _ = SphericalQG.from_nondimensional(
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
            f"spherical_qg: initial_condition.type={ic.type!r} not supported "
            "for a nondimensional build (only 'at_rest')."
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
    from_nondimensional=_build_nondimensional,
)
