"""Spherical shallow water model entry.

Wires :class:`somax.models.SphericalSWM` (#73) into the registry.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from somax._src.cli.scenarios import ScenarioBundle

from ._types import BuiltModel, ModelEntry, SupportFlags


def _spherical_mask(scenario: ScenarioBundle) -> Any:
    """Convert the scenario's raw T-grid mask into a ghosted ``Mask2D``.

    Two mismatches to bridge. ``Geometry.mask`` is a plain wet/dry
    array, but the finitevolx spherical operators read staggered
    fields off a ``Mask2D`` (``.h``, ``.u``, ``.v``), so passing the
    array straight through fails on the first attribute access. And it
    covers only the *physical* cells, while the model's grid carries a
    one-cell ghost ring — so it is padded to the ghosted shape first,
    the same way the model's own boundary conditions treat those
    cells: wrapping in longitude, repeating the edge row in latitude.

    Args:
        scenario: The built scenario bundle.

    Returns:
        A ``Mask2D`` on the ghosted grid, or ``None`` when the scenario
        carries no mask.
    """
    from finitevolx import Mask2D

    mask = scenario.geometry.mask
    if mask is None:
        return None
    wet = np.asarray(mask) > 0.5
    if wet.ndim != 2:
        raise ValueError(f"spherical mask must be 2-D (ny, nx); got shape {wet.shape}.")
    padded = np.pad(wet, ((0, 0), (1, 1)), mode="wrap")
    padded = np.pad(padded, ((1, 1), (0, 0)), mode="edge")
    return Mask2D.from_mask(jnp.asarray(padded))


def _numerics(params: dict[str, Any], *keys: str) -> dict[str, Any]:
    """Pick the numerical knobs a model exposes out of ``model.params``.

    Only the keys the target factory actually accepts, so a typo stays
    an error at the factory rather than being silently dropped here.
    """
    model_params = dict(params.get("params", {}))
    return {key: model_params[key] for key in keys if key in model_params}


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
        g=scenario.constants.g,
        # ``ModelSpec.stratification`` is empty for a single-layer
        # model, so the depth lives in ``model.params`` as it does for
        # the Cartesian shallow-water adapters. The stratification
        # block is still honoured if a config uses it.
        H0=float(model_params.get("H0", stratification.get("H0", 1000.0))),
        lateral_viscosity=float(model_params.get("lateral_viscosity", 0.0)),
        bottom_drag=float(model_params.get("bottom_drag", 0.0)),
        wind_amplitude=float(forcing.get("wind_amplitude", 0.0)),
        wind_profile=str(forcing.get("wind_profile", "zonal")),
        mask=_spherical_mask(scenario),
        **_numerics(params, "method"),
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
