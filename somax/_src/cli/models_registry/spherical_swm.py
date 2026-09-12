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


def _at_rest(model: Any) -> Any:
    """A resting state, dry over land.

    Filling ``h`` with ``H0`` everywhere would leave water standing on
    land: the masked operators never touch those cells, so the value
    persists into the saved states and into ``diagnose``, where it
    counts towards mass and potential energy.
    """
    from somax.models import SphericalSWMState

    shape = (model.grid.Ny, model.grid.Nx)
    depth = jnp.full(shape, model.consts.H0)
    if model.mask is not None:
        depth = depth * model.mask.h
    return SphericalSWMState(h=depth, u=jnp.zeros(shape), v=jnp.zeros(shape))


def _forcing_fields(scenario: ScenarioBundle, *names: str) -> dict[str, Any]:
    """Pass a scenario's precomputed forcing fields to the model.

    Both spherical entries advertise ``forcing=("tau_x", "tau_y")``, so
    a scenario that supplies those fields expects them to be used.
    Reading only the scalar ``forcing_params`` left such a run on the
    default analytic pattern — usually at zero amplitude, so unforced.
    """
    forcing = scenario.forcing
    out: dict[str, Any] = {}
    for model_name, scenario_name in zip(names[::2], names[1::2], strict=True):
        field = getattr(forcing, scenario_name, None)
        if field is not None:
            out[model_name] = jnp.asarray(field)
    return out


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
    from somax.models import SphericalSWM

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
        **_forcing_fields(scenario, "wind_stress_x", "tau_x", "wind_stress_y", "tau_y"),
        **_numerics(params, "method"),
    )

    ic = scenario.initial_condition
    if ic.type != "at_rest":
        raise NotImplementedError(
            f"spherical_swm: initial_condition.type={ic.type!r} not supported "
            "(only 'at_rest')."
        )
    return BuiltModel(model=model, state0=_at_rest(model))


SPHERICAL_SWM = ModelEntry(
    name="spherical_swm",
    family="swm",
    layers=1,
    coordinates="spherical",
    supports=SupportFlags(masks=True, spherical=True, forcing=("tau_x", "tau_y")),
    build=_build,
)
