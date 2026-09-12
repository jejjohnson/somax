"""Spherical quasi-geostrophic model entry.

Wires :class:`somax.models.SphericalQG` (#73) into the registry.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from somax._src.cli.scenarios import ScenarioBundle

from ._types import BuiltModel, ModelEntry, SupportFlags
from .spherical_swm import (
    _forcing_fields,
    _numerics,
    _require_spherical,
    _spherical_mask,
    _wind_amplitude,
)


def _reject_tau_y(scenario: ScenarioBundle) -> None:
    """Refuse a meridional stress this model has nowhere to put.

    ``SphericalQG`` advances a vorticity tendency, so its forcing is
    the stress *curl* — one T-point field, taken from ``tau_x``. There
    is no second component to consume, and silently dropping a
    supplied ``tau_y`` would give a run that is forced by half of what
    the scenario asked for.
    """
    if getattr(scenario.forcing, "tau_y", None) is not None:
        raise ValueError(
            "spherical_qg is forced by the wind stress curl, supplied as "
            "the single field 'tau_x'; it has no use for 'tau_y'. Combine "
            "the two components into a curl before building the scenario, "
            "or pair this scenario with spherical_swm, which takes both."
        )


def _build(scenario: ScenarioBundle, params: dict[str, Any]) -> BuiltModel:
    from somax.models import SphericalQG, SphericalQGState

    lon_bounds, lat_bounds = _require_spherical(scenario, "spherical_qg")
    geometry = scenario.geometry
    forcing = scenario.forcing_params
    model_params = dict(params.get("params", {}))
    _reject_tau_y(scenario)
    # QG is forced by the stress *curl*: a single T-point vorticity
    # source, which is what ``wind_forcing`` is. A scenario supplies
    # that pattern as ``tau_x``.
    fields = _forcing_fields(
        scenario, (geometry.ny + 2, geometry.nx + 2), "wind_forcing", "tau_x"
    )

    model = SphericalQG.create(
        nx=geometry.nx,
        ny=geometry.ny,
        lon_range=lon_bounds,
        lat_range=lat_bounds,
        lateral_viscosity=float(model_params.get("lateral_viscosity", 0.0)),
        bottom_drag=float(model_params.get("bottom_drag", 0.0)),
        wind_amplitude=_wind_amplitude(forcing, fields),
        wind_profile=str(forcing.get("wind_profile", "zonal")),
        mask=_spherical_mask(scenario),
        **fields,
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
    # ``tau_x`` only: this model is forced by the stress *curl*, a
    # single T-point vorticity source, not by a stress vector. A
    # supplied ``tau_y`` is rejected rather than silently dropped
    # (see ``_reject_tau_y``).
    supports=SupportFlags(masks=True, spherical=True, forcing=("tau_x",)),
    build=_build,
)
