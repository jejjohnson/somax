"""Linear shallow water model entry.

Phase 3 (#77) implements the ``double_gyre`` adapter for
:class:`somax.models.LinearShallowWater2D`.

Note: ``supports.masks = False`` — the linear SWM is deliberately
unmasked; the compatibility checker routes this model to
``double_gyre`` only (the sole rectangular-unmasked scenario).
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from somax._src.cli.scenarios import ScenarioBundle

from ._types import BuiltModel, ModelEntry, SupportFlags


def _build(scenario: ScenarioBundle, params: dict[str, Any]) -> BuiltModel:
    from somax.models import LinearShallowWater2D, LinearSW2DState

    geometry = scenario.geometry
    if geometry.Lx is None or geometry.Ly is None:
        raise ValueError(
            "linear_swm requires a Cartesian scenario geometry with Lx/Ly."
        )
    consts = scenario.constants
    model_params = dict(params.get("params", {}))

    model = LinearShallowWater2D.create(
        nx=geometry.nx,
        ny=geometry.ny,
        Lx=geometry.Lx,
        Ly=geometry.Ly,
        g=consts.g,
        f0=consts.f0,
        beta=consts.beta,
        H0=float(model_params.get("H0", 100.0)),
        lateral_viscosity=float(model_params.get("lateral_viscosity", 0.0)),
        bottom_drag=float(model_params.get("bottom_drag", 0.0)),
        bc=str(model_params.get("bc", "periodic")),
    )

    ic_type = scenario.initial_condition.type
    if ic_type != "at_rest":
        raise NotImplementedError(
            f"linear_swm supports initial_condition.type='at_rest' only; "
            f"got {ic_type!r}."
        )

    ny = model.grid.Ny
    nx = model.grid.Nx
    state0 = LinearSW2DState(
        h=jnp.zeros((ny, nx)),
        u=jnp.zeros((ny, nx)),
        v=jnp.zeros((ny, nx)),
    )
    return BuiltModel(model=model, state0=state0)


LINEAR_SWM = ModelEntry(
    name="linear_swm",
    family="swm",
    layers=1,
    coordinates="cartesian",
    supports=SupportFlags(masks=False, spherical=False, forcing=()),
    build=_build,
)
