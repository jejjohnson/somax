"""Reparameterized multilayer quasi-geostrophic model entry.

Phase 3 (#77) implements the ``double_gyre`` x ``reparam_multilayer_qg``
adapter. Uses :class:`somax.models.ReparameterizedQG`, which solves
QG in ``(u, v, h)`` state space with a geostrophic projection
(Thiry et al. 2024).
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from somax._src.cli.scenarios import ScenarioBundle

from ._helpers import require_stratification
from ._types import BuiltModel, ModelEntry, SupportFlags


def _build(scenario: ScenarioBundle, params: dict[str, Any]) -> BuiltModel:
    from somax.models import MultilayerSW2DState, ReparameterizedQG

    geometry = scenario.geometry
    if geometry.Lx is None or geometry.Ly is None:
        raise ValueError(
            "reparam_multilayer_qg requires a Cartesian scenario geometry with Lx/Ly."
        )
    consts = scenario.constants
    forcing = scenario.forcing_params
    model_params = dict(params.get("params", {}))

    H, g_prime = require_stratification(params, "reparam_multilayer_qg")

    model = ReparameterizedQG.create(
        nx=geometry.nx,
        ny=geometry.ny,
        Lx=geometry.Lx,
        Ly=geometry.Ly,
        g=consts.g,
        f0=consts.f0,
        beta=consts.beta,
        n_layers=len(H),
        H=H,
        g_prime=g_prime,
        lateral_viscosity=float(model_params.get("lateral_viscosity", 0.0)),
        bottom_drag=float(model_params.get("bottom_drag", 0.0)),
        wind_amplitude=float(forcing.get("wind_amplitude", 0.0)),
        wind_profile=str(forcing.get("wind_profile", "doublegyre")),
        bc=str(model_params.get("bc", "wall")),
        method=str(model_params.get("method", "upwind1")),
        poisson_bc=str(model_params.get("poisson_bc", "dst")),
    )

    ic = scenario.initial_condition
    if ic.type != "at_rest":
        raise NotImplementedError(
            f"reparam_multilayer_qg: initial_condition.type={ic.type!r} not "
            "supported (only 'at_rest' is wired up in Phase 3)."
        )
    nl = model.consts.n_layers
    ny, nx = model.grid.Ny, model.grid.Nx
    h0 = jnp.broadcast_to(model.strat.H[:, None, None], (nl, ny, nx))
    state0 = MultilayerSW2DState(
        h=h0,
        u=jnp.zeros((nl, ny, nx)),
        v=jnp.zeros((nl, ny, nx)),
    )
    return BuiltModel(model=model, state0=state0)


REPARAM_MULTILAYER_QG = ModelEntry(
    name="reparam_multilayer_qg",
    family="qg",
    layers="multi",
    coordinates="cartesian",
    supports=SupportFlags(masks=True, spherical=False, forcing=("tau_x", "tau_y")),
    build=_build,
)
