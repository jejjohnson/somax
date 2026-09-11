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


#: YAML keys accepted in a ``scenario.nondim`` block, mapped to the
#: factory's argument names. ``munk`` and ``stommel`` read better in a
#: config than ``delta_M`` and ``delta_S``.
_NONDIM_KEYS = {
    "rossby": "rossby",
    "beta_hat": "beta_hat",
    "munk": "delta_M",
    "stommel": "delta_S",
    "inertial": "delta_I",
    "aspect": "aspect",
    "check_resolution": "check_resolution",
}


def _build_nondimensional(
    scenario: ScenarioBundle, nondim: dict[str, Any]
) -> BuiltModel:
    """Build from a ``scenario.nondim`` block instead of SI constants."""
    from somax.models import BarotropicQG, BarotropicQGState

    unknown = sorted(set(nondim) - set(_NONDIM_KEYS))
    if unknown:
        raise ValueError(
            f"barotropic_qg: unknown scenario.nondim key(s) {unknown}. "
            f"Accepted: {sorted(_NONDIM_KEYS)}."
        )
    missing = [key for key in ("rossby", "beta_hat", "munk") if key not in nondim]
    if missing:
        raise ValueError(
            f"barotropic_qg: scenario.nondim missing required key(s) {missing}. "
            "'rossby', 'beta_hat' and 'munk' have no sensible default."
        )

    kwargs = {_NONDIM_KEYS[key]: value for key, value in nondim.items()}
    geometry = scenario.geometry
    model, _ = BarotropicQG.from_nondimensional(
        nx=geometry.nx,
        ny=geometry.ny,
        wind_profile=str(scenario.forcing_params.get("wind_profile", "doublegyre")),
        **kwargs,
    )

    ic = scenario.initial_condition
    if ic.type != "at_rest":
        raise NotImplementedError(
            f"barotropic_qg: initial_condition.type={ic.type!r} not supported "
            "for a nondimensional build (only 'at_rest')."
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
    from_nondimensional=_build_nondimensional,
)
