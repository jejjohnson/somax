"""Idealised wind-driven double-gyre on a rectangular basin.

Phase 3 (#77) populates this scenario. It carries the rectangular
Cartesian geometry, Stommel-style double-gyre wind-stress knob
(``forcing.wind_amplitude``), and three IC variants (``at_rest``,
``jet``, ``gaussian_eddy``) via :class:`InitialConditionSpec`. Models
read the bundle in their own ``build`` callables and translate the IC
type into a concrete state pytree (single- vs multi-layer shapes
differ per model).

The underlying wind profile is the standard sinusoidal doublegyre
pattern exposed by the existing model factories
(``wind_profile="doublegyre"``). Phase 4 will introduce real-basin
climatology forcing via precomputed :class:`ForcingFields`.
"""

from __future__ import annotations

from typing import Any

from ._types import (
    Constants,
    ForcingFields,
    Geometry,
    InitialConditionKind,
    InitialConditionSpec,
    ScenarioBundle,
    ScenarioEntry,
)


_VALID_IC_TYPES: tuple[InitialConditionKind, ...] = (
    "at_rest",
    "jet",
    "gaussian_eddy",
)


def _build(params: dict[str, Any]) -> ScenarioBundle:
    """Assemble a :class:`ScenarioBundle` from a ``scenario:`` YAML block.

    Args:
        params: Nested dict with ``grid``, ``consts``, ``forcing``, and
            ``initial_condition`` sub-blocks. Missing sub-blocks get
            sensible defaults (the basin is 1000 km² with f-plane
            defaults if nothing is specified).
    """
    grid = dict(params.get("grid", {}))
    consts = dict(params.get("consts", {}))
    forcing = dict(params.get("forcing", {}))
    ic = dict(params.get("initial_condition", {}))

    try:
        nx = int(grid["nx"])
        ny = int(grid["ny"])
    except KeyError as exc:
        raise ValueError(
            f"double_gyre: scenario.grid missing required key {exc.args[0]!r}; "
            "both 'nx' and 'ny' are required."
        ) from exc

    geometry = Geometry(
        kind="rectangular",
        nx=nx,
        ny=ny,
        Lx=float(grid.get("Lx", 1.0e6)),
        Ly=float(grid.get("Ly", 1.0e6)),
    )

    constants = Constants(
        f0=float(consts.get("f0", 1.0e-4)),
        beta=float(consts.get("beta", 1.6e-11)),
        rho0=float(consts.get("rho0", 1025.0)),
        g=float(consts.get("g", 9.81)),
    )

    ic_type = str(ic.get("type", "at_rest"))
    if ic_type not in _VALID_IC_TYPES:
        raise ValueError(
            f"double_gyre: unknown initial_condition.type {ic_type!r}; "
            f"supported: {sorted(_VALID_IC_TYPES)}."
        )

    initial_condition = InitialConditionSpec(
        type=ic_type,  # type: ignore[arg-type]
        params=dict(ic.get("params", {})),
    )

    return ScenarioBundle(
        name="double_gyre",
        geometry=geometry,
        constants=constants,
        forcing=ForcingFields(),
        initial_condition=initial_condition,
        forcing_params=forcing,
    )


DOUBLE_GYRE = ScenarioEntry(
    name="double_gyre",
    geometry_kind="rectangular",
    build=_build,
)
