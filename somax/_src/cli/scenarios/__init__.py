"""Scenario registry — what we simulate (geometry + forcing + IC).

Phase 2 (#76) ships this module with all five scenarios registered as
:class:`ScenarioEntry` instances whose ``build`` raises
:class:`NotImplementedError`. Phases 3-5 populate the stubs without
touching the registry plumbing.

Public API
----------
- :data:`SCENARIOS` — the registry dict, keyed by scenario name.
- :func:`list_scenarios` — sorted names.
- :func:`get_scenario` — lookup by name with a helpful error.
- Types re-exported from :mod:`._types`.
"""

from __future__ import annotations

from ._types import (
    Constants,
    ForcingFields,
    Geometry,
    GeometryKind,
    InitialConditionKind,
    InitialConditionSpec,
    ScenarioBundle,
    ScenarioEntry,
)
from .double_gyre import DOUBLE_GYRE
from .global_ocean import GLOBAL_OCEAN
from .gulf_stream import GULF_STREAM
from .med_sea import MED_SEA
from .north_atlantic import NORTH_ATLANTIC
from .southern_ocean import SOUTHERN_OCEAN


SCENARIOS: dict[str, ScenarioEntry] = {
    "double_gyre": DOUBLE_GYRE,
    "north_atlantic": NORTH_ATLANTIC,
    "med_sea": MED_SEA,
    "gulf_stream": GULF_STREAM,
    "southern_ocean": SOUTHERN_OCEAN,
    "global_ocean": GLOBAL_OCEAN,
}


def list_scenarios() -> list[str]:
    """Return registered scenario names, sorted alphabetically."""
    return sorted(SCENARIOS)


def get_scenario(name: str) -> ScenarioEntry:
    """Look up a scenario by name. Raises :class:`KeyError` with a helpful message."""
    try:
        return SCENARIOS[name]
    except KeyError as exc:
        available = ", ".join(list_scenarios())
        raise KeyError(f"unknown scenario {name!r}; available: {available}") from exc


__all__ = [
    "SCENARIOS",
    "Constants",
    "ForcingFields",
    "Geometry",
    "GeometryKind",
    "InitialConditionKind",
    "InitialConditionSpec",
    "ScenarioBundle",
    "ScenarioEntry",
    "get_scenario",
    "list_scenarios",
]
