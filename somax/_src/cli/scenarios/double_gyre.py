"""Idealised wind-driven double-gyre on a rectangular basin.

Stubbed in Phase 2 (#76). Populated in Phase 3 (port of the existing
``doublegyre_qg`` + ``doublegyre_baroclinic_qg`` test cases into the
scenario x model decomposition).
"""

from __future__ import annotations

from typing import Any

from ._types import ScenarioBundle, ScenarioEntry


def _build(params: dict[str, Any]) -> ScenarioBundle:
    raise NotImplementedError(
        "double_gyre.build is stubbed; blocked on Phase 3 (port of "
        "existing double-gyre test cases into the scenario x model "
        "decomposition — see epic #72)."
    )


DOUBLE_GYRE = ScenarioEntry(
    name="double_gyre",
    geometry_kind="rectangular",
    build=_build,
)
