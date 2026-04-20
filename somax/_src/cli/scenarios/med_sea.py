"""Mediterranean Sea basin — masked Cartesian geometry.

Stubbed in Phase 2 (#76). Populated in Phase 4 (real-basin scenarios;
see ``north_atlantic`` for the same dependency notes).
"""

from __future__ import annotations

from typing import Any

from ._types import ScenarioBundle, ScenarioEntry


def _build(params: dict[str, Any]) -> ScenarioBundle:
    raise NotImplementedError(
        "med_sea.build is stubbed; blocked on Phase 4 (real-basin "
        "scenarios — requires masks (#29 / finitevolX#185) and a "
        "basin-geometry data source (D6/D7 in epic #72))."
    )


MED_SEA = ScenarioEntry(
    name="med_sea",
    geometry_kind="real_basin",
    build=_build,
)
