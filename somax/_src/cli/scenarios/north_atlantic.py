"""North Atlantic basin — masked Cartesian geometry.

Stubbed in Phase 2 (#76). Populated in Phase 4 after the mask
plumbing lands (#29 / finitevolX#185) and a real-basin data source
is picked (D6 / D7 in epic #72).
"""

from __future__ import annotations

from typing import Any

from ._types import ScenarioBundle, ScenarioEntry


def _build(params: dict[str, Any]) -> ScenarioBundle:
    raise NotImplementedError(
        "north_atlantic.build is stubbed; blocked on Phase 4 "
        "(real-basin scenarios — requires masks (#29 / finitevolX#185) "
        "and a basin-geometry data source (D6/D7 in epic #72))."
    )


NORTH_ATLANTIC = ScenarioEntry(
    name="north_atlantic",
    geometry_kind="real_basin",
    build=_build,
)
