"""Southern Ocean — masked spherical-cap geometry.

Stubbed in Phase 2 (#76). Populated in Phase 5 after the spherical
operators land in finitevolX (#165) and the spherical somax models
are written (D4 in epic #72).
"""

from __future__ import annotations

from typing import Any

from ._types import ScenarioBundle, ScenarioEntry


def _build(params: dict[str, Any]) -> ScenarioBundle:
    raise NotImplementedError(
        "southern_ocean.build is stubbed; blocked on Phase 5 "
        "(spherical scenario + plumbing — requires finitevolX#165 for "
        "spherical operators and D4 for spherical models in somax)."
    )


SOUTHERN_OCEAN = ScenarioEntry(
    name="southern_ocean",
    geometry_kind="spherical_cap",
    build=_build,
)
