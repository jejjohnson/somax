"""Global ocean — full-sphere geometry with continental masks.

Stubbed in Phase 2 (#76). Populated in Phase 5 after the spherical
operators land in finitevolX (#165), somax-side spherical models
(D4) exist, and a global bathymetry / coastline data source is
picked (D6/D7 in epic #72).

Geometry kind is ``spherical_cap`` — a "cap" spanning the full sphere
is the same type as a regional cap; the continental distribution
enters as the ``Mask2D`` on the T-grid, not as a new geometry kind.
"""

from __future__ import annotations

from typing import Any

from ._types import ScenarioBundle, ScenarioEntry


def _build(params: dict[str, Any]) -> ScenarioBundle:
    raise NotImplementedError(
        "global_ocean.build is stubbed; blocked on Phase 5 (spherical "
        "scenario + plumbing — requires finitevolX#165 for spherical "
        "operators, D4 for spherical somax models, and D6/D7 for a "
        "global bathymetry data source — see epic #72)."
    )


GLOBAL_OCEAN = ScenarioEntry(
    name="global_ocean",
    geometry_kind="spherical_cap",
    build=_build,
)
