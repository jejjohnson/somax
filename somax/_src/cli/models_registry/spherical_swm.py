"""Spherical shallow water model entry.

Stubbed in Phase 2 (#76). Populated in Phase 5 after the spherical
operators land in finitevolX (#165) and the somax-side spherical
SWM class exists (D4 in epic #72).
"""

from __future__ import annotations

from typing import Any

from somax._src.cli.scenarios import ScenarioBundle

from ._types import BuiltModel, ModelEntry, SupportFlags


def _build(scenario: ScenarioBundle, params: dict[str, Any]) -> BuiltModel:
    raise NotImplementedError(
        "spherical_swm.build is stubbed; blocked on Phase 5 "
        "(spherical models in somax — requires finitevolX#165 and "
        "D4 in epic #72)."
    )


SPHERICAL_SWM = ModelEntry(
    name="spherical_swm",
    family="swm",
    layers=1,
    coordinates="spherical",
    supports=SupportFlags(masks=True, spherical=True, forcing=("tau_x", "tau_y")),
    build=_build,
)
