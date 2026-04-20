"""Linear shallow water model entry.

Stubbed in Phase 2 (#76). Populated in Phase 3 using
:class:`somax.models.LinearShallowWater2D`.

Note: ``supports.masks = False`` — the linear SWM is deliberately
unmasked; the compatibility checker routes this model to
``double_gyre`` only (the sole rectangular-unmasked scenario).
"""

from __future__ import annotations

from typing import Any

from somax._src.cli.scenarios import ScenarioBundle

from ._types import BuiltModel, ModelEntry, SupportFlags


def _build(scenario: ScenarioBundle, params: dict[str, Any]) -> BuiltModel:
    raise NotImplementedError(
        "linear_swm.build is stubbed; blocked on Phase 3 (port "
        "LinearShallowWater2D adapter — see epic #72)."
    )


LINEAR_SWM = ModelEntry(
    name="linear_swm",
    family="swm",
    layers=1,
    coordinates="cartesian",
    supports=SupportFlags(masks=False, spherical=False, forcing=()),
    build=_build,
)
