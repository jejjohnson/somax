"""Multilayer nonlinear shallow water model entry.

Stubbed in Phase 2 (#76). Populated in Phase 3 using
:class:`somax.models.MultilayerShallowWater2D`.
"""

from __future__ import annotations

from typing import Any

from somax._src.cli.scenarios import ScenarioBundle

from ._types import BuiltModel, ModelEntry, SupportFlags


def _build(scenario: ScenarioBundle, params: dict[str, Any]) -> BuiltModel:
    raise NotImplementedError(
        "multilayer_nonlinear_swm.build is stubbed; blocked on Phase 3 "
        "(port MultilayerShallowWater2D adapter — see epic #72)."
    )


MULTILAYER_NONLINEAR_SWM = ModelEntry(
    name="multilayer_nonlinear_swm",
    family="swm",
    layers="multi",
    coordinates="cartesian",
    supports=SupportFlags(masks=True, spherical=False, forcing=("tau_x", "tau_y")),
    build=_build,
)
