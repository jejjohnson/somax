"""Barotropic quasi-geostrophic model entry.

Stubbed in Phase 2 (#76). Populated in Phase 3 using
:class:`somax.models.BarotropicQG`.
"""

from __future__ import annotations

from typing import Any

from somax._src.cli.scenarios import ScenarioBundle

from ._types import BuiltModel, ModelEntry, SupportFlags


def _build(scenario: ScenarioBundle, params: dict[str, Any]) -> BuiltModel:
    raise NotImplementedError(
        "barotropic_qg.build is stubbed; blocked on Phase 3 (port "
        "BarotropicQG adapter — see epic #72)."
    )


BAROTROPIC_QG = ModelEntry(
    name="barotropic_qg",
    family="qg",
    layers=1,
    coordinates="cartesian",
    supports=SupportFlags(masks=True, spherical=False, forcing=("tau_x", "tau_y")),
    build=_build,
)
