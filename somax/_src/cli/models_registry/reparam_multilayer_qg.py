"""Reparameterized multilayer quasi-geostrophic model entry.

Stubbed in Phase 2 (#76). Populated in Phase 3 using
:class:`somax.models.ReparameterizedQG`.

The underlying class (``ReparameterizedQG``) already exists
(upstream dependency D5 is satisfied per epic #72); the stub only
covers the scenario-facing adapter glue.
"""

from __future__ import annotations

from typing import Any

from somax._src.cli.scenarios import ScenarioBundle

from ._types import BuiltModel, ModelEntry, SupportFlags


def _build(scenario: ScenarioBundle, params: dict[str, Any]) -> BuiltModel:
    raise NotImplementedError(
        "reparam_multilayer_qg.build is stubbed; blocked on Phase 3 "
        "(port ReparameterizedQG adapter — see epic #72)."
    )


REPARAM_MULTILAYER_QG = ModelEntry(
    name="reparam_multilayer_qg",
    family="qg",
    layers="multi",
    coordinates="cartesian",
    supports=SupportFlags(masks=True, spherical=False, forcing=("tau_x", "tau_y")),
    build=_build,
)
