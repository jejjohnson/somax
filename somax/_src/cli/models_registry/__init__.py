"""Model registry — how we simulate (equations + integration).

Phase 2 (#76) ships this module with all eight models registered as
:class:`ModelEntry` instances whose ``build`` raises
:class:`NotImplementedError`. Phases 3 / 5 populate the stubs.

Public API
----------
- :data:`MODELS` — registry dict, keyed by model name.
- :func:`list_models` — sorted names.
- :func:`get_model` — lookup by name with a helpful error.
- Types re-exported from :mod:`._types`.
"""

from __future__ import annotations

from ._types import (
    BuiltModel,
    Coordinates,
    Family,
    Layers,
    ModelEntry,
    SupportFlags,
)
from .barotropic_qg import BAROTROPIC_QG
from .linear_swm import LINEAR_SWM
from .multilayer_nonlinear_swm import MULTILAYER_NONLINEAR_SWM
from .multilayer_qg import MULTILAYER_QG
from .nonlinear_swm import NONLINEAR_SWM
from .reparam_multilayer_qg import REPARAM_MULTILAYER_QG
from .spherical_qg import SPHERICAL_QG
from .spherical_swm import SPHERICAL_SWM


MODELS: dict[str, ModelEntry] = {
    "linear_swm": LINEAR_SWM,
    "nonlinear_swm": NONLINEAR_SWM,
    "barotropic_qg": BAROTROPIC_QG,
    "multilayer_nonlinear_swm": MULTILAYER_NONLINEAR_SWM,
    "multilayer_qg": MULTILAYER_QG,
    "reparam_multilayer_qg": REPARAM_MULTILAYER_QG,
    "spherical_swm": SPHERICAL_SWM,
    "spherical_qg": SPHERICAL_QG,
}


def list_models() -> list[str]:
    """Return registered model names, sorted alphabetically."""
    return sorted(MODELS)


def get_model(name: str) -> ModelEntry:
    """Look up a model by name. Raises :class:`KeyError` with a helpful message."""
    try:
        return MODELS[name]
    except KeyError as exc:
        available = ", ".join(list_models())
        raise KeyError(f"unknown model {name!r}; available: {available}") from exc


__all__ = [
    "MODELS",
    "BuiltModel",
    "Coordinates",
    "Family",
    "Layers",
    "ModelEntry",
    "SupportFlags",
    "get_model",
    "list_models",
]
