"""Structured types for the model registry.

A *model* owns how we simulate — equations of motion, state pytree,
integration wiring. It consumes a :class:`ScenarioBundle` (see
:mod:`somax._src.cli.scenarios`) plus per-model parameters and
produces a :class:`BuiltModel` ready for the runner.

Phase 2 (#76) ships these types and the empty registry. Phases 3-5
populate the stubs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from somax._src.cli.scenarios import ScenarioBundle


Family = Literal["swm", "qg"]
Layers = Literal[1, "multi"]
Coordinates = Literal["cartesian", "spherical"]


@dataclass(frozen=True)
class SupportFlags:
    """Capability flags a model advertises for compatibility checking.

    Args:
        masks: ``True`` iff the model's operators thread an
            Arakawa C-grid mask (``Mask1D`` / ``Mask2D``) through
            ``self.mask`` — required to pair the model with any masked
            scenario (real basins, spherical caps with bathymetry).
        spherical: ``True`` iff the model is built on spherical
            operators. Should track ``coordinates == "spherical"``.
        forcing: Tuple of :class:`ForcingFields` field names the model
            actually consumes (e.g. ``("tau_x", "tau_y")``). Empty
            means the model doesn't read any forcing.
    """

    masks: bool
    spherical: bool
    forcing: tuple[str, ...] = ()


@dataclass(frozen=True)
class BuiltModel:
    """Runner-ready output of a model build.

    Args:
        model: The somax model instance (an ``eqx.Module`` subclass).
        state0: The initial state, already post-applied with the
            model's boundary conditions.
    """

    model: Any
    state0: Any


@dataclass(frozen=True)
class ModelEntry:
    """Registry entry for a model.

    Exposes enough metadata for the compatibility checker to decide
    whether a ``(scenario, model)`` pair is valid without calling
    ``build`` — this lets stubbed Phase-2 entries participate in
    compatibility logic ahead of their Phase-3/4/5 implementations.

    Args:
        name: Unique registry key.
        family: ``"swm"`` (shallow water) or ``"qg"``
            (quasi-geostrophic).
        layers: ``1`` for single-layer models, ``"multi"`` for
            vertically stratified multilayer models.
        coordinates: ``"cartesian"`` or ``"spherical"``. Must match the
            :data:`SupportFlags.spherical` bit.
        supports: Capability flags — masks, spherical, consumed
            forcing fields.
        build: Callable that takes a :class:`ScenarioBundle` plus a
            model-parameter dict and returns a :class:`BuiltModel`.
            Stubs in Phase 2 raise :class:`NotImplementedError`.
    """

    name: str
    family: Family
    layers: Layers
    coordinates: Coordinates
    supports: SupportFlags
    build: Callable[[ScenarioBundle, dict[str, Any]], BuiltModel] = field(repr=False)
