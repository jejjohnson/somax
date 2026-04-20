"""Structured types that every scenario produces.

A *scenario* owns what we simulate — geometry (grid / basin shape),
basin constants (``f0``, ``beta``, …), forcing fields, and the initial
condition shape. A *model* (see :mod:`somax._src.cli.models_registry`)
owns how we simulate it (equations of motion, integration). The
scenario x model decomposition is tracked by epic #72; this module
contains the dataclasses they exchange.

Phase 2 (#76) introduces these types and the empty registries.
Phases 3-5 populate the registries.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from jaxtyping import Array, Float


GeometryKind = Literal["rectangular", "spherical_cap", "real_basin"]


@dataclass(frozen=True)
class Geometry:
    """Domain geometry.

    ``kind`` is the compatibility-dispatch key — models expose a
    matching ``coordinates`` flag plus a ``supports.masks`` bit and the
    compatibility checker pairs them (see
    :mod:`somax._src.cli._compatibility`).

    Args:
        kind: ``"rectangular"`` for unmasked Cartesian basins,
            ``"real_basin"`` for masked Cartesian basins (coastlines),
            or ``"spherical_cap"`` for spherical-geometry sectors.
        nx: Number of interior cells in x.
        ny: Number of interior cells in y.
        Lx: Domain length in x (m). Required for Cartesian geometries.
        Ly: Domain length in y (m). Required for Cartesian geometries.
        lon_bounds: ``(lon_min, lon_max)`` in degrees. Required for
            ``spherical_cap``.
        lat_bounds: ``(lat_min, lat_max)`` in degrees. Required for
            ``spherical_cap``.
        mask: Optional wet/dry mask on the T-grid. Required when
            ``kind`` is ``"real_basin"`` or a masked spherical cap.
        bathymetry: Optional bottom topography on the T-grid.
    """

    kind: GeometryKind
    nx: int
    ny: int
    Lx: float | None = None
    Ly: float | None = None
    lon_bounds: tuple[float, float] | None = None
    lat_bounds: tuple[float, float] | None = None
    mask: Float[Array, "ny nx"] | None = None
    bathymetry: Float[Array, "ny nx"] | None = None


@dataclass(frozen=True)
class Constants:
    """Frozen basin constants that the model consumes.

    Args:
        f0: Reference Coriolis parameter (1/s).
        beta: Meridional gradient of ``f`` (1/(m·s)).
        rho0: Reference density (kg/m³).
        g: Gravitational acceleration (m/s²).
    """

    f0: float
    beta: float
    rho0: float = 1025.0
    g: float = 9.81


@dataclass(frozen=True)
class ForcingFields:
    """Precomputed forcing fields, per scenario.

    All fields are optional so each scenario only fills the ones that
    are physically meaningful. Models declare which fields they consume
    via :class:`SupportFlags.forcing`.

    Args:
        tau_x: Zonal wind stress pattern on the T-grid (Pa or normalised).
        tau_y: Meridional wind stress pattern on the T-grid.
        heat: Surface heat-flux pattern on the T-grid.
    """

    tau_x: Float[Array, "ny nx"] | None = None
    tau_y: Float[Array, "ny nx"] | None = None
    heat: Float[Array, "ny nx"] | None = None


InitialConditionKind = Literal["at_rest", "jet", "gaussian_eddy", "prescribed"]


@dataclass(frozen=True)
class InitialConditionSpec:
    """Initial-condition shape for a scenario.

    The ``type`` is a registry key; the model's ``build`` uses it to
    pick an appropriate initial state (every model has to support at
    least ``"at_rest"``). Shape-specific numbers live in ``params``.

    Args:
        type: One of the registered initial-condition shapes.
        params: Keyword arguments for that shape (e.g. ``jet_speed``,
            ``jet_width``). Empty dict when the shape is parameter-free.
    """

    type: InitialConditionKind
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScenarioBundle:
    """Everything a model needs from a scenario.

    Produced by ``ScenarioEntry.build(params)`` at dispatch time;
    consumed by ``ModelEntry.build(bundle, model_params)``.

    Args:
        name: Scenario registry key (informational — the registry is the
            authoritative source).
        geometry: Grid / basin shape.
        constants: Frozen basin constants.
        forcing: Precomputed forcing fields (``tau_x`` / ``tau_y`` /
            ``heat``). Optional — scenarios with only scalar forcing
            knobs (e.g. ``double_gyre``'s ``wind_amplitude``) leave the
            spatial fields as ``None`` and expose the scalars via
            :data:`forcing_params` instead.
        initial_condition: Shape + parameters for the initial state.
        forcing_params: Scenario-level scalar forcing knobs consumed by
            model ``build`` callables. Keys are scenario-specific
            (``wind_amplitude``, ``wind_profile``, …). Phase 3 uses this
            for the Stommel-style wind amplitude on ``double_gyre``;
            Phase 4/5 scenarios that pre-compute ``tau_x`` / ``tau_y``
            fields can leave it empty.
    """

    name: str
    geometry: Geometry
    constants: Constants
    forcing: ForcingFields
    initial_condition: InitialConditionSpec
    forcing_params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScenarioEntry:
    """Registry entry for a scenario.

    Every registered scenario exposes its ``geometry_kind`` (consumed by
    the compatibility checker) and a ``build`` callable that produces a
    :class:`ScenarioBundle` from a parameter dict. Phase-2 stubs raise
    :class:`NotImplementedError` from ``build`` until the later phases
    populate them.

    Args:
        name: Unique registry key. Must match the key in
            :data:`somax._src.cli.scenarios.SCENARIOS`.
        geometry_kind: The :data:`Geometry.kind` the built bundle will
            carry. Exposed ahead of ``build`` so the compatibility
            checker can work against a stub entry.
        build: Callable that turns a parameter dict into a fully built
            :class:`ScenarioBundle`. Parameters are the scenario-side
            knobs in a YAML ``scenario:`` block (``wind_amplitude``,
            ``jet_speed``, …).
    """

    name: str
    geometry_kind: GeometryKind
    build: Callable[[dict[str, Any]], ScenarioBundle]
