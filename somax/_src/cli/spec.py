"""Canonical run specification for somax-sim.

This module defines the :class:`RunSpec` dataclass and its components.
``RunSpec`` is the internal canonical representation of a single
simulation run; everything else in the CLI funnels into it.

Phase 3 (#77) completed the hard cutover from the legacy single-block
``testcase:`` YAML schema to the two-block ``scenario:`` + ``model:``
schema. The structured shape mirrors how somax separates ``Params``
(differentiable) from ``PhysConsts`` (frozen) internally and gives
researchers a clean mental model of which parameters they vary in a
sweep. Decomposing ``testcase`` into ``scenario`` (what we simulate)
and ``model`` (how we simulate) further disentangles geometry /
forcing / ICs from equations-of-motion / integration, enabling the
``scenario x model`` dispatch in :mod:`somax._src.cli._factories`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScenarioSpec:
    """YAML ``scenario:`` block — what we simulate.

    Args:
        name: Registry key in
            :data:`somax._src.cli.scenarios.SCENARIOS`.
        grid: Grid block — typically ``nx``, ``ny`` (plus ``Lx``,
            ``Ly`` for Cartesian scenarios or ``lon_bounds`` /
            ``lat_bounds`` for spherical ones).
        consts: Frozen basin constants — ``f0``, ``beta``, ``rho0``,
            ``g``.
        forcing: Scenario-level forcing knobs (e.g. ``wind_amplitude``
            that the scenario uses to build ``ForcingFields``).
        initial_condition: ``{"type": "...", "params": {...}}`` — IC
            shape and its parameters.
    """

    name: str
    grid: dict[str, Any] = field(default_factory=dict)
    consts: dict[str, Any] = field(default_factory=dict)
    forcing: dict[str, Any] = field(default_factory=dict)
    initial_condition: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelSpec:
    """YAML ``model:`` block — how we simulate.

    Args:
        name: Registry key in
            :data:`somax._src.cli.models_registry.MODELS`.
        stratification: Vertical structure — layer thicknesses ``H``
            and reduced gravities ``g_prime``. Empty for single-layer
            models.
        params: Differentiable parameters — viscosity, drag, etc.
    """

    name: str
    stratification: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class TimesteppingSpec:
    """Time integration window and snapshot cadence.

    All times are in seconds. ``save_interval`` controls the spacing of
    diffrax ``saveat`` snapshots written to ``snapshots.zarr``.

    Args:
        t0: Integration start time (typically ``0.0``).
        t1: Integration end time.
        dt: Initial time step. With a constant stepper this is the actual
            step.
        save_interval: Spacing between saved snapshots in seconds. If
            equal to ``t1 - t0``, only the endpoints are saved.
    """

    t0: float
    t1: float
    dt: float
    save_interval: float


@dataclass
class OutputSpec:
    """Output-side configuration.

    Args:
        write_snapshots: If ``False``, skip writing ``snapshots.zarr``.
            Useful for spinup runs that only care about the final state.
        write_metrics: If ``False``, skip writing ``metrics.json``.
            Spinup runs typically set this to ``False`` since they have
            no analysis-side metrics worth tracking.
    """

    write_snapshots: bool = True
    write_metrics: bool = True


@dataclass
class DebugSpec:
    """Debug-mode override values.

    These are merged on top of the rest of the spec when ``--debug`` is
    passed. The merge is a *deep* dict merge — only the keys present
    here override the corresponding keys in the main spec, leaving
    everything else untouched.

    Use case: each scenario/model pair picks its own "small but
    interesting" debug parameters in the config, and ``somax-sim run
    --debug`` activates them.

    Args:
        scenario: Subset of :class:`ScenarioSpec` fields to override
            (keys: ``grid``, ``consts``, ``forcing``,
            ``initial_condition``). Each top-level key is itself a dict
            that gets shallow-merged into the corresponding block.
        model: Subset of :class:`ModelSpec` fields to override (keys:
            ``stratification``, ``params``). Same shallow-per-block
            semantics as ``scenario``.
        timestepping: Subset of :class:`TimesteppingSpec` fields to
            override.
    """

    scenario: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    timestepping: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunSpec:
    """Canonical specification for a single somax simulation run.

    A ``RunSpec`` is built either by loading a YAML config (via
    :func:`from_yaml`) or by direct construction in code. The cyclopts
    CLI subcommands all build a ``RunSpec`` and dispatch to
    :func:`somax._src.cli._run.simulate`.

    Args:
        scenario: Which scenario to run and its parameters.
        model: Which model to run on that scenario and its parameters.
        timestepping: Integration window and snapshot cadence.
        output: Output-side toggles.
        debug: Debug-mode overrides (only applied if ``--debug`` is set).
        assertions: Optional opt-in preflight + postflight assertions.
            Each entry is ``{name: params_dict}`` where ``name`` is a
            registered assertion (see :mod:`somax._src.cli._assertions`)
            and ``params_dict`` is its kwargs. Empty by default — runs
            without any opt-in assertions still get the unconditional
            non-finite-state safety check from the runner.
    """

    scenario: ScenarioSpec
    model: ModelSpec
    timestepping: TimesteppingSpec
    output: OutputSpec = field(default_factory=OutputSpec)
    debug: DebugSpec = field(default_factory=DebugSpec)
    assertions: dict[str, dict[str, Any]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Run sanity checks. Raises :class:`ValueError` on failure.

        Validates the time integration window and the scenario / model
        names. Defers parameter validation to the underlying ``build``
        callables.
        """
        ts = self.timestepping
        if ts.t1 <= ts.t0:
            raise ValueError(
                f"timestepping.t1 ({ts.t1}) must be > timestepping.t0 ({ts.t0})"
            )
        if ts.dt <= 0:
            raise ValueError(f"timestepping.dt ({ts.dt}) must be > 0")
        if ts.save_interval <= 0:
            raise ValueError(
                f"timestepping.save_interval ({ts.save_interval}) must be > 0"
            )
        if ts.save_interval > (ts.t1 - ts.t0):
            raise ValueError(
                f"timestepping.save_interval ({ts.save_interval}) cannot exceed "
                f"the integration window ({ts.t1 - ts.t0})"
            )

        # Existence in the registries is checked at dispatch time
        # (importing either registry here would create a circular import).
        if not isinstance(self.scenario.name, str) or not self.scenario.name:
            raise ValueError("scenario.name must be a non-empty string")
        if not isinstance(self.model.name, str) or not self.model.name:
            raise ValueError("model.name must be a non-empty string")

    # ------------------------------------------------------------------
    # Debug merge
    # ------------------------------------------------------------------

    def with_debug_applied(self) -> RunSpec:
        """Return a new ``RunSpec`` with ``debug`` overrides merged in.

        The original instance is left untouched. The merge is a *deep*
        dict merge for the ``scenario`` and ``model`` blocks (so e.g.
        ``debug.scenario.grid = {"nx": 32}`` overrides only ``nx`` and
        leaves the other grid keys alone) and a shallow merge for
        ``timestepping``.

        Returns:
            A new :class:`RunSpec` with debug applied. Identity-equal
            to ``self`` if there are no debug entries.
        """
        if not (self.debug.scenario or self.debug.model or self.debug.timestepping):
            return self

        new_scenario = ScenarioSpec(
            name=self.scenario.name,
            grid=copy.deepcopy(self.scenario.grid),
            consts=copy.deepcopy(self.scenario.consts),
            forcing=copy.deepcopy(self.scenario.forcing),
            initial_condition=copy.deepcopy(self.scenario.initial_condition),
        )
        _merge_block_dict(new_scenario, self.debug.scenario, owner="scenario")

        new_model = ModelSpec(
            name=self.model.name,
            stratification=copy.deepcopy(self.model.stratification),
            params=copy.deepcopy(self.model.params),
        )
        _merge_block_dict(new_model, self.debug.model, owner="model")

        new_timestepping = TimesteppingSpec(
            t0=self.debug.timestepping.get("t0", self.timestepping.t0),
            t1=self.debug.timestepping.get("t1", self.timestepping.t1),
            dt=self.debug.timestepping.get("dt", self.timestepping.dt),
            save_interval=self.debug.timestepping.get(
                "save_interval", self.timestepping.save_interval
            ),
        )

        return RunSpec(
            scenario=new_scenario,
            model=new_model,
            timestepping=new_timestepping,
            output=self.output,
            debug=DebugSpec(),  # debug is consumed
            assertions=copy.deepcopy(self.assertions),
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Render as a plain nested dict (suitable for ``yaml.safe_dump``)."""
        return {
            "scenario": {
                "name": self.scenario.name,
                "grid": copy.deepcopy(self.scenario.grid),
                "consts": copy.deepcopy(self.scenario.consts),
                "forcing": copy.deepcopy(self.scenario.forcing),
                "initial_condition": copy.deepcopy(self.scenario.initial_condition),
            },
            "model": {
                "name": self.model.name,
                "stratification": copy.deepcopy(self.model.stratification),
                "params": copy.deepcopy(self.model.params),
            },
            "timestepping": {
                "t0": self.timestepping.t0,
                "t1": self.timestepping.t1,
                "dt": self.timestepping.dt,
                "save_interval": self.timestepping.save_interval,
            },
            "output": {
                "write_snapshots": self.output.write_snapshots,
                "write_metrics": self.output.write_metrics,
            },
            "debug": {
                "scenario": copy.deepcopy(self.debug.scenario),
                "model": copy.deepcopy(self.debug.model),
                "timestepping": copy.deepcopy(self.debug.timestepping),
            },
            "assertions": copy.deepcopy(self.assertions),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunSpec:
        """Construct from a plain nested dict (loaded YAML).

        Missing optional blocks (``output``, ``debug``) get defaults.
        Extra unknown keys are silently ignored at this layer; the
        validator catches problems with required fields.
        """
        try:
            scenario_data = data["scenario"]
            model_data = data["model"]
            ts_data = data["timestepping"]
        except KeyError as exc:
            raise ValueError(
                f"RunSpec config missing required block: {exc.args[0]}"
            ) from exc

        scenario = ScenarioSpec(
            name=scenario_data["name"],
            grid=dict(scenario_data.get("grid", {})),
            consts=dict(scenario_data.get("consts", {})),
            forcing=dict(scenario_data.get("forcing", {})),
            initial_condition=dict(scenario_data.get("initial_condition", {})),
        )
        model = ModelSpec(
            name=model_data["name"],
            stratification=dict(model_data.get("stratification", {})),
            params=dict(model_data.get("params", {})),
        )
        timestepping = TimesteppingSpec(
            t0=float(ts_data["t0"]),
            t1=float(ts_data["t1"]),
            dt=float(ts_data["dt"]),
            save_interval=float(ts_data["save_interval"]),
        )
        out_data = data.get("output", {})
        output = OutputSpec(
            write_snapshots=bool(out_data.get("write_snapshots", True)),
            write_metrics=bool(out_data.get("write_metrics", True)),
        )
        dbg_data = data.get("debug", {})
        debug = DebugSpec(
            scenario=dict(dbg_data.get("scenario", {})),
            model=dict(dbg_data.get("model", {})),
            timestepping=dict(dbg_data.get("timestepping", {})),
        )
        assertions_data = data.get("assertions", {}) or {}
        # Each entry: name -> params (dict) | None.
        assertions: dict[str, dict[str, Any]] = {
            str(name): dict(params or {}) for name, params in assertions_data.items()
        }
        return cls(
            scenario=scenario,
            model=model,
            timestepping=timestepping,
            output=output,
            debug=debug,
            assertions=assertions,
        )


def _merge_block_dict(
    target: ScenarioSpec | ModelSpec,
    override: dict[str, Any],
    *,
    owner: str,
) -> None:
    """Apply a ``debug.<owner>`` block-shaped override dict onto ``target``.

    Each top-level key in ``override`` must name a dict-valued field on
    ``target``; its value must itself be a dict, which is
    shallow-merged onto the existing block (only the keys present in
    the override are touched).
    """
    spec_cls = type(target).__name__
    for block_name, block_override in override.items():
        if not hasattr(target, block_name):
            raise ValueError(
                f"debug.{owner}.{block_name!r} does not match any {spec_cls} field"
            )
        existing = getattr(target, block_name)
        if not isinstance(existing, dict) or not isinstance(block_override, dict):
            raise ValueError(
                f"debug.{owner}.{block_name!r} merge requires both sides "
                f"to be dicts; got {type(existing).__name__} and "
                f"{type(block_override).__name__}"
            )
        existing.update(block_override)


def load_yaml(path: str) -> RunSpec:
    """Load a YAML config file into a :class:`RunSpec`.

    Args:
        path: Path to a YAML file produced by the config authoring layer
            (or handwritten).

    Returns:
        Validated :class:`RunSpec`.
    """
    import yaml

    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(
            f"Config file {path!r} did not parse as a top-level mapping; "
            f"got {type(data).__name__}"
        )
    spec = RunSpec.from_dict(data)
    spec.validate()
    return spec


def dump_yaml(spec: RunSpec, path: str) -> None:
    """Persist a :class:`RunSpec` as YAML."""
    import yaml

    with open(path, "w") as f:
        yaml.safe_dump(spec.to_dict(), f, sort_keys=False, default_flow_style=False)
