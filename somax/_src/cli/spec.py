"""Canonical run specification for somax-sim.

This module defines the :class:`RunSpec` dataclass and its components.
``RunSpec`` is the internal canonical representation of a single
simulation run; everything else in the CLI funnels into it.

The structured shape (``grid`` / ``consts`` / ``stratification`` /
``params`` blocks under ``testcase``) is the v0.1 decision per Q-E in
the design doc — it mirrors how somax separates ``Params`` (differentiable)
from ``PhysConsts`` (frozen) internally and gives researchers a clean
mental model of which parameters they would vary in a sweep.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TestCaseSpec:
    """Identification + structured kwargs for a test case factory.

    Args:
        name: Registry key in :data:`somax._src.cli._factories.TEST_CASES`.
        grid: Grid block — typically ``nx``, ``ny``, ``Lx``, ``Ly``.
        consts: Frozen physical constants — ``f0``, ``beta``, layer count.
        stratification: Vertical structure — layer thicknesses ``H`` and
            reduced gravities ``g_prime``. May be empty for single-layer
            models.
        params: Differentiable parameters — viscosity, drag, forcing
            amplitudes.
    """

    name: str
    grid: dict[str, Any] = field(default_factory=dict)
    consts: dict[str, Any] = field(default_factory=dict)
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

    Use case: each test case picks its own "small but interesting"
    debug parameters in the config, and ``somax-sim run --debug``
    activates them.

    Args:
        testcase: Subset of :class:`TestCaseSpec` fields to override.
            Each top-level key (``grid``, ``consts``, ``params``, etc.)
            is itself a dict that gets shallow-merged into the
            corresponding block.
        timestepping: Subset of :class:`TimesteppingSpec` fields to
            override.
    """

    testcase: dict[str, Any] = field(default_factory=dict)
    timestepping: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunSpec:
    """Canonical specification for a single somax simulation run.

    A ``RunSpec`` is built either by loading a YAML config (via
    :func:`from_yaml`) or by direct construction in code. The cyclopts
    CLI subcommands all build a ``RunSpec`` and dispatch to
    :func:`somax._src.cli._run.simulate`.

    Args:
        testcase: Which test case to run and its parameters.
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

    testcase: TestCaseSpec
    timestepping: TimesteppingSpec
    output: OutputSpec = field(default_factory=OutputSpec)
    debug: DebugSpec = field(default_factory=DebugSpec)
    assertions: dict[str, dict[str, Any]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Run sanity checks. Raises :class:`ValueError` on failure.

        Validates the time integration window and the testcase name.
        Defers parameter validation to the underlying factory.
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

        # Test-case existence is checked at dispatch time (importing the
        # registry here would create a circular import).
        if not isinstance(self.testcase.name, str) or not self.testcase.name:
            raise ValueError("testcase.name must be a non-empty string")

    # ------------------------------------------------------------------
    # Debug merge
    # ------------------------------------------------------------------

    def with_debug_applied(self) -> RunSpec:
        """Return a new ``RunSpec`` with ``debug`` overrides merged in.

        The original instance is left untouched. The merge is a *deep*
        dict merge for the ``testcase`` blocks (so e.g.
        ``debug.testcase.grid = {"nx": 32}`` overrides only ``nx`` and
        leaves the other grid keys alone) and a shallow merge for
        ``timestepping``.

        Returns:
            A new :class:`RunSpec` with debug applied. Identity-equal
            to ``self`` if there are no debug entries.
        """
        if not self.debug.testcase and not self.debug.timestepping:
            return self

        # Deep-copy mutable nested dicts before mutating.
        new_testcase = TestCaseSpec(
            name=self.testcase.name,
            grid=copy.deepcopy(self.testcase.grid),
            consts=copy.deepcopy(self.testcase.consts),
            stratification=copy.deepcopy(self.testcase.stratification),
            params=copy.deepcopy(self.testcase.params),
        )
        for block_name, override in self.debug.testcase.items():
            if not hasattr(new_testcase, block_name):
                raise ValueError(
                    f"debug.testcase.{block_name!r} does not match any "
                    f"TestCaseSpec field"
                )
            target = getattr(new_testcase, block_name)
            if not isinstance(target, dict) or not isinstance(override, dict):
                raise ValueError(
                    f"debug.testcase.{block_name!r} merge requires both sides "
                    f"to be dicts; got {type(target).__name__} and "
                    f"{type(override).__name__}"
                )
            target.update(override)

        new_timestepping = TimesteppingSpec(
            t0=self.debug.timestepping.get("t0", self.timestepping.t0),
            t1=self.debug.timestepping.get("t1", self.timestepping.t1),
            dt=self.debug.timestepping.get("dt", self.timestepping.dt),
            save_interval=self.debug.timestepping.get(
                "save_interval", self.timestepping.save_interval
            ),
        )

        return RunSpec(
            testcase=new_testcase,
            timestepping=new_timestepping,
            output=self.output,
            debug=DebugSpec(),  # debug is consumed
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Render as a plain nested dict (suitable for ``yaml.safe_dump``)."""
        return {
            "testcase": {
                "name": self.testcase.name,
                "grid": copy.deepcopy(self.testcase.grid),
                "consts": copy.deepcopy(self.testcase.consts),
                "stratification": copy.deepcopy(self.testcase.stratification),
                "params": copy.deepcopy(self.testcase.params),
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
                "testcase": copy.deepcopy(self.debug.testcase),
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
            tc_data = data["testcase"]
            ts_data = data["timestepping"]
        except KeyError as exc:
            raise ValueError(
                f"RunSpec config missing required block: {exc.args[0]}"
            ) from exc

        testcase = TestCaseSpec(
            name=tc_data["name"],
            grid=dict(tc_data.get("grid", {})),
            consts=dict(tc_data.get("consts", {})),
            stratification=dict(tc_data.get("stratification", {})),
            params=dict(tc_data.get("params", {})),
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
            testcase=dict(dbg_data.get("testcase", {})),
            timestepping=dict(dbg_data.get("timestepping", {})),
        )
        assertions_data = data.get("assertions", {}) or {}
        # Each entry: name -> params (dict) | None.
        assertions: dict[str, dict[str, Any]] = {
            str(name): dict(params or {}) for name, params in assertions_data.items()
        }
        return cls(
            testcase=testcase,
            timestepping=timestepping,
            output=output,
            debug=debug,
            assertions=assertions,
        )


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
