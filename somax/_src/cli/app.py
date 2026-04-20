"""Cyclopts CLI for somax simulations.

Installed as the ``somax-sim`` console script via
``pyproject.toml``'s ``[project.scripts]`` table.

The CLI is intentionally a thin shell over :mod:`somax._src.cli._run`:
each subcommand parses its arguments, builds a :class:`RunSpec` (either
from a YAML file or from CLI overrides), and delegates to a single
``simulate`` / ``spinup`` / ``restart`` function. There are also
discovery commands (``list-testcases`` / ``list-models`` /
``show-config``) for ergonomic introspection.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter
from loguru import logger

from somax._src.cli import _factories, _run
from somax._src.cli.spec import RunSpec, load_yaml


# ----------------------------------------------------------------------
# Logging setup — loguru with a compact, colored sink
# ----------------------------------------------------------------------


def _configure_logging(verbose: bool = False) -> None:
    """Reset loguru's sinks to a single colored stderr sink for the CLI.

    Loguru ships with a default stderr sink that is fine for libraries
    but a little verbose for an interactive CLI; we replace it with a
    short, colored format that matches what users expect from a
    well-behaved tool. ``--verbose`` raises the level from INFO to DEBUG.
    """
    level = "DEBUG" if verbose else "INFO"
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:HH:mm:ss}</green> "
            "<level>{level: <7}</level> "
            "<cyan>somax-sim</cyan> | <level>{message}</level>"
        ),
        colorize=True,
    )


# ----------------------------------------------------------------------
# App
# ----------------------------------------------------------------------


app = App(
    name="somax-sim",
    help="somax simulation runner — fresh runs, spinups, restarts.",
    version_flags=["--version", "-V"],
)


# ----------------------------------------------------------------------
# run — fresh simulation from initial conditions
# ----------------------------------------------------------------------


@app.command
def run(
    *,
    config: Annotated[Path, Parameter(help="Path to a YAML run-spec file.")],
    output_dir: Annotated[
        Path,
        Parameter(
            help="Directory for snapshots.zarr / final_state.zarr / metrics.json."
        ),
    ],
    debug: Annotated[
        bool,
        Parameter(help="Apply the cfg.debug overrides (smaller grid, shorter run)."),
    ] = False,
    diagnostics_per_save: Annotated[
        int,
        Parameter(
            help=(
                "Number of diagnostic sub-chunks logged per save interval. "
                "1 (default) = one log line per snapshot. Higher values give "
                "finer-grained monitoring in <output_dir>/run.log without "
                "changing the snapshot cadence."
            ),
        ),
    ] = 1,
    verbose: Annotated[
        bool,
        Parameter(
            help=(
                "Enable DEBUG-level logging on stderr (also tees the per-chunk "
                "diagnostics from run.log to the terminal)."
            ),
        ),
    ] = False,
    checkpoint_every_n_chunks: Annotated[
        int,
        Parameter(
            help=(
                "Crash-recovery cadence (#70). When > 0, writes the current "
                "state to <output_dir>/checkpoint.zarr after every Nth "
                "snapshot-aligned chunk. The new run can be resumed via "
                "`somax-sim restart --from <output_dir>/checkpoint.zarr` — "
                "integration picks up at the checkpoint's sim-time. 0 = off."
            ),
        ),
    ] = 0,
) -> None:
    """Run a fresh simulation from factory-built initial conditions.

    Reads ``config`` (YAML), optionally merges the ``debug`` block, then
    integrates the model and writes ``snapshots.zarr``, ``final_state.zarr``,
    ``metrics.json``, and ``run.log`` under ``output_dir``.
    """
    _configure_logging(verbose)
    spec = _load_and_prepare(config, debug=debug)
    _run.simulate(
        spec,
        output_dir,
        diagnostics_per_save=diagnostics_per_save,
        checkpoint_every_n_chunks=checkpoint_every_n_chunks,
    )


# ----------------------------------------------------------------------
# spinup — endpoint-only run that produces a restart artifact
# ----------------------------------------------------------------------


@app.command
def spinup(
    *,
    config: Annotated[
        Path,
        Parameter(help="Path to a YAML run-spec file (typically a spinup config)."),
    ],
    output_dir: Annotated[
        Path, Parameter(help="Directory for the resulting final_state.zarr.")
    ],
    debug: Annotated[
        bool,
        Parameter(help="Apply the cfg.debug overrides (smaller grid, shorter run)."),
    ] = False,
    diagnostics_per_save: Annotated[
        int,
        Parameter(
            help=(
                "Number of diagnostic sub-chunks logged per save interval "
                "(default 1). Higher values give finer monitoring."
            ),
        ),
    ] = 1,
    verbose: Annotated[
        bool,
        Parameter(
            help=(
                "Enable DEBUG-level logging on stderr (also tees per-chunk "
                "diagnostics to the terminal)."
            ),
        ),
    ] = False,
    checkpoint_every_n_chunks: Annotated[
        int,
        Parameter(
            help=(
                "Crash-recovery cadence (#70). ``> 0`` writes rolling "
                "<output_dir>/checkpoint.zarr during the spinup. 0 = off."
            ),
        ),
    ] = 0,
) -> None:
    """Run a spinup integration. Saves only the endpoint state.

    Spinup runs are intended to produce a ``final_state.zarr`` artifact
    that downstream production runs use as their initial condition. No
    snapshots, no metrics — just the equilibrium state.

    Pass ``--debug`` to smoke-test the spinup → restart chain quickly;
    DVC stages should not pass ``--debug``. A structured ``run.log`` is
    always written under ``output_dir``.
    """
    _configure_logging(verbose)
    spec = _load_and_prepare(config, debug=debug)
    _run.spinup(
        spec,
        output_dir,
        diagnostics_per_save=diagnostics_per_save,
        checkpoint_every_n_chunks=checkpoint_every_n_chunks,
    )


# ----------------------------------------------------------------------
# restart — resume from a saved state
# ----------------------------------------------------------------------


@app.command
def restart(
    *,
    config: Annotated[
        Path,
        Parameter(
            help="YAML run-spec for the production run (model is built from here)."
        ),
    ],
    from_: Annotated[
        Path,
        Parameter(
            name=["--from", "-f"],
            help="Path to a zarr store containing the saved state to resume from.",
        ),
    ],
    output_dir: Annotated[
        Path,
        Parameter(
            help="Directory for snapshots.zarr / final_state.zarr / metrics.json."
        ),
    ],
    debug: Annotated[bool, Parameter(help="Apply the cfg.debug overrides.")] = False,
    diagnostics_per_save: Annotated[
        int,
        Parameter(
            help=(
                "Number of diagnostic sub-chunks logged per save interval "
                "(default 1). Higher values give finer monitoring."
            ),
        ),
    ] = 1,
    verbose: Annotated[
        bool,
        Parameter(
            help=(
                "Enable DEBUG-level logging on stderr (also tees per-chunk "
                "diagnostics to the terminal)."
            ),
        ),
    ] = False,
    checkpoint_every_n_chunks: Annotated[
        int,
        Parameter(
            help=(
                "Crash-recovery cadence (#70) for the *resumed* run. When "
                "> 0, writes the current state to "
                "<output_dir>/checkpoint.zarr after every Nth "
                "snapshot-aligned chunk. 0 = off."
            ),
        ),
    ] = 0,
) -> None:
    """Resume a simulation from a previously saved state.

    The model is built from ``config`` (so you can change viscosity,
    forcing, etc. between runs); only the *state* is loaded from the
    ``--from`` zarr store.

    If ``--from`` points at a crash-recovery checkpoint (a zarr store
    with a ``somax_sim_t`` attr), the integration window is
    automatically rebased so the run picks up where the checkpoint
    left off. Legacy ``final_state.zarr`` stores (no attr) are
    honoured as-is.
    """
    _configure_logging(verbose)
    spec = _load_and_prepare(config, debug=debug)
    _run.restart(
        spec,
        output_dir,
        restart_from=from_,
        diagnostics_per_save=diagnostics_per_save,
        checkpoint_every_n_chunks=checkpoint_every_n_chunks,
    )


# ----------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------


@app.command(name="list-testcases")
def list_testcases() -> None:
    """List all registered test cases (legacy ``testcase:`` schema)."""
    print("Registered test cases:")
    for name in _factories.list_test_cases():
        print(f"  - {name}")


_PHASE2_NOTE = (
    "Note: Phase-2 scaffold (#76). Every entry below is a stub — none are "
    "runnable via `somax-sim run` yet (the runner still dispatches on "
    "`testcase.name`; Phase 3 cuts over). For currently-runnable entries, "
    "see `somax-sim list-testcases`."
)


@app.command(name="list-scenarios")
def list_scenarios() -> None:
    """List scenarios in the Phase-2 registry (``scenario:`` schema)."""
    from somax._src.cli.scenarios import SCENARIOS, list_scenarios as _list

    print(_PHASE2_NOTE)
    print()
    print("Registered scenarios:")
    for name in _list():
        entry = SCENARIOS[name]
        print(f"  - {name}  (geometry: {entry.geometry_kind})  [stub]")


@app.command(name="list-models")
def list_models() -> None:
    """List models in the Phase-2 registry (``model:`` schema)."""
    from somax._src.cli.models_registry import MODELS, list_models as _list

    print(_PHASE2_NOTE)
    print()
    print("Registered models:")
    for name in _list():
        entry = MODELS[name]
        layers = entry.layers if entry.layers == "multi" else f"{entry.layers}-layer"
        masks = "masks=yes" if entry.supports.masks else "masks=no"
        print(
            f"  - {name}  ({entry.family}, {layers}, {entry.coordinates}, {masks})"
            "  [stub]"
        )


@app.command(name="list-model-classes")
def list_model_classes() -> None:
    """List the importable model classes in ``somax.models`` (library discovery).

    Complements ``list-models`` (which lists Phase-2 registry entries by name
    for the upcoming ``scenario:`` + ``model:`` YAML schema). Use this when
    you want to see which ``somax.models.*`` classes are actually importable
    today from Python.
    """
    from somax import models

    names = sorted(
        n
        for n in dir(models)
        if not n.startswith("_") and isinstance(getattr(models, n), type)
    )
    print("Model classes in `somax.models`:")
    for n in names:
        print(f"  - {n}")


@app.command(name="list-pairs")
def list_pairs() -> None:
    """Print the scenario x model compatibility matrix."""
    from somax._src.cli._compatibility import compatibility_matrix
    from somax._src.cli.models_registry import list_models as _list_models
    from somax._src.cli.scenarios import list_scenarios as _list_scenarios

    matrix = compatibility_matrix()
    models = _list_models()
    scenarios = _list_scenarios()

    name_col = max(len("scenario"), *(len(s) for s in scenarios))
    widths = [max(len(m), 3) for m in models]
    header = "  ".join(
        [
            f"{'scenario':<{name_col}}",
            *(f"{m:<{w}}" for m, w in zip(models, widths, strict=True)),
        ]
    )
    print(header)
    print("-" * len(header))
    for s in scenarios:
        row = [f"{s:<{name_col}}"]
        for m, w in zip(models, widths, strict=True):
            mark = "x" if matrix[s][m] else "."
            row.append(f"{mark:<{w}}")
        print("  ".join(row))
    print("\nLegend: x = compatible, . = incompatible")


@app.command(name="show-config")
def show_config(
    path: Annotated[Path, Parameter(help="Path to a YAML config.")],
) -> None:
    """Pretty-print a YAML config after loading + validation.

    Useful for verifying that your authored YAML parses cleanly into a
    :class:`RunSpec` before handing it to ``run`` / ``spinup`` / ``restart``.
    """
    spec = load_yaml(str(path))
    import yaml

    print(f"# Resolved RunSpec from {path}")
    print(yaml.safe_dump(spec.to_dict(), sort_keys=False, default_flow_style=False))


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _load_and_prepare(config: Path, *, debug: bool) -> RunSpec:
    """Load YAML → RunSpec → debug-merge → validate."""
    spec = load_yaml(str(config))
    if debug:
        spec = spec.with_debug_applied()
        spec.validate()
    return spec


# ----------------------------------------------------------------------
# Console-script entry
# ----------------------------------------------------------------------


def main() -> None:  # pragma: no cover - thin wrapper
    """Console script entry point. Delegates to the cyclopts App."""
    sys.exit(app() or 0)


if __name__ == "__main__":  # pragma: no cover
    main()
