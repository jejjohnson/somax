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
) -> None:
    """Run a fresh simulation from factory-built initial conditions.

    Reads ``config`` (YAML), optionally merges the ``debug`` block, then
    integrates the model and writes ``snapshots.zarr``, ``final_state.zarr``,
    ``metrics.json``, and ``run.log`` under ``output_dir``.
    """
    _configure_logging(verbose)
    spec = _load_and_prepare(config, debug=debug)
    _run.simulate(spec, output_dir, diagnostics_per_save=diagnostics_per_save)


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
    _run.spinup(spec, output_dir, diagnostics_per_save=diagnostics_per_save)


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
) -> None:
    """Resume a simulation from a previously saved state.

    The model is built from ``config`` (so you can change viscosity,
    forcing, etc. between runs); only the *state* is loaded from the
    ``--from`` zarr store.
    """
    _configure_logging(verbose)
    spec = _load_and_prepare(config, debug=debug)
    _run.restart(
        spec,
        output_dir,
        restart_from=from_,
        diagnostics_per_save=diagnostics_per_save,
    )


# ----------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------


@app.command(name="list-testcases")
def list_testcases() -> None:
    """List all registered test cases."""
    print("Registered test cases:")
    for name in _factories.list_test_cases():
        print(f"  - {name}")


@app.command(name="list-models")
def list_models() -> None:
    """List the model classes available in somax."""
    from somax import models

    names = sorted(
        n
        for n in dir(models)
        if not n.startswith("_") and isinstance(getattr(models, n), type)
    )
    print("Available model classes:")
    for n in names:
        print(f"  - {n}")


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
