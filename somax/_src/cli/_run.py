"""Core run / spinup / restart implementations for somax-sim.

These are the functions that the cyclopts CLI subcommands delegate to.
They take a validated :class:`RunSpec` and an output directory, perform
the simulation, and write the persistence artifacts (zarr snapshots,
final state, metrics JSON, resolved config).

This module is the *runtime* equivalent of the design doc's "L1
RunSpec dataclass" plus the result-writing concerns. The CLI shell in
``app.py`` is the user-facing façade.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from pathlib import Path
from typing import Any

import diffrax as dfx
import jax.numpy as jnp
import numpy as np

from somax import io
from somax._src.cli._factories import get_adapter
from somax._src.cli.spec import RunSpec, dump_yaml


logger = logging.getLogger("somax-sim")


# ----------------------------------------------------------------------
# Public results object
# ----------------------------------------------------------------------


@dataclasses.dataclass
class SimulationResult:
    """In-memory summary of a completed simulation.

    The persistent artifacts (snapshots.zarr, final_state.zarr,
    metrics.json, resolved.yaml) are written to ``output_dir`` by the
    run functions; this object is returned to callers (and to test
    code) for inspection.

    Args:
        output_dir: Directory containing the written artifacts.
        snapshots_path: Path to the zarr snapshots store, or ``None`` if
            snapshot writing was disabled.
        final_state_path: Path to the zarr final-state store.
        metrics_path: Path to the metrics JSON file, or ``None`` if
            metrics writing was disabled.
        wallclock_seconds: Total simulation wallclock time.
        n_steps: Best-effort step count from the diffrax stats.
    """

    output_dir: Path
    snapshots_path: Path | None
    final_state_path: Path
    metrics_path: Path | None
    wallclock_seconds: float
    n_steps: int | None


# ----------------------------------------------------------------------
# saveat construction
# ----------------------------------------------------------------------


def _build_saveat(
    spec: RunSpec, *, only_final: bool = False
) -> tuple[dfx.SaveAt, jnp.ndarray]:
    """Build a diffrax SaveAt and the matching ts array.

    Args:
        spec: Validated run spec.
        only_final: If ``True``, save only the endpoint (used by spinup).

    Returns:
        Tuple ``(saveat, ts)`` where ``ts`` is the array of save times
        in seconds.
    """
    ts_data = spec.timestepping
    if only_final:
        ts = jnp.asarray([ts_data.t1])
    else:
        # Inclusive endpoint, evenly spaced.
        n_save = round((ts_data.t1 - ts_data.t0) / ts_data.save_interval) + 1
        n_save = max(int(n_save), 2)
        ts = jnp.linspace(ts_data.t0, ts_data.t1, n_save)
    return dfx.SaveAt(ts=ts), ts


# ----------------------------------------------------------------------
# Diagnostic scalars → JSON
# ----------------------------------------------------------------------


def _flatten_diagnostics(diag: Any) -> dict[str, Any]:
    """Extract scalar fields from a Diagnostics pytree as a JSON-friendly dict.

    Multi-element arrays are summarized via mean / max / min so they
    fit in a flat metrics file. Per-layer scalars are flattened to
    ``<field>_layer_<i>`` keys.
    """
    out: dict[str, Any] = {}
    if diag is None:
        return out
    if not dataclasses.is_dataclass(diag):
        return out
    for field in dataclasses.fields(diag):
        value = getattr(diag, field.name)
        if value is None:
            continue
        try:
            arr = np.asarray(value)
        except (TypeError, ValueError):
            continue
        if arr.ndim == 0:
            out[field.name] = float(arr)
        elif arr.ndim == 1 and arr.size <= 16:
            for i, scalar in enumerate(arr.tolist()):
                out[f"{field.name}_layer_{i}"] = float(scalar)
        else:
            out[f"{field.name}_mean"] = float(arr.mean())
            out[f"{field.name}_max"] = float(arr.max())
            out[f"{field.name}_min"] = float(arr.min())
    return out


# ----------------------------------------------------------------------
# Persistence helpers
# ----------------------------------------------------------------------


def _ensure_clean_dir(path: Path) -> None:
    """Make ``path`` exist as a directory. Existing contents are kept."""
    path.mkdir(parents=True, exist_ok=True)


def _attrs_for(spec: RunSpec, *, mode: str) -> dict[str, str]:
    """Standard Dataset attrs that document where this artifact came from."""
    return {
        "somax_sim_mode": mode,
        "testcase_name": spec.testcase.name,
        "t0": str(spec.timestepping.t0),
        "t1": str(spec.timestepping.t1),
        "dt": str(spec.timestepping.dt),
        "save_interval": str(spec.timestepping.save_interval),
    }


# ----------------------------------------------------------------------
# Inner integrate — shared between run/spinup/restart
# ----------------------------------------------------------------------


def _integrate_and_write(
    spec: RunSpec,
    output_dir: Path,
    *,
    mode: str,
    initial_state: Any | None,
) -> SimulationResult:
    """Build the model, integrate, write artifacts, return summary.

    Args:
        spec: Validated, debug-merged run spec.
        output_dir: Where to write artifacts.
        mode: ``"run"``, ``"spinup"``, or ``"restart"`` — used in attrs
            and to decide what to save.
        initial_state: If non-None, override the factory-built initial
            state with this one (used by ``restart``).
    """
    output_dir = Path(output_dir)
    _ensure_clean_dir(output_dir)

    logger.info("somax-sim mode=%s testcase=%s", mode, spec.testcase.name)
    logger.info("output_dir=%s", output_dir)

    # 1. Build model and initial state via the adapter registry.
    adapter = get_adapter(spec.testcase.name)
    model, factory_state0 = adapter(
        grid=spec.testcase.grid,
        consts=spec.testcase.consts,
        stratification=spec.testcase.stratification,
        params=spec.testcase.params,
    )
    state0 = initial_state if initial_state is not None else factory_state0

    # 2. Build saveat. Spinup only saves the endpoint.
    only_final = mode == "spinup"
    saveat, ts = _build_saveat(spec, only_final=only_final)

    # 3. Integrate.
    logger.info(
        "integrating: t0=%s t1=%s dt=%s saveat_n=%d",
        spec.timestepping.t0,
        spec.timestepping.t1,
        spec.timestepping.dt,
        ts.shape[0],
    )
    # diffrax defaults max_steps to 4096 which is far too small for any
    # multi-day simulation. Compute the actual minimum + a 20% buffer.
    window = spec.timestepping.t1 - spec.timestepping.t0
    expected_steps = int(window / spec.timestepping.dt)
    max_steps = max(16384, int(expected_steps * 1.2))

    t_start = time.perf_counter()
    sol = model.integrate(
        state0,
        spec.timestepping.t0,
        spec.timestepping.t1,
        spec.timestepping.dt,
        saveat=saveat,
        max_steps=max_steps,
    )
    wallclock = time.perf_counter() - t_start
    logger.info("integration finished in %.2f s", wallclock)

    # 4. Write snapshots.zarr (skipped for spinup or if disabled).
    snapshots_path: Path | None = None
    if spec.output.write_snapshots and not only_final:
        snapshots_path = output_dir / "snapshots.zarr"
        snapshots_ds = io.snapshots_to_dataset(
            sol.ys, np.asarray(ts), attrs=_attrs_for(spec, mode=mode)
        )
        io.save_dataset(snapshots_ds, snapshots_path, mode="w")
        logger.info("wrote %s", snapshots_path)

    # 5. Always write final_state.zarr — this is the restart artifact.
    final_state = _extract_final_state(sol.ys, type(state0))
    final_state_path = output_dir / "final_state.zarr"
    final_ds = io.state_to_dataset(
        final_state,
        time=float(spec.timestepping.t1),
        attrs=_attrs_for(spec, mode=mode),
    )
    io.save_dataset(final_ds, final_state_path, mode="w")
    logger.info("wrote %s", final_state_path)

    # 6. Diagnostics → metrics.json.
    metrics_path: Path | None = None
    if spec.output.write_metrics and not only_final:
        try:
            diagnostics = model.diagnose(final_state)
        except Exception as exc:
            logger.warning("model.diagnose failed: %s", exc)
            diagnostics = None
        metrics = _flatten_diagnostics(diagnostics)
        metrics["wallclock_seconds"] = wallclock
        metrics["n_steps"] = _maybe_step_count(sol)
        metrics["t0"] = spec.timestepping.t0
        metrics["t1"] = spec.timestepping.t1
        metrics["save_interval"] = spec.timestepping.save_interval
        metrics["mode"] = mode
        metrics_path = output_dir / "metrics.json"
        with metrics_path.open("w") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
        logger.info("wrote %s", metrics_path)

    # 7. Resolved config dump (debugging aid; not git/dvc tracked).
    resolved_path = output_dir / "resolved.yaml"
    dump_yaml(spec, str(resolved_path))
    logger.info("wrote %s", resolved_path)

    return SimulationResult(
        output_dir=output_dir,
        snapshots_path=snapshots_path,
        final_state_path=final_state_path,
        metrics_path=metrics_path,
        wallclock_seconds=wallclock,
        n_steps=_maybe_step_count(sol),
    )


def _extract_final_state(stacked: Any, state_class: type) -> Any:
    """Pull the last time slice out of a diffrax-stacked state pytree."""
    kwargs = {}
    for field in dataclasses.fields(state_class):
        leaf = getattr(stacked, field.name)
        kwargs[field.name] = leaf[-1]
    return state_class(**kwargs)


def _maybe_step_count(sol: dfx.Solution) -> int | None:
    """Best-effort recovery of the diffrax step count for metrics."""
    stats = getattr(sol, "stats", None)
    if not stats:
        return None
    n = stats.get("num_steps")
    if n is None:
        return None
    try:
        return int(n)
    except (TypeError, ValueError):
        return None


# ----------------------------------------------------------------------
# Public entry points — one per CLI subcommand
# ----------------------------------------------------------------------


def simulate(spec: RunSpec, output_dir: str | Path) -> SimulationResult:
    """Run a fresh simulation from factory-built initial conditions.

    Args:
        spec: Run specification. Should already be debug-merged and
            validated by the caller.
        output_dir: Directory for written artifacts.
    """
    return _integrate_and_write(spec, Path(output_dir), mode="run", initial_state=None)


def spinup(spec: RunSpec, output_dir: str | Path) -> SimulationResult:
    """Run a spinup integration. Saves only the endpoint.

    Spinup runs are intended to produce a ``final_state.zarr`` artifact
    that downstream production runs use as their initial condition.
    No snapshots, no metrics — just the equilibrium state.
    """
    return _integrate_and_write(
        spec, Path(output_dir), mode="spinup", initial_state=None
    )


def restart(
    spec: RunSpec,
    output_dir: str | Path,
    *,
    restart_from: str | Path,
) -> SimulationResult:
    """Resume a simulation from a previously saved state.

    The model is built from ``spec`` (so you can change viscosity,
    forcing, etc. between runs); only the *state* is loaded from the
    restart file.

    Args:
        spec: Run specification (model construction comes from here).
        output_dir: Directory for written artifacts.
        restart_from: Path to a zarr store containing a saved state
            (typically a previous run's ``final_state.zarr``).
    """
    restart_path = Path(restart_from)
    logger.info("restart loading state from %s", restart_path)
    ds = io.load_dataset(restart_path)
    state0 = io.dataset_to_state(ds)
    return _integrate_and_write(
        spec, Path(output_dir), mode="restart", initial_state=state0
    )
