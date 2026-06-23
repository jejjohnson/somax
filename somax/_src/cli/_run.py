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
import time
from pathlib import Path
from typing import Any

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp
import numpy as np
from loguru import logger
from pipekit import StatefulOperator
from pipekit_cycle import Cycle

from somax import io
from somax._src.cli import _assertions
from somax._src.cli._factories import build
from somax._src.cli._progress import RunLogContext, start_run_log, stop_run_log
from somax._src.cli._units import (
    format_field_stats,
    format_time_seconds,
    format_wallclock,
)
from somax._src.cli.spec import RunSpec, dump_yaml
from somax._src.eval.metrics import compute_eval_metrics
from somax._src.monitor import ChunkInfo, default_monitors
from somax._src.monitor.protocol import Monitor


class IntegrationDivergedError(RuntimeError):
    """Raised when an integration produces non-finite values.

    The most common cause is a CFL violation: time step too large for
    the grid spacing and the fastest wave speed in the model. The error
    message lists which State fields contained NaN/Inf so you can
    diagnose where the blow-up started.
    """


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


def _build_save_times(spec: RunSpec, *, only_final: bool = False) -> jnp.ndarray:
    """Build the array of times at which snapshots should be saved.

    Always returns at least two points: ``[t0, ..., t1]``. For spinup
    runs (``only_final=True``) the array is ``[t0, t1]`` since we
    only need the endpoint state, but the chunked-integration path
    still needs the t0 anchor.

    Cadence semantics: snapshots are placed at exactly ``save_interval``
    apart, starting from ``t0`` — i.e. ``[t0, t0+SI, t0+2*SI, ...]`` —
    and ``t1`` is appended as the final snapshot if it isn't already a
    multiple of ``save_interval``. The last interval may therefore be
    shorter than ``save_interval`` (this is the desired behavior for
    cadences like "monthly snapshots over a year": you get 12 monthly
    snapshots plus a final snapshot at year's end). Earlier versions
    used ``linspace`` which silently rescaled the spacing to fit
    ``round((t1-t0)/SI) + 1`` evenly-spaced points; that hid mismatches.

    Args:
        spec: Validated run spec.
        only_final: If ``True``, snapshot only the endpoint (used by
            spinup).

    Returns:
        Array of save times in seconds, length >= 2.
    """
    ts_data = spec.timestepping
    if only_final:
        return jnp.asarray([ts_data.t0, ts_data.t1])

    # Build [t0, t0+SI, t0+2*SI, ...] strictly less than t1, then append
    # t1 as the closing snapshot. Use float64 throughout for predictable
    # rounding when dt and save_interval span many orders of magnitude.
    t0 = float(ts_data.t0)
    t1 = float(ts_data.t1)
    si = float(ts_data.save_interval)
    n_full = int((t1 - t0) // si)  # number of complete save_interval steps
    interior = np.asarray([t0 + i * si for i in range(n_full + 1)], dtype=np.float64)
    # Append t1 unless the last interior point already equals t1 to
    # within float tolerance.
    if interior.size == 0 or abs(interior[-1] - t1) > 1e-9 * max(abs(t1), 1.0):
        ts_arr = np.concatenate([interior, np.asarray([t1], dtype=np.float64)])
    else:
        ts_arr = interior
    return jnp.asarray(ts_arr)


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
# Scenario / model param packing for build() dispatch
# ----------------------------------------------------------------------


def _scenario_params(spec: RunSpec) -> dict[str, Any]:
    """Pack the scenario block into the kwargs dict ``build()`` expects."""
    return {
        "grid": dict(spec.scenario.grid),
        "consts": dict(spec.scenario.consts),
        "forcing": dict(spec.scenario.forcing),
        "initial_condition": dict(spec.scenario.initial_condition),
    }


def _model_params(spec: RunSpec) -> dict[str, Any]:
    """Pack the model block into the kwargs dict ``build()`` expects."""
    return {
        "stratification": dict(spec.model.stratification),
        "params": dict(spec.model.params),
    }


# ----------------------------------------------------------------------
# Persistence helpers
# ----------------------------------------------------------------------


def _ensure_clean_dir(path: Path) -> None:
    """Make ``path`` exist as a directory. Existing contents are kept."""
    path.mkdir(parents=True, exist_ok=True)


def _attrs_for(spec: RunSpec, *, mode: str) -> dict[str, Any]:
    """Standard Dataset attrs that document where this artifact came from.

    Numeric metadata is stored as native floats so downstream tooling
    (xarray comparisons, plotting, dataset filtering) doesn't have to
    parse strings.
    """
    return {
        "somax_sim_mode": mode,
        "scenario_name": spec.scenario.name,
        "model_name": spec.model.name,
        "t0": float(spec.timestepping.t0),
        "t1": float(spec.timestepping.t1),
        "dt": float(spec.timestepping.dt),
        "save_interval": float(spec.timestepping.save_interval),
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
    diagnostics_per_save: int = 1,
    checkpoint_every_n_chunks: int = 0,
    monitors: list[Monitor] | None = None,
) -> SimulationResult:
    """Build the model, integrate, write artifacts, return summary.

    Always runs the chunked-integration path so a structured
    ``run.log`` is written under ``output_dir`` regardless of how the
    user invoked the CLI. The ``diagnostics_per_save`` knob controls
    how finely each save interval is subdivided for diagnostic logging
    — higher values mean more lines in ``run.log`` per snapshot.

    Args:
        spec: Validated, debug-merged run spec.
        output_dir: Where to write artifacts.
        mode: ``"run"``, ``"spinup"``, or ``"restart"`` — used in attrs
            and to decide what to save.
        initial_state: If non-None, override the factory-built initial
            state with this one (used by ``restart``).
        diagnostics_per_save: Number of diagnostic sub-chunks per save
            interval. ``1`` = one diagnostic per snapshot (default).
            ``4`` = four diagnostics per snapshot (finer monitoring).
            Snapshot cadence is unaffected.
        checkpoint_every_n_chunks: Crash-recovery cadence (#70). ``0``
            (default) disables crash-recovery checkpoints; any positive
            ``N`` writes ``<output_dir>/checkpoint.zarr`` after every
            Nth snapshot-aligned chunk.

    Raises:
        ValueError: If ``checkpoint_every_n_chunks`` is negative.
    """
    if checkpoint_every_n_chunks < 0:
        raise ValueError(
            f"checkpoint_every_n_chunks must be >= 0, got "
            f"{checkpoint_every_n_chunks!r}. Use 0 to disable checkpointing "
            f"or a positive integer to enable it."
        )

    output_dir = Path(output_dir)
    _ensure_clean_dir(output_dir)

    # Delete any stale checkpoint.zarr from a prior run. With
    # checkpointing enabled, the chunked loop overwrites this atomically
    # from chunk 1 onwards, but a crash before chunk 1 would leave the
    # previous run's state visible to ``somax-sim restart``. With
    # checkpointing disabled, we still want to prevent the
    # "run-into-a-dirty-dir" footgun Codex flagged on PR #98: subsequent
    # ``restart --from <output_dir>/checkpoint.zarr`` could otherwise
    # silently resume from an unrelated prior run.
    stale_ckpt = output_dir / "checkpoint.zarr"
    if stale_ckpt.exists():
        import shutil

        shutil.rmtree(stale_ckpt)
        logger.info("removed stale checkpoint.zarr from output_dir before {} run", mode)

    logger.info(
        "somax-sim mode={} scenario={} model={}",
        mode,
        spec.scenario.name,
        spec.model.name,
    )
    logger.info("output_dir={}", output_dir)

    # 1. Build model and initial state via the scenario x model dispatcher.
    model, factory_state0 = build(
        spec.scenario.name,
        spec.model.name,
        scenario_params=_scenario_params(spec),
        model_params=_model_params(spec),
    )
    state0 = initial_state if initial_state is not None else factory_state0

    # 2. Preflight assertions (CFL, parameter bounds, ...). These run
    #    BEFORE we burn any compute on integration; failures fail loudly
    #    so DVC stages and CI catch config errors early.
    if spec.assertions:
        logger.info("running {} preflight assertion(s)", len(spec.assertions))
    _assertions.run_preflight(spec, model)

    # 3. Build save times. Spinup uses [t0, t1] (snapshot only the
    #    endpoint, but t0 anchors the chunked integrator).
    only_final = mode == "spinup"
    save_ts = _build_save_times(spec, only_final=only_final)

    # 4. Always run the chunked-integration path. Open the run.log
    #    sink + alive thread up front so any failure during the
    #    integration also lands in the file.
    logger.info(
        "integrating: t0={} t1={} dt={} save_n={} diagnostics_per_save={}",
        spec.timestepping.t0,
        spec.timestepping.t1,
        spec.timestepping.dt,
        save_ts.shape[0],
        diagnostics_per_save,
    )
    # diffrax defaults max_steps to 4096 which is far too small for any
    # multi-day simulation. Compute the actual minimum + a 20% buffer.
    window = spec.timestepping.t1 - spec.timestepping.t0
    expected_steps = int(window / spec.timestepping.dt)
    max_steps = max(16384, int(expected_steps * 1.2))

    run_log = start_run_log(output_dir, label=f"somax-sim/{mode}")
    run_log.log.debug(
        f"integration starting: scenario={spec.scenario.name} "
        f"model={spec.model.name} "
        f"t0={format_time_seconds(spec.timestepping.t0)} "
        f"t1={format_time_seconds(spec.timestepping.t1)} "
        f"dt={spec.timestepping.dt} s "
        f"save_n={save_ts.shape[0]} "
        f"diagnostics_per_save={diagnostics_per_save}"
    )

    n_save_intervals = max(int(save_ts.shape[0]) - 1, 1)
    n_diag_intervals = n_save_intervals * max(int(diagnostics_per_save), 1)
    per_chunk_max = max(2048, max_steps // n_diag_intervals + 256)

    t_start = time.perf_counter()
    try:
        sol = _chunked_integrate_with_diagnostics(
            model,
            state0,
            save_ts,
            spec.timestepping.dt,
            diagnostics_per_save=diagnostics_per_save,
            max_steps_per_chunk=per_chunk_max,
            run_log=run_log,
            mode=mode,
            monitors=monitors,
            spec=spec,
            checkpoint_every_n_chunks=checkpoint_every_n_chunks,
            checkpoint_dir=output_dir if checkpoint_every_n_chunks > 0 else None,
            checkpoint_label=f"{spec.scenario.name}/{spec.model.name}",
        )
    except eqx.EquinoxRuntimeError as exc:
        # An in-JIT guard (somax.guards, e.g. layer-thickness positivity or
        # the RHS non-finite tripwire) fired *inside* model.integrate. This
        # halts at the offending step — earlier than the chunk-boundary
        # monitors — but the runner's contract is that a blow-up surfaces as
        # IntegrationDivergedError and writes no artifacts, so translate it
        # while preserving the guard's diagnostic.
        stop_run_log(
            run_log,
            final_message=f"FAILED during integrate: {type(exc).__name__}: {exc}",
        )
        raise IntegrationDivergedError(
            f"somax-sim {mode} integration produced a non-finite or "
            f"non-physical state and an in-JIT guard halted it at the "
            f"offending step (likely a CFL violation): {exc}\n"
            f"  Refusing to write artifacts."
        ) from exc
    except Exception as exc:
        stop_run_log(
            run_log,
            final_message=f"FAILED during integrate: {type(exc).__name__}: {exc}",
        )
        raise
    wallclock = time.perf_counter() - t_start
    logger.info("integration finished in {}", format_wallclock(wallclock))
    run_log.log.debug(f"integration finished in {format_wallclock(wallclock)}")

    try:
        # 5. Sanity check: blow up loudly if integration produced NaN/Inf.
        #    This is the unconditional safety net (always runs, regardless
        #    of opt-in assertions). Without it, a CFL violation produces an
        #    all-NaN field, diagnose() happily computes NaN reductions, and
        #    the run "succeeds" with a metrics.json full of NaN. Refuse to
        #    write any artifacts so DVC stages and CI fail.
        _assert_finite_state(sol.ys, mode=mode)

        # 6. Compute the final state and the metrics dict (the latter feeds
        #    both the postflight assertions and metrics.json).
        final_state = _extract_final_state(sol.ys, type(state0))

        metrics: dict[str, Any] = {}
        if spec.output.write_metrics and not only_final:
            try:
                diagnostics = model.diagnose(final_state)
            except Exception as exc:
                logger.warning("model.diagnose failed: {}", exc)
                diagnostics = None
            metrics = _flatten_diagnostics(diagnostics)
            try:
                metrics.update(compute_eval_metrics(model, final_state))
            except Exception as exc:
                logger.warning("compute_eval_metrics failed: {}", exc)
            metrics["wallclock_seconds"] = wallclock
            metrics["n_steps"] = _maybe_step_count(sol)
            metrics["t0"] = spec.timestepping.t0
            metrics["t1"] = spec.timestepping.t1
            metrics["save_interval"] = spec.timestepping.save_interval
            metrics["mode"] = mode

        # 7. Postflight assertions. Run BEFORE writing artifacts so a failed
        #    assertion fails the run cleanly. Empty metrics (e.g. spinup)
        #    just means most postflight assertions become no-ops.
        if spec.assertions:
            logger.info("running {} postflight assertion(s)", len(spec.assertions))
        _assertions.run_postflight(spec, metrics)

        # 8. Write snapshots.zarr (skipped for spinup or if disabled).
        snapshots_path: Path | None = None
        if spec.output.write_snapshots and not only_final:
            snapshots_path = output_dir / "snapshots.zarr"
            snapshots_ds = io.snapshots_to_dataset(
                sol.ys, np.asarray(save_ts), attrs=_attrs_for(spec, mode=mode)
            )
            io.save_dataset(snapshots_ds, snapshots_path, mode="w")
            logger.info("wrote {}", snapshots_path)
            run_log.log.debug(f"wrote {snapshots_path.name}")

        # 9. Always write final_state.zarr — this is the restart artifact.
        final_state_path = output_dir / "final_state.zarr"
        final_ds = io.state_to_dataset(
            final_state,
            time=float(spec.timestepping.t1),
            attrs=_attrs_for(spec, mode=mode),
        )
        io.save_dataset(final_ds, final_state_path, mode="w")
        logger.info("wrote {}", final_state_path)
        run_log.log.debug(f"wrote {final_state_path.name}")

        # 10. Write metrics.json now that postflight has cleared.
        metrics_path: Path | None = None
        if spec.output.write_metrics and not only_final and metrics:
            metrics_path = output_dir / "metrics.json"
            with metrics_path.open("w") as f:
                json.dump(metrics, f, indent=2, sort_keys=True)
            logger.info("wrote {}", metrics_path)

        # 11. Resolved config dump (debugging aid; not git/dvc tracked).
        resolved_path = output_dir / "resolved.yaml"
        dump_yaml(spec, str(resolved_path))
        logger.info("wrote {}", resolved_path)

        # 12. Provenance manifest (git commit, library versions, platform,
        #     solver/timestepping) for run reproducibility + determinism CI.
        manifest_path = output_dir / "manifest.json"
        manifest = _build_manifest(spec, mode=mode, wallclock=wallclock)
        with manifest_path.open("w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        logger.info("wrote {}", manifest_path)
    except Exception as exc:
        stop_run_log(
            run_log,
            final_message=f"FAILED postflight: {type(exc).__name__}: {exc}",
        )
        raise

    stop_run_log(run_log, final_message="finished cleanly")

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


def _state_diagnostics(state: Any) -> dict[str, dict[str, float]]:
    """Compute per-field min/mean/max and NaN count for a single State.

    Used by the chunked-integration path to write physical diagnostics
    into ``run.log`` between integration chunks. Walks every field of
    the State pytree, converts to numpy, and reduces.

    Returns:
        ``{field_name: {"min": ..., "mean": ..., "max": ..., "nan": int}}``
        — one entry per field. Reductions are nanmin/nanmean/nanmax so
        the report is meaningful even if some entries are NaN.
    """
    out: dict[str, dict[str, float]] = {}
    for field in dataclasses.fields(state):
        leaf = np.asarray(getattr(state, field.name))
        n_nan = int(np.sum(~np.isfinite(leaf)))
        if leaf.size == n_nan:
            # Everything is NaN — stats would also be NaN; report explicitly.
            out[field.name] = {
                "min": float("nan"),
                "mean": float("nan"),
                "max": float("nan"),
                "nan": n_nan,
            }
            continue
        out[field.name] = {
            "min": float(np.nanmin(leaf)),
            "mean": float(np.nanmean(leaf)),
            "max": float(np.nanmax(leaf)),
            "nan": n_nan,
        }
    return out


def _format_state_stats(diag: dict[str, dict[str, float]]) -> str:
    """Format the state diagnostics dict with per-field units."""
    return " ".join(
        format_field_stats(
            name,
            min_val=stats["min"],
            mean_val=stats["mean"],
            max_val=stats["max"],
            nan_count=stats["nan"],
        )
        for name, stats in diag.items()
    )


def _format_physical_scalars(model: Any, state: Any) -> tuple[str, dict[str, float]]:
    """Compute and format model.diagnose() scalars for the run-log line.

    Returns ``(formatted_line, raw_dict)`` where the dict is the same
    flattened-diagnostics shape used for ``metrics.json`` (so callers can
    cross-check or persist it). The formatted line shows the canonical
    physical invariants — total energy, total enstrophy, total KE,
    relative vorticity ranges — that should evolve smoothly. Sudden
    jumps signal a problem long before NaNs appear.

    The function never raises; if ``diagnose`` fails (e.g. for a state
    that's already corrupted) it returns an empty line and dict.
    """
    try:
        diagnostics = model.diagnose(state)
    except Exception as exc:
        return f"diagnose failed: {type(exc).__name__}: {exc}", {}
    flat = _flatten_diagnostics(diagnostics)
    if not flat:
        return "", flat
    # Pick a small set of "headline" scalars in priority order. We
    # display whatever is present in the flat dict; the rest is
    # available via the persisted metrics.json.
    headline_keys = (
        "total_energy",
        "total_kinetic_energy",
        "energy",
        "kinetic_energy",
        "total_enstrophy",
        "enstrophy",
    )
    parts: list[str] = []
    for key in headline_keys:
        if key in flat:
            parts.append(f"{key}={flat[key]:.3g}")
    # Per-layer energy is also useful when present.
    layer_energies = sorted(k for k in flat if k.startswith("energy_layer_"))
    if layer_energies and "total_energy" not in flat and "energy" not in flat:
        layer_str = ",".join(f"{flat[k]:.3g}" for k in layer_energies)
        parts.append(f"energy_layers=[{layer_str}]")
    return " ".join(parts), flat


def _build_diagnostic_grid(
    save_ts: np.ndarray, diagnostics_per_save: int
) -> tuple[np.ndarray, set[int]]:
    """Build a diagnostic time grid that subdivides each save interval.

    The diagnostic grid always contains every element of ``save_ts``
    (so the chunked integrator can snapshot at the right times) plus
    ``diagnostics_per_save - 1`` extra interior points per save
    interval for higher-resolution diagnostic logging.

    Args:
        save_ts: Save times, length >= 2.
        diagnostics_per_save: Number of diagnostic sub-chunks per save
            interval. ``1`` returns ``save_ts`` unchanged.

    Returns:
        Tuple ``(diag_ts, save_indices)`` where ``save_ts == diag_ts[
        sorted(save_indices)]``. Iterating ``range(len(diag_ts) - 1)``
        yields the chunk-by-chunk integration intervals; ``i + 1`` in
        ``save_indices`` means "the endpoint of chunk i is a snapshot
        time."
    """
    n_per_save = max(int(diagnostics_per_save), 1)
    if n_per_save == 1:
        return save_ts, set(range(save_ts.shape[0]))

    parts: list[float] = [float(save_ts[0])]
    for i in range(save_ts.shape[0] - 1):
        sub = np.linspace(save_ts[i], save_ts[i + 1], n_per_save + 1)
        # Skip the first element of `sub` — it duplicates parts[-1].
        parts.extend(float(x) for x in sub[1:])
    diag_ts = np.asarray(parts)
    save_indices = {i * n_per_save for i in range(save_ts.shape[0])}
    return diag_ts, save_indices


@dataclasses.dataclass(frozen=True)
class _ChunkCarry:
    """Carry-state threaded through the chunked integration `Cycle`.

    Args:
        index: Next diagnostic-chunk index to integrate (0-based).
        snapshot_ordinal: Count of snapshots retained so far, excluding
            the initial state (matches ``len(save_states) - 1`` in the
            original loop — the cadence key for crash-recovery
            checkpoints).
        save_states: Snapshot states retained so far, starting with the
            initial state. Stacked into the returned trajectory. A
            ``list`` appended **in place** at snapshot boundaries — the
            frozen dataclass only forbids rebinding the field, not
            mutating the list it points to, so accumulation stays
            amortized O(1) per snapshot (a tuple would be O(N²)).

    Energy-growth tracking, which used to live on this carry, now lives
    inside :class:`somax.monitor.EnergyGrowthMonitor` (a host-side object),
    so the carry is back to pure bookkeeping.
    """

    index: int
    snapshot_ordinal: int
    save_states: list[Any]


class _ChunkStep(StatefulOperator):
    """One diagnostic chunk of the integration as a ``StatefulOperator``.

    ``_apply(state, carry)`` integrates ``model`` from one diagnostic time
    to the next, logs physical + state diagnostics, aborts on a non-finite
    state, retains the state when the chunk endpoint is a snapshot
    boundary, and writes a crash-recovery checkpoint when due — i.e. the
    body of the original chunked loop, with the per-iteration accumulators
    moved into :class:`_ChunkCarry`. Holds runtime-only references
    (model, run log), so it is excluded from config serialization.
    """

    __config_mixin_auto__ = False
    forbid_in_yaml = True

    def __init__(
        self,
        *,
        model: Any,
        state_class: type,
        dt: float,
        diag_ts: np.ndarray,
        save_indices: set[int],
        n_diag_intervals: int,
        max_steps_per_chunk: int,
        run_log: RunLogContext,
        mode: str,
        checkpoint_every_n_chunks: int,
        checkpoint_dir: Path | None,
        checkpoint_label: str | None,
        monitors: list[Monitor],
    ) -> None:
        self.model = model
        self.state_class = state_class
        self.dt = dt
        self.diag_ts = diag_ts
        self.save_indices = save_indices
        self.n_diag_intervals = n_diag_intervals
        self.max_steps_per_chunk = max_steps_per_chunk
        self.run_log = run_log
        self.mode = mode
        self.checkpoint_every_n_chunks = checkpoint_every_n_chunks
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_label = checkpoint_label
        self.monitors = monitors

    def _apply(self, state: Any, carry: _ChunkCarry) -> tuple[Any, _ChunkCarry]:
        log = self.run_log.log
        i = carry.index
        chunk_t0 = float(self.diag_ts[i])
        chunk_t1 = float(self.diag_ts[i + 1])

        t_chunk_start = time.perf_counter()
        sol = self.model.integrate(
            state,
            chunk_t0,
            chunk_t1,
            self.dt,
            saveat=dfx.SaveAt(t1=True),
            max_steps=self.max_steps_per_chunk,
        )
        chunk_wall = time.perf_counter() - t_chunk_start

        new_state = _extract_final_state(sol.ys, self.state_class)
        state_diag = _state_diagnostics(new_state)
        phys_line, _phys_flat = _format_physical_scalars(self.model, new_state)

        # Fan out to the registered monitors. The non-finite abort and the
        # energy-growth warning that used to be inline here are now the
        # NonFiniteMonitor / EnergyGrowthMonitor plugins (see
        # somax.monitor.default_monitors); each monitor records metrics +
        # messages and may request a clean termination.
        is_snapshot_boundary = (i + 1) in self.save_indices
        info = ChunkInfo(
            index=i,
            n_chunks=self.n_diag_intervals,
            t0=chunk_t0,
            t1=chunk_t1,
            wall_seconds=chunk_wall,
            is_snapshot=is_snapshot_boundary,
            stats=_solver_stats(sol),
        )
        monitor_msgs: list[str] = []
        monitor_metrics: dict[str, float] = {}
        terminate_reason: str | None = None
        for mon in self.monitors:
            verdict = mon.on_chunk_end(self.model, new_state, info)
            # Record-only diagnostics: prefix with the monitor name so keys
            # from different monitors don't collide in the per-chunk log.
            for key, value in verdict.metrics.items():
                monitor_metrics[f"{mon.name}.{key}"] = value
            for msg in verdict.messages:
                monitor_msgs.append(f"[{mon.name}] {msg}")
            if verdict.terminate and terminate_reason is None:
                terminate_reason = f"monitor {mon.name!r}: {verdict.reason}"

        metric_line = ""
        if monitor_metrics:
            metric_line = " | monitors: " + " ".join(
                f"{k}={v:.3g}" for k, v in monitor_metrics.items()
            )
        log.debug(
            f"chunk {i + 1}/{self.n_diag_intervals} "
            f"sim_t={format_time_seconds(chunk_t1)} | "
            f"{_format_state_stats(state_diag)}"
            + (f" | physics: {phys_line}" if phys_line else "")
            + f" | wall={format_wallclock(chunk_wall)}"
            + metric_line
            + ("".join(f" !! {m}" for m in monitor_msgs) if monitor_msgs else "")
        )

        if terminate_reason is not None:
            log.debug(
                f"ABORT at chunk {i + 1}/{self.n_diag_intervals}: {terminate_reason}"
            )
            raise IntegrationDivergedError(
                f"somax-sim {self.mode} integration halted during chunk "
                f"{i + 1}/{self.n_diag_intervals} "
                f"(sim_t={format_time_seconds(chunk_t1)}): {terminate_reason}.\n"
                f"  {_format_state_stats(state_diag)}\n"
                f"  Refusing to write artifacts. The condition appeared between "
                f"sim_t={format_time_seconds(chunk_t0)} and "
                f"sim_t={format_time_seconds(chunk_t1)} — earlier chunks were "
                f"clean."
            )

        # Only retain states at snapshot boundaries. Append in place to the
        # threaded list (amortized O(1)); the same list object flows through
        # every carry, so the final carry holds the full trajectory.
        # ``is_snapshot_boundary`` was computed above for the ChunkInfo.
        new_save_states = carry.save_states
        if is_snapshot_boundary:
            new_save_states.append(new_state)
        new_ordinal = carry.snapshot_ordinal + (1 if is_snapshot_boundary else 0)

        # Crash-recovery checkpoint (#70): write the current state to a
        # rolling ``checkpoint.zarr`` every N snapshot-aligned chunk
        # endpoints. Aligning to snapshot boundaries keeps the on-disk
        # checkpoint consistent with the saved snapshots up to that
        # point. We write AFTER the non-finite guard so a bad state
        # never lands on disk.
        if (
            self.checkpoint_every_n_chunks > 0
            and self.checkpoint_dir is not None
            and is_snapshot_boundary
        ):
            # ``new_ordinal`` == len(save_states) - 1 after the append.
            snapshot_ordinal = new_ordinal
            due = (
                snapshot_ordinal > 0
                and snapshot_ordinal % self.checkpoint_every_n_chunks == 0
            )
            if due:
                ckpt_path = Path(self.checkpoint_dir) / "checkpoint.zarr"
                ckpt_attrs: dict[str, Any] = {
                    "somax_sim_t": float(chunk_t1),
                    "somax_snapshot_ordinal": int(snapshot_ordinal),
                }
                if self.checkpoint_label:
                    ckpt_attrs["somax_checkpoint_of"] = str(self.checkpoint_label)
                ckpt_ds = io.state_to_dataset(
                    new_state, time=float(chunk_t1), attrs=ckpt_attrs
                )
                # A transient I/O error writing the crash-recovery checkpoint
                # (e.g. an SMB / network-mount hiccup) must not kill an
                # otherwise-healthy run — that defeats the point of the
                # checkpoint. Retry once, then warn and continue; the next
                # snapshot boundary will checkpoint again.
                for _attempt in (1, 2):
                    try:
                        io.save_dataset(ckpt_ds, ckpt_path, mode="w")
                        log.debug(
                            f"checkpoint written at snapshot {snapshot_ordinal} "
                            f"(sim_t={format_time_seconds(chunk_t1)}) "
                            f"→ {ckpt_path.name}"
                        )
                        break
                    except OSError as exc:
                        if _attempt == 1:
                            log.warning(f"checkpoint write failed ({exc!r}); retrying")
                        else:
                            log.warning(
                                f"checkpoint write failed twice ({exc!r}); "
                                f"continuing without this checkpoint"
                            )

        new_carry = _ChunkCarry(
            index=i + 1,
            snapshot_ordinal=new_ordinal,
            save_states=new_save_states,
        )
        return new_state, new_carry


def _chunked_integrate_with_diagnostics(
    model: Any,
    state0: Any,
    save_ts: jnp.ndarray,
    dt: float,
    *,
    diagnostics_per_save: int,
    max_steps_per_chunk: int,
    run_log: RunLogContext,
    mode: str,
    monitors: list[Monitor] | None = None,
    spec: RunSpec | None = None,
    checkpoint_every_n_chunks: int = 0,
    checkpoint_dir: Path | None = None,
    checkpoint_label: str | None = None,
) -> Any:
    """Run a multi-chunk integration that emits diagnostics between chunks.

    Each chunk runs from one diagnostic time to the next using
    ``SaveAt(t1=True)``. Between chunks we compute physical diagnostics
    (KE, enstrophy, energy) plus state-field min/mean/max and emit them
    via the bound run-log logger (``run_log.log.debug(...)``). They
    land in ``run.log`` always, and on stderr only if the user passed
    ``--verbose``.

    With ``diagnostics_per_save > 1`` each save interval is subdivided
    into N diagnostic sub-chunks, giving N times more log lines per
    snapshot without changing the snapshot cadence.

    If a chunk produces a non-finite state, the integration aborts early
    — the user sees the blow-up *during* the run, not after the full
    integration completes.

    The returned object is a small SimpleNamespace with a ``ys``
    attribute holding the time-stacked state pytree, with one entry per
    element of ``save_ts`` (snapshots only — diagnostic-only sub-chunk
    endpoints are not retained).

    Args:
        model: The constructed somax model.
        state0: Initial state.
        save_ts: Snapshot timestamps from ``_build_save_times`` —
            length >= 2.
        dt: Time step.
        diagnostics_per_save: Sub-chunks per save interval (default 1).
        max_steps_per_chunk: Per-chunk diffrax max_steps.
        run_log: RunLogContext supplying the bound logger.
        mode: ``"run"`` / ``"spinup"`` / ``"restart"`` for error messages.
        checkpoint_every_n_chunks: Crash-recovery cadence (#70). If
            ``> 0`` and ``checkpoint_dir`` is provided, the current
            state is written to ``<checkpoint_dir>/checkpoint.zarr``
            after every Nth *diagnostic* chunk endpoint that is also a
            snapshot boundary (so ``save_states`` and the on-disk
            checkpoint stay aligned). ``0`` disables checkpointing.
        checkpoint_dir: Directory for the rolling ``checkpoint.zarr``
            store. Ignored when ``checkpoint_every_n_chunks == 0``.
        checkpoint_label: Optional label (e.g. ``"<scenario>/<model>"``)
            stored in the checkpoint's attrs for debugging.

    Raises:
        IntegrationDivergedError: If any chunk produces non-finite state.

    The chunk-by-chunk loop is driven by :class:`pipekit_cycle.Cycle`: the
    body lives in :class:`_ChunkStep` (a ``StatefulOperator``) and the
    snapshot list / energy tracker / checkpoint bookkeeping are threaded
    through an immutable :class:`_ChunkCarry`. Observability (diagnostic
    logging, the non-finite abort, crash-recovery checkpoints) runs inside
    the step operator — pipekit's per-step hook dispatch is a reserved
    no-op today, so the step is the place those side effects live.
    """
    from types import SimpleNamespace

    import jax

    log = run_log.log
    active_monitors: list[Monitor] = (
        monitors if monitors is not None else default_monitors()
    )
    save_ts_np = np.asarray(save_ts)
    diag_ts, save_indices = _build_diagnostic_grid(save_ts_np, diagnostics_per_save)
    n_diag_intervals = diag_ts.shape[0] - 1

    # Initial diagnostics (before any integration).
    init_state_diag = _state_diagnostics(state0)
    init_phys, _init_flat = _format_physical_scalars(model, state0)
    log.debug(
        f"chunk 0/{n_diag_intervals} sim_t={format_time_seconds(float(diag_ts[0]))} | "
        f"{_format_state_stats(init_state_diag)}"
        + (f" | physics: {init_phys}" if init_phys else "")
        + " | initial state"
    )

    for mon in active_monitors:
        mon.on_run_start(model, state0, spec=spec)

    step = _ChunkStep(
        model=model,
        state_class=type(state0),
        dt=dt,
        diag_ts=diag_ts,
        save_indices=save_indices,
        n_diag_intervals=n_diag_intervals,
        max_steps_per_chunk=max_steps_per_chunk,
        run_log=run_log,
        mode=mode,
        checkpoint_every_n_chunks=checkpoint_every_n_chunks,
        checkpoint_dir=checkpoint_dir,
        checkpoint_label=checkpoint_label,
        monitors=active_monitors,
    )
    carry0 = _ChunkCarry(
        index=0,
        snapshot_ordinal=0,
        save_states=[state0],
    )
    cycle = Cycle(step_op=step, n_steps=n_diag_intervals)

    t_chunks_start = time.perf_counter()
    try:
        final_state, final_carry = cycle(state0, carry0)
    except Exception:
        for mon in active_monitors:
            mon.on_run_end(model, state0, ok=False)
        raise
    total_chunks_wall = time.perf_counter() - t_chunks_start
    log.debug(
        f"all {n_diag_intervals} chunks completed in "
        f"{format_wallclock(total_chunks_wall)}"
    )

    for mon in active_monitors:
        mon.on_run_end(model, final_state, ok=True)

    # Concatenate snapshot states into a single time-stacked pytree
    # matching the shape diffrax would have returned with
    # ``SaveAt(ts=save_ts)``.
    stacked = jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs, axis=0), *final_carry.save_states
    )
    return SimpleNamespace(ys=stacked, stats={})


def _assert_finite_state(stacked: Any, *, mode: str) -> None:
    """Raise :class:`IntegrationDivergedError` if any field contains NaN/Inf.

    Walks every field of the stacked-state pytree (the diffrax solution's
    ``ys``) and checks for non-finite values. The diagnostic message
    includes which fields contained how many bad values, and the
    fraction of NaNs at the *first* time slice (a quick way to tell
    "blew up immediately" from "blew up partway through").

    Args:
        stacked: ``sol.ys`` — pytree where each leaf has a leading time axis.
        mode: ``"run"`` / ``"spinup"`` / ``"restart"``, included in the error.
    """
    bad_fields: list[str] = []
    first_step_bad: list[str] = []
    for field in dataclasses.fields(stacked):
        leaf = np.asarray(getattr(stacked, field.name))
        n_bad = int(np.sum(~np.isfinite(leaf)))
        if n_bad:
            bad_fields.append(f"{field.name} ({n_bad}/{leaf.size} non-finite)")
            # Was the very first save-step already corrupted?
            if leaf.shape[0] >= 1:
                first_n_bad = int(np.sum(~np.isfinite(leaf[0])))
                if first_n_bad:
                    first_step_bad.append(
                        f"{field.name} ({first_n_bad}/{leaf[0].size} at t0)"
                    )

    if not bad_fields:
        return

    msg_lines = [
        f"somax-sim {mode} integration produced non-finite values:",
        "  " + "; ".join(bad_fields),
    ]
    if first_step_bad:
        msg_lines.append("  → already non-finite at the first saved step:")
        msg_lines.append("    " + "; ".join(first_step_bad))
        msg_lines.append(
            "  This usually means CFL violation at t0 or bad initial conditions."
        )
    else:
        msg_lines.append(
            "  Mid-integration blow-up. Most common cause: CFL violation "
            "(time step too large for the grid spacing and the fastest wave)."
        )
    msg_lines.append(
        "  Refusing to write artifacts. Check the timestepping (dt vs dx, "
        "wave speeds) and try a smaller dt."
    )
    raise IntegrationDivergedError("\n".join(msg_lines))


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


def _git_provenance() -> dict[str, Any]:
    """Best-effort ``{git_commit, git_dirty}`` for the somax checkout.

    Returns ``{"git_commit": None, "git_dirty": None}`` if git is unavailable
    or this is not a working tree (e.g. an installed wheel).
    """
    import subprocess

    repo_dir = Path(__file__).resolve().parent
    try:
        commit = (
            subprocess.check_output(
                ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        status = subprocess.check_output(
            ["git", "-C", str(repo_dir), "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        ).decode()
        return {"git_commit": commit, "git_dirty": bool(status.strip())}
    except (subprocess.SubprocessError, OSError):
        return {"git_commit": None, "git_dirty": None}


def _build_manifest(spec: RunSpec, *, mode: str, wallclock: float) -> dict[str, Any]:
    """Assemble the provenance manifest written next to ``metrics.json``.

    Captures the git commit + dirty flag, key library versions, the JAX
    platform / x64 flag / device count, a config hash, and the
    solver/timestepping knobs — enough to reproduce a run and to drive a
    determinism check in CI.
    """
    import hashlib
    import importlib.metadata as importlib_metadata
    import platform as _platform

    import jax

    def _version(pkg: str) -> str | None:
        try:
            return importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            return None

    try:
        config_repr = json.dumps(spec.to_dict(), sort_keys=True, default=str)
    except (TypeError, ValueError, AttributeError):
        config_repr = repr(spec)
    config_sha = hashlib.sha256(config_repr.encode("utf-8")).hexdigest()

    manifest: dict[str, Any] = {
        "mode": mode,
        "somax_version": _version("somax"),
        "jax_version": getattr(jax, "__version__", None),
        "diffrax_version": _version("diffrax"),
        "finitevolx_version": _version("finitevolx"),
        "x64_enabled": bool(jax.config.read("jax_enable_x64")),
        "platform": jax.default_backend(),
        "python": _platform.python_version(),
        "device_count": jax.device_count(),
        "config_sha256": config_sha,
        "t0": spec.timestepping.t0,
        "t1": spec.timestepping.t1,
        "dt": spec.timestepping.dt,
        "wallclock_seconds": wallclock,
    }
    manifest.update(_git_provenance())
    return manifest


def _solver_stats(sol: dfx.Solution) -> dict[str, Any]:
    """Host-side copy of a diffrax solution's solver stats + result.

    Pulls ``num_accepted_steps`` / ``num_rejected_steps`` / ``num_steps``
    (coerced to plain ints) and the ``result`` so they survive onto a
    :class:`somax.monitor.ChunkInfo`, where the runner previously discarded
    them (``stats={}``). Best-effort: returns whatever is available.
    """
    import contextlib

    out: dict[str, Any] = {}
    stats = getattr(sol, "stats", None)
    if stats:
        for key in ("num_steps", "num_accepted_steps", "num_rejected_steps"):
            value = stats.get(key)
            if value is None:
                continue
            with contextlib.suppress(TypeError, ValueError):
                out[key] = int(np.asarray(value))
    result = getattr(sol, "result", None)
    if result is not None:
        # ``str(result)`` is uninformative (every diffrax RESULTS renders as
        # ``RESULTS<>``), so record a clean success flag instead.
        with contextlib.suppress(Exception):
            out["result_successful"] = bool(result == dfx.RESULTS.successful)
    return out


# ----------------------------------------------------------------------
# Public entry points — one per CLI subcommand
# ----------------------------------------------------------------------


def simulate(
    spec: RunSpec,
    output_dir: str | Path,
    *,
    diagnostics_per_save: int = 1,
    checkpoint_every_n_chunks: int = 0,
    monitors: list[Monitor] | None = None,
) -> SimulationResult:
    """Run a fresh simulation from factory-built initial conditions.

    A structured ``run.log`` is always written under ``output_dir``;
    set ``diagnostics_per_save`` to subdivide each save interval for
    finer monitoring.

    Args:
        spec: Run specification. Should already be debug-merged and
            validated by the caller.
        output_dir: Directory for written artifacts.
        diagnostics_per_save: Diagnostic sub-chunks per save interval
            (default 1). Higher = more lines per snapshot in run.log.
        checkpoint_every_n_chunks: Crash-recovery cadence (#70). ``0``
            (default) disables crash checkpoints.
        monitors: Chunk-boundary monitors (see :mod:`somax.monitor`).
            ``None`` (default) uses ``default_monitors()`` — the same
            non-finite abort + energy-growth warning as before, plus
            record-only conservation/solver/throughput diagnostics.
    """
    return _integrate_and_write(
        spec,
        Path(output_dir),
        mode="run",
        initial_state=None,
        diagnostics_per_save=diagnostics_per_save,
        checkpoint_every_n_chunks=checkpoint_every_n_chunks,
        monitors=monitors,
    )


def spinup(
    spec: RunSpec,
    output_dir: str | Path,
    *,
    diagnostics_per_save: int = 1,
    checkpoint_every_n_chunks: int = 0,
    monitors: list[Monitor] | None = None,
) -> SimulationResult:
    """Run a spinup integration. Saves only the endpoint.

    Spinup runs are intended to produce a ``final_state.zarr`` artifact
    that downstream production runs use as their initial condition.
    No snapshots, no metrics — just the equilibrium state.
    """
    return _integrate_and_write(
        spec,
        Path(output_dir),
        mode="spinup",
        initial_state=None,
        diagnostics_per_save=diagnostics_per_save,
        checkpoint_every_n_chunks=checkpoint_every_n_chunks,
        monitors=monitors,
    )


def _maybe_override_t0_from_checkpoint(spec: RunSpec, ds: Any) -> RunSpec:
    """Pick up ``somax_sim_t`` from a restart dataset and rebase ``t0``.

    Crash-recovery checkpoints (#70) record the sim-time of the last
    safe chunk endpoint in the ``somax_sim_t`` attr. When present, we
    resume from there rather than restarting the spec's window from
    scratch. Legacy ``final_state.zarr`` stores written before the
    checkpointing work have no such attr and fall through unchanged.
    """
    sim_t_raw = ds.attrs.get("somax_sim_t")
    if sim_t_raw is None:
        return spec
    try:
        sim_t = float(sim_t_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"restart artifact has non-numeric somax_sim_t={sim_t_raw!r} "
            f"(type {type(sim_t_raw).__name__}); this attribute must be a "
            f"number of seconds. If this is a hand-edited zarr store, fix "
            f"or remove the attribute."
        ) from exc
    if sim_t <= spec.timestepping.t0:
        logger.info(
            "restart artifact has somax_sim_t={} <= spec.t0={}; "
            "integrating from spec.t0 unchanged.",
            sim_t,
            spec.timestepping.t0,
        )
        return spec
    if sim_t >= spec.timestepping.t1:
        raise ValueError(
            f"restart artifact has somax_sim_t={sim_t} which is >= "
            f"spec.timestepping.t1={spec.timestepping.t1}; the integration "
            f"window has nothing left to run. Extend t1 in the config or "
            f"restart from an earlier checkpoint."
        )

    # Rebase the window. Shallow-copy the spec so the caller's spec is
    # not mutated.
    from somax._src.cli.spec import TimesteppingSpec

    logger.info(
        "restart: resuming from checkpoint at sim_t={} (overriding spec.t0)",
        sim_t,
    )
    new_ts = TimesteppingSpec(
        t0=sim_t,
        t1=spec.timestepping.t1,
        dt=spec.timestepping.dt,
        save_interval=min(
            spec.timestepping.save_interval, spec.timestepping.t1 - sim_t
        ),
    )
    new_spec = dataclasses.replace(spec, timestepping=new_ts)
    new_spec.validate()
    return new_spec


def restart(
    spec: RunSpec,
    output_dir: str | Path,
    *,
    restart_from: str | Path,
    diagnostics_per_save: int = 1,
    checkpoint_every_n_chunks: int = 0,
    monitors: list[Monitor] | None = None,
) -> SimulationResult:
    """Resume a simulation from a previously saved state.

    The model is built from ``spec`` (so you can change viscosity,
    forcing, etc. between runs); only the *state* is loaded from the
    restart file.

    If the restart artifact is a crash-recovery checkpoint (#70) — i.e.
    it carries a ``somax_sim_t`` attr — the integration window's
    ``t0`` is automatically rebased to that sim-time so the run picks
    up exactly where the checkpoint left off. Legacy
    ``final_state.zarr`` stores without the attr are honoured as-is
    (integration starts at ``spec.timestepping.t0`` as before).

    Args:
        spec: Run specification (model construction comes from here).
        output_dir: Directory for written artifacts.
        restart_from: Path to a zarr store containing a saved state
            (typically a previous run's ``final_state.zarr`` or a
            crash ``checkpoint.zarr``).
        diagnostics_per_save: Diagnostic sub-chunks per save interval
            (default 1).
        checkpoint_every_n_chunks: Crash-recovery cadence for the
            *new* run (the one we're about to start). ``0`` = off.
    """
    restart_path = Path(restart_from)
    logger.info("restart loading state from {}", restart_path)
    ds = io.load_dataset(restart_path)

    # Build the model first to learn the *expected* state class for this
    # scenario x model pair, then load the zarr with that class as the
    # target. Catches the common footgun: --from runs/swm/final_state.zarr
    # + --config configs/doublegyre_bt_qg.yaml would otherwise crash
    # deep inside JAX with an inscrutable trace.
    _model, factory_state0 = build(
        spec.scenario.name,
        spec.model.name,
        scenario_params=_scenario_params(spec),
        model_params=_model_params(spec),
    )
    expected_state_class = type(factory_state0)
    state0 = io.dataset_to_state(ds, state_class=expected_state_class)
    if not isinstance(state0, expected_state_class):
        raise TypeError(
            f"restart artifact at {restart_path} contains a "
            f"{type(state0).__name__}, but the pair "
            f"{spec.scenario.name!r} x {spec.model.name!r} expects "
            f"{expected_state_class.__name__}. Check that --from points at "
            f"a final_state.zarr written by a compatible run."
        )

    spec = _maybe_override_t0_from_checkpoint(spec, ds)

    return _integrate_and_write(
        spec,
        Path(output_dir),
        mode="restart",
        initial_state=state0,
        diagnostics_per_save=diagnostics_per_save,
        checkpoint_every_n_chunks=checkpoint_every_n_chunks,
        monitors=monitors,
    )
