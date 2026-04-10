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
import jax.numpy as jnp
import numpy as np
from loguru import logger

from somax import io
from somax._src.cli import _assertions
from somax._src.cli._factories import get_adapter
from somax._src.cli._progress import RunLogContext, start_run_log, stop_run_log
from somax._src.cli._units import (
    format_field_stats,
    format_time_seconds,
    format_wallclock,
)
from somax._src.cli.spec import RunSpec, dump_yaml


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
    # Inclusive endpoint, evenly spaced.
    n_save = round((ts_data.t1 - ts_data.t0) / ts_data.save_interval) + 1
    n_save = max(int(n_save), 2)
    return jnp.linspace(ts_data.t0, ts_data.t1, n_save)


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
    diagnostics_per_save: int = 1,
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
    """
    output_dir = Path(output_dir)
    _ensure_clean_dir(output_dir)

    logger.info("somax-sim mode={} testcase={}", mode, spec.testcase.name)
    logger.info("output_dir={}", output_dir)

    # 1. Build model and initial state via the adapter registry.
    adapter = get_adapter(spec.testcase.name)
    model, factory_state0 = adapter(
        grid=spec.testcase.grid,
        consts=spec.testcase.consts,
        stratification=spec.testcase.stratification,
        params=spec.testcase.params,
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
        f"integration starting: testcase={spec.testcase.name} "
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
        )
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


def _has_non_finite(diag: dict[str, dict[str, float]]) -> bool:
    """True iff any field reported one or more non-finite values."""
    return any(stats["nan"] > 0 for stats in diag.values())


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

    Raises:
        IntegrationDivergedError: If any chunk produces non-finite state.
    """
    from types import SimpleNamespace

    import jax

    log = run_log.log
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

    state = state0
    save_states: list[Any] = [state0]
    t_chunks_start = time.perf_counter()

    # Track previous total-energy-like scalar across chunks so we can
    # surface large jumps as a "growth" warning even before NaN appears.
    prev_energy: float | None = None

    for i in range(n_diag_intervals):
        chunk_t0 = float(diag_ts[i])
        chunk_t1 = float(diag_ts[i + 1])

        t_chunk_start = time.perf_counter()
        sol = model.integrate(
            state,
            chunk_t0,
            chunk_t1,
            dt,
            saveat=dfx.SaveAt(t1=True),
            max_steps=max_steps_per_chunk,
        )
        chunk_wall = time.perf_counter() - t_chunk_start

        new_state = _extract_final_state(sol.ys, type(state0))
        state_diag = _state_diagnostics(new_state)
        phys_line, phys_flat = _format_physical_scalars(model, new_state)

        # Detect large energy jumps (10x growth between chunks is suspicious)
        # to give the user an early warning before NaN appears.
        energy_value: float | None = None
        for k in (
            "total_energy",
            "energy",
            "total_kinetic_energy",
            "kinetic_energy",
        ):
            if k in phys_flat:
                energy_value = float(phys_flat[k])
                break
        warning = ""
        if (
            prev_energy is not None
            and energy_value is not None
            and abs(prev_energy) > 0
            and not np.isnan(energy_value)
            and abs(energy_value) > 10 * abs(prev_energy)
        ):
            # Both branches inside this block know prev_energy and
            # energy_value are non-None floats; help ty narrow the types.
            prev_e: float = prev_energy
            cur_e: float = energy_value
            growth = cur_e / prev_e if prev_e != 0 else float("inf")
            warning = f" !! energy grew {growth:.1f}x from previous chunk"
        if energy_value is not None and not np.isnan(energy_value):
            prev_energy = energy_value

        log.debug(
            f"chunk {i + 1}/{n_diag_intervals} "
            f"sim_t={format_time_seconds(chunk_t1)} | "
            f"{_format_state_stats(state_diag)}"
            + (f" | physics: {phys_line}" if phys_line else "")
            + f" | wall={format_wallclock(chunk_wall)}"
            + warning
        )

        if _has_non_finite(state_diag):
            log.debug(f"ABORT at chunk {i + 1}/{n_diag_intervals}: non-finite state")
            raise IntegrationDivergedError(
                f"somax-sim {mode} integration produced non-finite values "
                f"during chunk {i + 1}/{n_diag_intervals} "
                f"(sim_t={format_time_seconds(chunk_t1)}).\n"
                f"  {_format_state_stats(state_diag)}\n"
                f"  Refusing to write artifacts. The non-finite values "
                f"appeared between sim_t={format_time_seconds(chunk_t0)} and "
                f"sim_t={format_time_seconds(chunk_t1)} — earlier chunks were "
                f"finite. This is consistent with a slow numerical instability, "
                f"not an immediate CFL violation."
            )

        # Only retain states at snapshot boundaries.
        if (i + 1) in save_indices:
            save_states.append(new_state)
        state = new_state

    total_chunks_wall = time.perf_counter() - t_chunks_start
    log.debug(
        f"all {n_diag_intervals} chunks completed in "
        f"{format_wallclock(total_chunks_wall)}"
    )

    # Concatenate snapshot states into a single time-stacked pytree
    # matching the shape diffrax would have returned with
    # ``SaveAt(ts=save_ts)``.
    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *save_states)
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


# ----------------------------------------------------------------------
# Public entry points — one per CLI subcommand
# ----------------------------------------------------------------------


def simulate(
    spec: RunSpec,
    output_dir: str | Path,
    *,
    diagnostics_per_save: int = 1,
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
    """
    return _integrate_and_write(
        spec,
        Path(output_dir),
        mode="run",
        initial_state=None,
        diagnostics_per_save=diagnostics_per_save,
    )


def spinup(
    spec: RunSpec,
    output_dir: str | Path,
    *,
    diagnostics_per_save: int = 1,
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
    )


def restart(
    spec: RunSpec,
    output_dir: str | Path,
    *,
    restart_from: str | Path,
    diagnostics_per_save: int = 1,
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
        diagnostics_per_save: Diagnostic sub-chunks per save interval
            (default 1).
    """
    restart_path = Path(restart_from)
    logger.info("restart loading state from {}", restart_path)
    ds = io.load_dataset(restart_path)
    state0 = io.dataset_to_state(ds)
    return _integrate_and_write(
        spec,
        Path(output_dir),
        mode="restart",
        initial_state=state0,
        diagnostics_per_save=diagnostics_per_save,
    )
