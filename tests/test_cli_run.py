"""Tests for the somax-sim run pipeline.

Focus areas:
- ``_assert_finite_state`` correctly detects NaN/Inf
- ``simulate()`` raises :class:`IntegrationDivergedError` and refuses to
  write artifacts when integration produces non-finite values
- ``simulate()`` writes the expected artifacts and finite metrics for a
  small, well-behaved configuration
- ``spinup`` → ``restart`` chain round-trips state correctly via the
  zarr restart artifact
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest


pytest.importorskip("xarray")
pytest.importorskip("zarr")
pytest.importorskip("cyclopts")

from somax._src.cli import _run
from somax._src.cli._run import (
    IntegrationDivergedError,
    _assert_finite_state,
)
from somax._src.cli.spec import (
    DebugSpec,
    ModelSpec,
    OutputSpec,
    RunSpec,
    ScenarioSpec,
    TimesteppingSpec,
)
from somax._src.models.swm.nonlinear_2d import NonlinearSW2DState


# ----------------------------------------------------------------------
# _assert_finite_state — unit tests
# ----------------------------------------------------------------------


class TestAssertFiniteState:
    def test_clean_state_passes(self) -> None:
        nt, ny, nx = 3, 4, 4
        stacked = NonlinearSW2DState(
            h=jnp.ones((nt, ny, nx)),
            u=jnp.zeros((nt, ny, nx)),
            v=jnp.zeros((nt, ny, nx)),
        )
        _assert_finite_state(stacked, mode="run")  # no exception

    def test_nan_in_one_field_raises(self) -> None:
        nt, ny, nx = 3, 4, 4
        h = jnp.ones((nt, ny, nx))
        h = h.at[1, 0, 0].set(jnp.nan)
        stacked = NonlinearSW2DState(
            h=h,
            u=jnp.zeros((nt, ny, nx)),
            v=jnp.zeros((nt, ny, nx)),
        )
        with pytest.raises(IntegrationDivergedError) as exc_info:
            _assert_finite_state(stacked, mode="run")
        msg = str(exc_info.value)
        assert "h" in msg
        assert "1/" in msg  # 1 non-finite element

    def test_inf_in_one_field_raises(self) -> None:
        stacked = NonlinearSW2DState(
            h=jnp.full((2, 3, 3), jnp.inf),
            u=jnp.zeros((2, 3, 3)),
            v=jnp.zeros((2, 3, 3)),
        )
        with pytest.raises(IntegrationDivergedError, match=r"non-finite"):
            _assert_finite_state(stacked, mode="run")

    def test_first_step_nan_flagged_in_message(self) -> None:
        """When the *first* save-step is already NaN, error explains why."""
        nt, ny, nx = 3, 4, 4
        # Entire first slice is NaN.
        h = jnp.ones((nt, ny, nx))
        h = h.at[0].set(jnp.nan)
        stacked = NonlinearSW2DState(
            h=h,
            u=jnp.zeros((nt, ny, nx)),
            v=jnp.zeros((nt, ny, nx)),
        )
        with pytest.raises(IntegrationDivergedError) as exc_info:
            _assert_finite_state(stacked, mode="run")
        msg = str(exc_info.value)
        assert "already non-finite at the first saved step" in msg
        assert "CFL" in msg

    def test_mode_appears_in_message(self) -> None:
        stacked = NonlinearSW2DState(
            h=jnp.full((2, 3, 3), jnp.nan),
            u=jnp.zeros((2, 3, 3)),
            v=jnp.zeros((2, 3, 3)),
        )
        with pytest.raises(IntegrationDivergedError, match=r"spinup integration"):
            _assert_finite_state(stacked, mode="spinup")


# ----------------------------------------------------------------------
# Helpers for end-to-end tests
# ----------------------------------------------------------------------


def _swm_jet_spec(
    *, t1_seconds: float, dt: float, nx: int = 32, ny: int = 32
) -> RunSpec:
    """Build a small, parameterized swm_jet RunSpec for tests.

    Uses the Phase-3 ``double_gyre`` x ``multilayer_nonlinear_swm`` pair
    (the Phase-3 port of the legacy ``baroclinic_instability_swm`` test
    case). The test cases below use this to construct both stable and
    CFL-violating configurations on demand.
    """
    return RunSpec(
        scenario=ScenarioSpec(
            name="double_gyre",
            grid={"nx": nx, "ny": ny, "Lx": 1.0e6, "Ly": 1.0e6},
            consts={"f0": 1.0e-4, "beta": 1.6e-11},
            forcing={},
            initial_condition={
                "type": "jet",
                "params": {
                    "jet_speed": 0.5,
                    "jet_width": 5.0e4,
                    "perturbation": 0.01,
                },
            },
        ),
        model=ModelSpec(
            name="multilayer_nonlinear_swm",
            stratification={
                "H": [500.0, 4500.0],
                "g_prime": [9.81, 0.025],
            },
            params={
                "lateral_viscosity": 100.0,
                "bottom_drag": 1.0e-7,
            },
        ),
        timestepping=TimesteppingSpec(
            t0=0.0,
            t1=t1_seconds,
            dt=dt,
            save_interval=t1_seconds,
        ),
        output=OutputSpec(write_snapshots=True, write_metrics=True),
        debug=DebugSpec(),
    )


# ----------------------------------------------------------------------
# End-to-end: well-behaved run
# ----------------------------------------------------------------------


class TestSimulateHappyPath:
    def test_short_stable_run_writes_finite_metrics(self, tmp_path: Path) -> None:
        """A short, CFL-respecting run should write artifacts with no NaN."""
        # 1 hour of integration at 32² with dt=10 → CFL ≈ 0.07.
        spec = _swm_jet_spec(t1_seconds=3600.0, dt=10.0, nx=32, ny=32)
        result = _run.simulate(spec, tmp_path)

        # Artifacts exist.
        assert result.snapshots_path is not None
        assert result.snapshots_path.is_dir()
        assert result.final_state_path.is_dir()
        assert result.metrics_path is not None
        assert result.metrics_path.is_file()
        assert (tmp_path / "resolved.yaml").is_file()

        # Metrics are all finite (this is the check that swm_jet at dt=300
        # would have failed).
        with result.metrics_path.open() as f:
            metrics = json.load(f)
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                assert np.isfinite(value), f"metric {key!r} is non-finite: {value}"

    def test_eval_metrics_persisted_for_2d_fluid_run(self, tmp_path: Path) -> None:
        """A single-layer (2D) SWM run writes the ``somax.eval`` metrics into
        ``metrics.json``, and a ``bounded_metric`` postflight assertion can
        gate one of them.

        The happy-path test above uses ``multilayer_nonlinear_swm`` (a 3D
        state), for which ``compute_eval_metrics`` returns ``{}``; this covers
        the 2D fluid case where the eval keys are actually produced. The
        ``bounded_metric`` assertion on ``rms_divergence`` would raise if the
        key were absent, so a clean run also proves the assertion read it.
        """
        spec = RunSpec(
            scenario=ScenarioSpec(
                name="double_gyre",
                grid={"nx": 32, "ny": 32, "Lx": 1.0e6, "Ly": 1.0e6},
                consts={"f0": 1.0e-4, "beta": 1.6e-11},
                forcing={},
                initial_condition={
                    "type": "jet",
                    "params": {
                        "jet_speed": 0.5,
                        "jet_width": 5.0e4,
                        "perturbation": 0.01,
                    },
                },
            ),
            model=ModelSpec(
                name="nonlinear_swm",
                stratification={},
                params={
                    "H0": 1000.0,
                    "lateral_viscosity": 100.0,
                    "bottom_drag": 1.0e-7,
                },
            ),
            timestepping=TimesteppingSpec(
                t0=0.0, t1=3600.0, dt=10.0, save_interval=3600.0
            ),
            output=OutputSpec(write_snapshots=False, write_metrics=True),
            debug=DebugSpec(),
            assertions={"bounded_metric": {"name": "rms_divergence", "min": 0.0}},
        )
        result = _run.simulate(spec, tmp_path)

        assert result.metrics_path is not None
        with result.metrics_path.open() as f:
            metrics = json.load(f)
        for key in (
            "rms_divergence",
            "total_enstrophy",
            "kinetic_energy",
            "geostrophic_imbalance",
        ):
            assert key in metrics, (
                f"missing eval metric {key!r}; have {sorted(metrics)}"
            )
            assert np.isfinite(metrics[key]), f"eval metric {key!r} non-finite"
        # The jet carries real flow, so kinetic energy is strictly positive
        # (guards against a degenerate all-zero metric path).
        assert metrics["kinetic_energy"] > 0.0


# ----------------------------------------------------------------------
# End-to-end: CFL violation
# ----------------------------------------------------------------------


class TestSimulateCFLBlowup:
    def test_supercritical_dt_raises_and_writes_nothing(self, tmp_path: Path) -> None:
        """The CFL-violating case must raise BEFORE writing any artifacts.

        Reproduces the original swm_jet bug: 64² grid with dt=300 over a
        long enough integration that the CFL violation actually blows up.
        H_total=5000m → c≈221 m/s, dx=1e6/64≈15.6 km, CFL ≈ 4.24 (well
        above the stability threshold).
        """
        # 4 hours at dt=300 = 48 steps, which is plenty for the supercritical
        # CFL to drive everything to NaN.
        spec = _swm_jet_spec(t1_seconds=4 * 3600.0, dt=300.0, nx=64, ny=64)

        with pytest.raises(IntegrationDivergedError) as exc_info:
            _run.simulate(spec, tmp_path)

        # The error message should mention the CFL diagnosis.
        msg = str(exc_info.value)
        assert "non-finite" in msg
        assert "CFL" in msg

        # The diverged-integration check runs BEFORE write_snapshots,
        # final_state, metrics. None of those artifacts should exist.
        assert not (tmp_path / "snapshots.zarr").exists()
        assert not (tmp_path / "final_state.zarr").exists()
        assert not (tmp_path / "metrics.json").exists()


# ----------------------------------------------------------------------
# End-to-end: spinup → restart chain via zarr
# ----------------------------------------------------------------------


class TestRunLogIntegration:
    def test_simulate_always_writes_run_log_file(self, tmp_path: Path) -> None:
        """Every simulate() call should create run.log under the output dir."""
        spec = _swm_jet_spec(t1_seconds=600.0, dt=10.0, nx=16, ny=16)
        result = _run.simulate(spec, tmp_path)
        log_path = tmp_path / "run.log"
        assert log_path.is_file()
        content = log_path.read_text()
        # Loguru-formatted lines all carry the label tag.
        assert "somax-sim/run" in content
        # The "integration starting" line is at the top.
        assert "integration starting" in content
        # The chunked path emits per-chunk lines.
        assert "chunk 0/" in content
        # And the "finished cleanly" final line lands at the bottom.
        assert "finished cleanly" in content
        # Other artifacts still get written.
        assert result.snapshots_path is not None

    def test_simulate_emits_physical_diagnostics(self, tmp_path: Path) -> None:
        """Each chunk line should include physics scalars (energy/enstrophy)."""
        spec = _swm_jet_spec(t1_seconds=600.0, dt=10.0, nx=16, ny=16)
        _run.simulate(spec, tmp_path)
        content = (tmp_path / "run.log").read_text()
        # The diagnose() output for the multilayer SWM exposes
        # total_energy / total_enstrophy — both should appear at least
        # once in the run.log lines (chunk 0 line at minimum).
        assert "total_energy=" in content
        assert "total_enstrophy=" in content

    def test_diagnostics_per_save_increases_chunk_lines(self, tmp_path: Path) -> None:
        """diagnostics_per_save=4 should give 4x more chunk-N/M lines."""
        spec = _swm_jet_spec(t1_seconds=600.0, dt=10.0, nx=16, ny=16)
        out_n1 = tmp_path / "n1"
        out_n4 = tmp_path / "n4"
        result_n1 = _run.simulate(spec, out_n1, diagnostics_per_save=1)
        result_n4 = _run.simulate(spec, out_n4, diagnostics_per_save=4)

        def _count_chunk_lines(p: Path) -> int:
            return sum(1 for line in p.read_text().splitlines() if "| chunk " in line)

        c_n1 = _count_chunk_lines(out_n1 / "run.log")
        c_n4 = _count_chunk_lines(out_n4 / "run.log")
        # The helper uses save_interval == t1, so there is exactly one
        # save interval. n=1 → 2 chunk lines (chunk 0 + chunk 1).
        # n=4 → 5 chunk lines (chunk 0 + chunks 1..4). At minimum n=4
        # must produce 3 more chunk lines than n=1.
        assert c_n4 - c_n1 >= 3, (c_n1, c_n4)
        # Snapshot count is unchanged — diagnostics knob does not
        # affect what gets persisted.
        assert result_n1.snapshots_path is not None
        assert result_n4.snapshots_path is not None


class TestSpinupRestartChain:
    def test_spinup_then_restart_round_trips_state(self, tmp_path: Path) -> None:
        """Spinup writes final_state.zarr; restart loads it and continues."""
        spinup_dir = tmp_path / "spinup"
        prod_dir = tmp_path / "prod"

        # Step 1: spinup. write_snapshots is automatically False for spinup.
        spinup_spec = _swm_jet_spec(t1_seconds=600.0, dt=10.0, nx=16, ny=16)
        _run.spinup(spinup_spec, spinup_dir)

        assert (spinup_dir / "final_state.zarr").is_dir()
        # Spinup mode skips snapshots and metrics — see _integrate_and_write.
        assert not (spinup_dir / "snapshots.zarr").exists()
        assert not (spinup_dir / "metrics.json").exists()

        # Step 2: restart from the spinup's final_state.
        prod_spec = _swm_jet_spec(t1_seconds=600.0, dt=10.0, nx=16, ny=16)
        result = _run.restart(
            prod_spec, prod_dir, restart_from=spinup_dir / "final_state.zarr"
        )

        # Production stage writes the full artifact set.
        assert result.snapshots_path is not None
        assert result.snapshots_path.is_dir()
        assert result.final_state_path.is_dir()
        assert result.metrics_path is not None

        # Metrics should be all finite.
        with result.metrics_path.open() as f:
            metrics = json.load(f)
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                assert np.isfinite(value), f"restart metric {key!r} non-finite: {value}"


# ----------------------------------------------------------------------
# Crash-recovery checkpointing (#70)
# ----------------------------------------------------------------------


def _swm_jet_spec_with_saves(
    *,
    t1_seconds: float,
    dt: float,
    save_interval: float,
    nx: int = 16,
    ny: int = 16,
) -> RunSpec:
    """Variant of ``_swm_jet_spec`` with a controllable save_interval.

    The default helper sets ``save_interval == t1`` (one saved interval →
    one snapshot ordinal), which is enough for happy-path tests but too
    coarse to exercise the crash-recovery cadence. The checkpoint is
    written at snapshot-aligned chunk endpoints, so we need
    ``save_interval < t1`` to see more than one ordinal tick.
    """
    spec = _swm_jet_spec(t1_seconds=t1_seconds, dt=dt, nx=nx, ny=ny)
    spec.timestepping.save_interval = save_interval
    spec.validate()
    return spec


class TestCrashRecoveryCheckpointing:
    def test_checkpoint_zarr_not_written_when_disabled(self, tmp_path: Path) -> None:
        """Default (checkpoint_every_n_chunks=0) must not create checkpoint.zarr."""
        spec = _swm_jet_spec_with_saves(t1_seconds=600.0, dt=10.0, save_interval=200.0)
        _run.simulate(spec, tmp_path)
        assert not (tmp_path / "checkpoint.zarr").exists()

    def test_checkpoint_zarr_written_when_enabled(self, tmp_path: Path) -> None:
        """Opting in writes checkpoint.zarr with the sim_t attr."""
        # Three saved intervals → snapshot ordinals 1, 2, 3.
        spec = _swm_jet_spec_with_saves(t1_seconds=600.0, dt=10.0, save_interval=200.0)
        _run.simulate(spec, tmp_path, checkpoint_every_n_chunks=1)

        ckpt_path = tmp_path / "checkpoint.zarr"
        assert ckpt_path.is_dir(), "checkpoint.zarr should have been written"

        # The rolling checkpoint is overwritten each chunk; the final
        # write corresponds to the last snapshot ordinal, i.e. sim_t=t1.
        from somax import io as somax_io

        ds = somax_io.load_dataset(ckpt_path)
        assert "somax_sim_t" in ds.attrs, (
            "checkpoint.zarr missing the somax_sim_t attribute"
        )
        assert float(ds.attrs["somax_sim_t"]) == pytest.approx(600.0)
        assert "somax_snapshot_ordinal" in ds.attrs
        assert "somax_checkpoint_of" in ds.attrs
        assert ds.attrs["somax_checkpoint_of"] == "double_gyre/multilayer_nonlinear_swm"

    def test_checkpoint_cadence_n_equals_2(self, tmp_path: Path) -> None:
        """With every_n=2, only even-ordinal snapshots overwrite the file.

        This is verifiable from the final checkpoint's sim_t: with four
        snapshot ordinals (t=150, 300, 450, 600) and ``every_n=2``, only
        ordinals 2 and 4 trigger a write, so the surviving file should
        carry ``somax_sim_t=600`` (the last write) and ordinal ``4``.
        """
        spec = _swm_jet_spec_with_saves(t1_seconds=600.0, dt=10.0, save_interval=150.0)
        _run.simulate(spec, tmp_path, checkpoint_every_n_chunks=2)

        from somax import io as somax_io

        ds = somax_io.load_dataset(tmp_path / "checkpoint.zarr")
        assert int(ds.attrs["somax_snapshot_ordinal"]) == 4
        assert float(ds.attrs["somax_sim_t"]) == pytest.approx(600.0)

    def test_restart_rebases_t0_from_checkpoint(self, tmp_path: Path) -> None:
        """``restart --from checkpoint.zarr`` resumes at the checkpoint's sim_t.

        We pick an early snapshot (``every_n=1`` on a 2-interval run),
        then feed the resulting ``checkpoint.zarr`` back into ``restart``
        with a wider window. The rebased t0 should equal the checkpoint
        sim_t — we observe this by asserting the effective integration
        window in run.log.
        """
        # Stage 1: short run to emit a checkpoint at sim_t=200.
        stage1_dir = tmp_path / "stage1"
        stage1_spec = _swm_jet_spec_with_saves(
            t1_seconds=200.0, dt=10.0, save_interval=200.0
        )
        _run.simulate(stage1_spec, stage1_dir, checkpoint_every_n_chunks=1)

        stage1_ckpt = stage1_dir / "checkpoint.zarr"
        assert stage1_ckpt.is_dir()

        from somax import io as somax_io

        ds = somax_io.load_dataset(stage1_ckpt)
        assert float(ds.attrs["somax_sim_t"]) == pytest.approx(200.0)

        # Stage 2: restart with a window that *starts* at 0 but whose t1
        # is beyond the checkpoint's sim_t. The runner should rebase t0
        # to 200 internally.
        stage2_dir = tmp_path / "stage2"
        stage2_spec = _swm_jet_spec_with_saves(
            t1_seconds=600.0, dt=10.0, save_interval=200.0
        )
        result = _run.restart(stage2_spec, stage2_dir, restart_from=stage1_ckpt)

        log_text = (stage2_dir / "run.log").read_text()
        # The ``integration starting:`` line records the effective t0
        # for this run; after rebase it must read ``t0=200 s (...)``.
        # Using a regex anchored on that exact line avoids false
        # positives from unrelated "200" occurrences (e.g.
        # ``save_interval=200``).
        import re

        m = re.search(r"integration starting:[^\n]*\bt0=(\d+(?:\.\d+)?) s\b", log_text)
        assert m is not None, (
            f"'integration starting:' line with a t0=<seconds> field not "
            f"found in run.log:\n{log_text}"
        )
        t0_logged = float(m.group(1))
        assert t0_logged == pytest.approx(200.0), (
            f"expected rebased t0=200 s in run.log, got t0={t0_logged}"
        )
        # Belt-and-braces: the original spec.t0 (0 s) should NOT appear
        # as the starting t0 on that same line.
        assert re.search(r"integration starting:[^\n]*\bt0=0 s\b", log_text) is None
        assert result.final_state_path.is_dir()

    def test_restart_from_legacy_final_state_is_unchanged(self, tmp_path: Path) -> None:
        """A ``final_state.zarr`` with no ``somax_sim_t`` attr must behave
        exactly like before: t0 comes from the spec, not the artifact.
        """
        # Stage 1: spinup writes final_state.zarr WITHOUT the sim_t attr
        # (checkpointing disabled).
        stage1_dir = tmp_path / "spinup"
        stage1_spec = _swm_jet_spec_with_saves(
            t1_seconds=600.0, dt=10.0, save_interval=600.0
        )
        _run.spinup(stage1_spec, stage1_dir)
        final_state = stage1_dir / "final_state.zarr"
        assert final_state.is_dir()
        assert not (stage1_dir / "checkpoint.zarr").exists()

        from somax import io as somax_io

        ds = somax_io.load_dataset(final_state)
        assert "somax_sim_t" not in ds.attrs

        # Stage 2: restart from final_state.zarr — no rebase, integration
        # runs the full spec window.
        stage2_dir = tmp_path / "prod"
        stage2_spec = _swm_jet_spec_with_saves(
            t1_seconds=600.0, dt=10.0, save_interval=600.0
        )
        result = _run.restart(stage2_spec, stage2_dir, restart_from=final_state)
        assert result.final_state_path.is_dir()

    def test_rerun_removes_stale_checkpoint_from_prior_run(
        self, tmp_path: Path
    ) -> None:
        """Re-running into an output dir that holds an old checkpoint must
        wipe that checkpoint (otherwise a later ``restart --from
        <dir>/checkpoint.zarr`` would silently resume from the prior,
        unrelated run — the Codex review concern on PR #98).
        """
        # Stage 1: emit a checkpoint at sim_t=200.
        stage1_spec = _swm_jet_spec_with_saves(
            t1_seconds=200.0, dt=10.0, save_interval=200.0
        )
        _run.simulate(stage1_spec, tmp_path, checkpoint_every_n_chunks=1)
        assert (tmp_path / "checkpoint.zarr").is_dir()

        # Stage 2: re-run into the SAME directory with checkpointing OFF.
        # The stale checkpoint.zarr must be gone afterwards — otherwise a
        # user resuming from <tmp_path>/checkpoint.zarr would pick up the
        # prior run's state.
        stage2_spec = _swm_jet_spec_with_saves(
            t1_seconds=200.0, dt=10.0, save_interval=200.0
        )
        _run.simulate(stage2_spec, tmp_path)
        assert not (tmp_path / "checkpoint.zarr").exists(), (
            "stale checkpoint.zarr survived a subsequent checkpointing-off run"
        )

    def test_negative_checkpoint_cadence_raises(self, tmp_path: Path) -> None:
        """Negative N must fail loudly up front (not silently disable)."""
        spec = _swm_jet_spec_with_saves(t1_seconds=200.0, dt=10.0, save_interval=200.0)
        with pytest.raises(ValueError, match=r">= 0"):
            _run.simulate(spec, tmp_path, checkpoint_every_n_chunks=-1)

    def test_restart_with_non_numeric_sim_t_attr_raises(self, tmp_path: Path) -> None:
        """A hand-edited / corrupted ``somax_sim_t`` attr must raise a
        clear ``ValueError`` rather than a cryptic ``TypeError``.
        """
        # Stage 1: produce a valid checkpoint, then corrupt its attr.
        stage1_dir = tmp_path / "stage1"
        stage1_spec = _swm_jet_spec_with_saves(
            t1_seconds=200.0, dt=10.0, save_interval=200.0
        )
        _run.simulate(stage1_spec, stage1_dir, checkpoint_every_n_chunks=1)
        ckpt = stage1_dir / "checkpoint.zarr"

        # Overwrite somax_sim_t with a non-numeric value by rewriting the
        # zarr store via xarray (the attr lives on the root group).
        import xarray as xr

        ds = xr.open_zarr(ckpt, consolidated=False).load()
        ds.attrs["somax_sim_t"] = "not-a-number"
        # Rewrite in-place.
        import shutil

        shutil.rmtree(ckpt)
        ds.to_zarr(ckpt, mode="w", zarr_format=3, consolidated=False)

        stage2_dir = tmp_path / "stage2"
        stage2_spec = _swm_jet_spec_with_saves(
            t1_seconds=600.0, dt=10.0, save_interval=200.0
        )
        with pytest.raises(ValueError, match=r"non-numeric somax_sim_t"):
            _run.restart(stage2_spec, stage2_dir, restart_from=ckpt)

    def test_restart_from_checkpoint_at_t1_raises(self, tmp_path: Path) -> None:
        """A checkpoint whose sim_t >= spec.t1 leaves nothing to integrate."""
        # Stage 1: checkpoint at t=600.
        stage1_dir = tmp_path / "stage1"
        stage1_spec = _swm_jet_spec_with_saves(
            t1_seconds=600.0, dt=10.0, save_interval=600.0
        )
        _run.simulate(stage1_spec, stage1_dir, checkpoint_every_n_chunks=1)

        # Stage 2: restart spec also has t1=600 → rebase would make
        # the window empty. Must raise.
        stage2_dir = tmp_path / "stage2"
        stage2_spec = _swm_jet_spec_with_saves(
            t1_seconds=600.0, dt=10.0, save_interval=600.0
        )
        with pytest.raises(ValueError, match="nothing left"):
            _run.restart(
                stage2_spec, stage2_dir, restart_from=stage1_dir / "checkpoint.zarr"
            )


# ----------------------------------------------------------------------
# _build_save_times — Copilot review on PR #71 flagged that the previous
# linspace path silently rescaled the snapshot spacing when save_interval
# did not divide (t1 - t0) evenly. The function now uses arange semantics:
# spacing is exactly save_interval and t1 is appended as the closing
# snapshot if it isn't already a multiple.
# ----------------------------------------------------------------------


class TestBuildSaveTimes:
    def _spec(self, *, t0=0.0, t1=10.0, dt=0.1, save_interval=1.0):
        return RunSpec(
            scenario=ScenarioSpec(
                name="double_gyre",
                grid={"nx": 16, "ny": 16, "Lx": 1.0e6, "Ly": 1.0e6},
                consts={"f0": 1.0e-4, "beta": 1.6e-11},
                forcing={},
                initial_condition={
                    "type": "jet",
                    "params": {
                        "jet_speed": 0.5,
                        "jet_width": 5.0e4,
                        "perturbation": 0.01,
                    },
                },
            ),
            model=ModelSpec(
                name="multilayer_nonlinear_swm",
                stratification={"H": [500.0, 4500.0], "g_prime": [9.81, 0.025]},
                params={"lateral_viscosity": 100.0, "bottom_drag": 1.0e-7},
            ),
            timestepping=TimesteppingSpec(
                t0=t0, t1=t1, dt=dt, save_interval=save_interval
            ),
            output=OutputSpec(write_snapshots=True, write_metrics=True),
            debug=DebugSpec(),
        )

    def test_evenly_dividing_window_matches_legacy_shape(self) -> None:
        ts = np.asarray(
            _run._build_save_times(self._spec(t0=0.0, t1=10.0, save_interval=1.0))
        )
        # 11 snapshots: t0, t0+SI, ..., t1.
        assert ts.shape == (11,)
        np.testing.assert_array_almost_equal(ts, np.arange(11.0))

    def test_uneven_window_appends_t1_as_final_snapshot(self) -> None:
        # Window=10, save_interval=6. Legacy linspace would have produced
        # 3 snapshots at spacing 5 (silently rescaled). The arange path
        # produces snapshots at t0, t0+6, then appends t1=10 as the
        # closing snapshot. The first interval is exactly 6 — the
        # *requested* cadence — and only the trailing interval is shorter.
        ts = np.asarray(
            _run._build_save_times(self._spec(t0=0.0, t1=10.0, save_interval=6.0))
        )
        np.testing.assert_array_almost_equal(ts, np.asarray([0.0, 6.0, 10.0]))
        # The first save_interval is exactly 6 (the requested cadence).
        assert float(ts[1] - ts[0]) == 6.0

    def test_monthly_over_year_yields_12_full_intervals_plus_closing(self) -> None:
        # The motivating real config: 1 Julian year, monthly cadence.
        # 365.25 / 30 = 12.175, so we get 12 complete monthly snapshots
        # plus a final t1 snapshot for the partial 13th interval.
        year = 365.25 * 86400.0
        month = 30 * 86400.0
        ts = np.asarray(
            _run._build_save_times(self._spec(t0=0.0, t1=year, save_interval=month))
        )
        # 0, 1*30d, 2*30d, ..., 12*30d, plus t1.
        assert ts.shape == (14,)
        for i in range(13):
            assert float(ts[i]) == i * month
        assert float(ts[-1]) == year

    def test_only_final_returns_endpoints(self) -> None:
        ts = np.asarray(
            _run._build_save_times(self._spec(t0=0.0, t1=10.0), only_final=True)
        )
        np.testing.assert_array_almost_equal(ts, np.asarray([0.0, 10.0]))


# ----------------------------------------------------------------------
# _attrs_for — Copilot review on PR #71 asked for native numeric types
# in zarr attrs (no str() wrappers).
# ----------------------------------------------------------------------


class TestAttrsForNumericMetadata:
    def test_attrs_use_native_floats(self) -> None:
        spec = RunSpec(
            scenario=ScenarioSpec(name="double_gyre"),
            model=ModelSpec(name="barotropic_qg"),
            timestepping=TimesteppingSpec(
                t0=0.0, t1=600.0, dt=10.0, save_interval=600.0
            ),
        )
        attrs = _run._attrs_for(spec, mode="run")
        assert attrs["t0"] == 0.0 and isinstance(attrs["t0"], float)
        assert attrs["t1"] == 600.0 and isinstance(attrs["t1"], float)
        assert attrs["dt"] == 10.0 and isinstance(attrs["dt"], float)
        assert attrs["save_interval"] == 600.0 and isinstance(
            attrs["save_interval"], float
        )
        # Strings still belong to the string fields.
        assert isinstance(attrs["somax_sim_mode"], str)
        assert attrs["scenario_name"] == "double_gyre"
        assert attrs["model_name"] == "barotropic_qg"


# ----------------------------------------------------------------------
# restart() — Copilot review on PR #71 asked us to fail fast when the
# restart artifact's state class doesn't match what the scenario x model
# pair expects, instead of crashing deep inside JAX with an inscrutable
# trace.
# ----------------------------------------------------------------------


class TestRestartStateValidation:
    def test_mismatched_state_class_raises_clear_error(self, tmp_path: Path) -> None:
        # Step 1: write a final_state.zarr from a SWM (multilayer) run.
        spinup_dir = tmp_path / "spinup_swm"
        swm_spec = _swm_jet_spec(t1_seconds=600.0, dt=10.0, nx=16, ny=16)
        _run.spinup(swm_spec, spinup_dir)
        final_state_path = spinup_dir / "final_state.zarr"
        assert final_state_path.is_dir()

        # Step 2: try to restart a *different model* (BarotropicQG's
        # State class has different fields from the multilayer SWM
        # written above).
        prod_dir = tmp_path / "prod_qg"
        wrong_spec = RunSpec(
            scenario=ScenarioSpec(
                name="double_gyre",
                grid={"nx": 16, "ny": 16, "Lx": 1.0e6, "Ly": 1.0e6},
                consts={"f0": 1.0e-4, "beta": 1.6e-11},
                forcing={"wind_amplitude": 1.0e-12, "wind_profile": "doublegyre"},
                initial_condition={"type": "at_rest"},
            ),
            model=ModelSpec(
                name="barotropic_qg",
                stratification={},
                params={
                    "lateral_viscosity": 100.0,
                    "bottom_drag": 1.0e-7,
                },
            ),
            timestepping=TimesteppingSpec(
                t0=0.0, t1=600.0, dt=10.0, save_interval=600.0
            ),
            output=OutputSpec(write_snapshots=True, write_metrics=True),
            debug=DebugSpec(),
        )
        # The mismatch is detected as a missing/extra State field at
        # dataset_to_state time, with a clear error pointing the user at
        # the wrong --from store. The error type may be either ValueError
        # (missing field) or TypeError (isinstance check), depending on
        # which State classes are involved — both are clearer than the
        # legacy deep-JAX trace.
        with pytest.raises((ValueError, TypeError)):
            _run.restart(wrong_spec, prod_dir, restart_from=final_state_path)


# ----------------------------------------------------------------------
# Monitors + manifest (observability) — integration (auto-marked slow)
# ----------------------------------------------------------------------


class TestMonitorWiring:
    def test_default_monitors_preserve_clean_run(self, tmp_path: Path) -> None:
        """Default monitors don't change a well-behaved run's outcome."""
        spec = _swm_jet_spec(t1_seconds=600.0, dt=10.0, nx=16, ny=16)
        result = _run.simulate(spec, tmp_path)
        assert (tmp_path / "metrics.json").exists()
        assert result.metrics_path is not None

    def test_custom_monitor_can_terminate(self, tmp_path: Path) -> None:
        """A user monitor requesting termination aborts and writes nothing."""
        from somax.monitor import BaseMonitor, MonitorVerdict, default_monitors

        class _Stop(BaseMonitor):
            name = "stop"

            def on_chunk_end(self, model, state, info):
                return MonitorVerdict(terminate=True, reason="forced stop")

        spec = _swm_jet_spec(t1_seconds=600.0, dt=10.0, nx=16, ny=16)
        with pytest.raises(IntegrationDivergedError, match="forced stop"):
            _run.simulate(spec, tmp_path, monitors=[*default_monitors(), _Stop()])
        assert not (tmp_path / "metrics.json").exists()
        assert not (tmp_path / "final_state.zarr").exists()

    def test_cfl_blowup_raises_integration_diverged(self, tmp_path: Path) -> None:
        """An in-JIT guard blow-up still surfaces as IntegrationDivergedError."""
        spec = _swm_jet_spec(t1_seconds=4 * 3600.0, dt=300.0, nx=64, ny=64)
        with pytest.raises(IntegrationDivergedError) as exc_info:
            _run.simulate(spec, tmp_path)
        msg = str(exc_info.value)
        assert "non-finite" in msg
        assert "CFL" in msg
        assert not (tmp_path / "final_state.zarr").exists()


class TestManifest:
    def test_manifest_written_with_provenance(self, tmp_path: Path) -> None:
        spec = _swm_jet_spec(t1_seconds=600.0, dt=10.0, nx=16, ny=16)
        _run.simulate(spec, tmp_path)
        manifest_path = tmp_path / "manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        for key in (
            "jax_version",
            "somax_version",
            "x64_enabled",
            "platform",
            "config_sha256",
            "dt",
            "t0",
            "t1",
        ):
            assert key in manifest
        assert len(manifest["config_sha256"]) == 64

    def test_config_hash_stable_across_runs(self, tmp_path: Path) -> None:
        spec = _swm_jet_spec(t1_seconds=600.0, dt=10.0, nx=16, ny=16)
        _run.simulate(spec, tmp_path / "a")
        _run.simulate(spec, tmp_path / "b")
        ha = json.loads((tmp_path / "a" / "manifest.json").read_text())["config_sha256"]
        hb = json.loads((tmp_path / "b" / "manifest.json").read_text())["config_sha256"]
        assert ha == hb
