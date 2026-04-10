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
    OutputSpec,
    RunSpec,
    TestCaseSpec,
    TimesteppingSpec,
)
from somax._src.models.swm.nonlinear_2d import NonlinearSW2DState


# pytest tries to collect anything named Test* as a test class.
TestCaseSpec.__test__ = False


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

    The test cases below use this to construct both stable and CFL-violating
    configurations on demand.
    """
    return RunSpec(
        testcase=TestCaseSpec(
            name="baroclinic_instability_swm",
            grid={"nx": nx, "ny": ny, "Lx": 1.0e6, "Ly": 1.0e6},
            consts={"f0": 1.0e-4, "beta": 1.6e-11},
            stratification={
                "H": [500.0, 4500.0],
                "g_prime": [9.81, 0.025],
            },
            params={
                "lateral_viscosity": 100.0,
                "bottom_drag": 1.0e-7,
                "jet_speed": 0.5,
                "jet_width": 5.0e4,
                "perturbation": 0.01,
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
# _build_save_times — Copilot review on PR #71 flagged that the previous
# linspace path silently rescaled the snapshot spacing when save_interval
# did not divide (t1 - t0) evenly. The function now uses arange semantics:
# spacing is exactly save_interval and t1 is appended as the closing
# snapshot if it isn't already a multiple.
# ----------------------------------------------------------------------


class TestBuildSaveTimes:
    def _spec(self, *, t0=0.0, t1=10.0, dt=0.1, save_interval=1.0):
        return RunSpec(
            testcase=TestCaseSpec(
                name="baroclinic_instability_swm",
                grid={"nx": 16, "ny": 16, "Lx": 1.0e6, "Ly": 1.0e6},
                consts={"f0": 1.0e-4, "beta": 1.6e-11},
                stratification={"H": [500.0, 4500.0], "g_prime": [9.81, 0.025]},
                params={
                    "lateral_viscosity": 100.0,
                    "bottom_drag": 1.0e-7,
                    "jet_speed": 0.5,
                    "jet_width": 5.0e4,
                    "perturbation": 0.01,
                },
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
            testcase=TestCaseSpec(name="x"),
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
        assert isinstance(attrs["testcase_name"], str)


# ----------------------------------------------------------------------
# restart() — Copilot review on PR #71 asked us to fail fast when the
# restart artifact's state class doesn't match what the testcase expects,
# instead of crashing deep inside JAX with an inscrutable trace.
# ----------------------------------------------------------------------


class TestRestartStateValidation:
    def test_mismatched_state_class_raises_clear_error(self, tmp_path: Path) -> None:
        # Step 1: write a final_state.zarr from a SWM (multilayer) run.
        spinup_dir = tmp_path / "spinup_swm"
        swm_spec = _swm_jet_spec(t1_seconds=600.0, dt=10.0, nx=16, ny=16)
        _run.spinup(swm_spec, spinup_dir)
        final_state_path = spinup_dir / "final_state.zarr"
        assert final_state_path.is_dir()

        # Step 2: try to restart a *different testcase* (doublegyre_qg's
        # State class has different fields from the multilayer SWM
        # written above).
        prod_dir = tmp_path / "prod_qg"
        wrong_spec = RunSpec(
            testcase=TestCaseSpec(
                name="doublegyre_qg",
                grid={"nx": 16, "ny": 16, "Lx": 1.0e6, "Ly": 1.0e6},
                consts={"f0": 1.0e-4, "beta": 1.6e-11},
                stratification={},
                params={
                    "lateral_viscosity": 100.0,
                    "bottom_drag": 1.0e-7,
                    "wind_amplitude": 1.0e-12,
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
