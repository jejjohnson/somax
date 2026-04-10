"""Tests for the optional preflight + postflight assertions framework."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


pytest.importorskip("xarray")
pytest.importorskip("zarr")

from somax._src.cli import _run
from somax._src.cli._assertions import (
    POSTFLIGHT_ASSERTIONS,
    PREFLIGHT_ASSERTIONS,
    AssertionFailedError,
    check_bounded_metric,
    check_cfl,
    run_postflight,
    run_preflight,
)
from somax._src.cli.spec import (
    DebugSpec,
    OutputSpec,
    RunSpec,
    TestCaseSpec,
    TimesteppingSpec,
)


# pytest collection hint: TestCaseSpec is a dataclass, not a test class.
TestCaseSpec.__test__ = False


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _spec_with_assertions(**assertions: dict[str, Any]) -> RunSpec:
    """Build a minimal valid RunSpec carrying the given assertions."""
    return RunSpec(
        testcase=TestCaseSpec(
            name="doublegyre_qg",
            grid={"nx": 32, "ny": 32, "Lx": 1.0e6, "Ly": 1.0e6},
            consts={"f0": 1.0e-4, "beta": 1.6e-11},
            stratification={},
            params={
                "lateral_viscosity": 100.0,
                "bottom_drag": 1.0e-7,
                "wind_amplitude": 1.0e-12,
            },
        ),
        timestepping=TimesteppingSpec(t0=0.0, t1=600.0, dt=10.0, save_interval=600.0),
        output=OutputSpec(),
        debug=DebugSpec(),
        assertions=dict(assertions),
    )


def _fake_model(dx: float, dy: float | None = None) -> Any:
    """Build a stub object that satisfies the ``model.grid.dx/dy`` interface."""
    if dy is None:
        dy = dx
    return SimpleNamespace(grid=SimpleNamespace(dx=dx, dy=dy))


# ----------------------------------------------------------------------
# CFL preflight assertion — unit tests
# ----------------------------------------------------------------------


class TestCFLAssertion:
    def test_subcritical_cfl_passes(self) -> None:
        spec = _spec_with_assertions()
        model = _fake_model(dx=15625.0)
        # CFL = 221 * 10 / 15625 ≈ 0.14 — well below 0.5.
        check_cfl(spec, model, wave_speed_m_per_s=221.0, max_cfl=0.5)

    def test_at_limit_passes(self) -> None:
        spec = _spec_with_assertions()
        model = _fake_model(dx=4420.0)  # CFL = 221 * 10 / 4420 ≈ 0.5 exactly
        check_cfl(spec, model, wave_speed_m_per_s=221.0, max_cfl=0.5)

    def test_supercritical_cfl_raises(self) -> None:
        spec = _spec_with_assertions()
        model = _fake_model(dx=15625.0)
        # Match the original swm_jet bug: dt=300 → CFL ≈ 4.24
        spec.timestepping.dt = 300.0
        with pytest.raises(AssertionFailedError, match=r"cfl check FAILED"):
            check_cfl(spec, model, wave_speed_m_per_s=221.0, max_cfl=0.5)

    def test_error_message_includes_safe_dt(self) -> None:
        spec = _spec_with_assertions()
        spec.timestepping.dt = 300.0
        model = _fake_model(dx=15625.0)
        with pytest.raises(AssertionFailedError) as exc_info:
            check_cfl(spec, model, wave_speed_m_per_s=221.0, max_cfl=0.5)
        msg = str(exc_info.value)
        assert "wave_speed = 221" in msg
        assert "dt         = 300" in msg
        assert "dx_min     = 15625" in msg
        assert "maximum stable dt at this CFL" in msg

    def test_zero_wave_speed_rejected(self) -> None:
        spec = _spec_with_assertions()
        model = _fake_model(dx=15625.0)
        with pytest.raises(AssertionFailedError, match="must be > 0"):
            check_cfl(spec, model, wave_speed_m_per_s=0.0)

    def test_zero_max_cfl_rejected(self) -> None:
        spec = _spec_with_assertions()
        model = _fake_model(dx=15625.0)
        with pytest.raises(AssertionFailedError, match="max_cfl must be > 0"):
            check_cfl(spec, model, wave_speed_m_per_s=221.0, max_cfl=0.0)

    def test_anisotropic_grid_uses_smaller_dx(self) -> None:
        """min(dx, dy) should drive CFL."""
        spec = _spec_with_assertions()
        # dy is the smaller direction → CFL = 100 * 10 / 1000 = 1.0
        model = _fake_model(dx=10000.0, dy=1000.0)
        with pytest.raises(AssertionFailedError, match="dx_min     = 1000"):
            check_cfl(spec, model, wave_speed_m_per_s=100.0, max_cfl=0.5)

    def test_model_without_grid_raises(self) -> None:
        with pytest.raises(AssertionFailedError, match=r"has no \.grid attribute"):
            check_cfl(SimpleNamespace(), object(), wave_speed_m_per_s=1.0)


# ----------------------------------------------------------------------
# bounded_metric postflight assertion — unit tests
# ----------------------------------------------------------------------


class TestBoundedMetricAssertion:
    def test_value_in_range_passes(self) -> None:
        check_bounded_metric(
            None, {"kinetic_energy": 0.5}, name="kinetic_energy", min=0.0, max=1.0
        )

    def test_value_below_min_raises(self) -> None:
        with pytest.raises(AssertionFailedError, match=r"below min"):
            check_bounded_metric(None, {"ke": -0.1}, name="ke", min=0.0, max=1.0)

    def test_value_above_max_raises(self) -> None:
        with pytest.raises(AssertionFailedError, match=r"above max"):
            check_bounded_metric(None, {"ke": 1.5}, name="ke", min=0.0, max=1.0)

    def test_only_min_specified(self) -> None:
        check_bounded_metric(None, {"x": 100.0}, name="x", min=0.0)
        with pytest.raises(AssertionFailedError, match=r"below min"):
            check_bounded_metric(None, {"x": -1.0}, name="x", min=0.0)

    def test_only_max_specified(self) -> None:
        check_bounded_metric(None, {"x": -100.0}, name="x", max=0.0)
        with pytest.raises(AssertionFailedError, match=r"above max"):
            check_bounded_metric(None, {"x": 1.0}, name="x", max=0.0)

    def test_missing_metric_raises(self) -> None:
        with pytest.raises(
            AssertionFailedError, match=r"metric 'nonexistent' not present"
        ):
            check_bounded_metric(
                None, {"a": 1.0, "b": 2.0}, name="nonexistent", min=0.0
            )

    def test_nan_metric_raises(self) -> None:
        with pytest.raises(AssertionFailedError, match=r"non-finite"):
            check_bounded_metric(None, {"x": float("nan")}, name="x", min=0.0, max=1.0)

    def test_inf_metric_raises(self) -> None:
        with pytest.raises(AssertionFailedError, match=r"non-finite"):
            check_bounded_metric(None, {"x": float("inf")}, name="x", max=1.0)

    def test_non_numeric_metric_raises(self) -> None:
        with pytest.raises(AssertionFailedError, match=r"not a numeric scalar"):
            check_bounded_metric(None, {"x": "not a number"}, name="x", min=0.0)


# ----------------------------------------------------------------------
# run_preflight / run_postflight dispatch
# ----------------------------------------------------------------------


class TestDispatch:
    def test_empty_assertions_is_noop(self) -> None:
        spec = _spec_with_assertions()
        model = _fake_model(dx=10000.0)
        run_preflight(spec, model)  # no exception
        run_postflight(spec, {"x": 1.0})  # no exception

    def test_unknown_preflight_name_raises(self) -> None:
        spec = _spec_with_assertions(unknown_check={})
        model = _fake_model(dx=10000.0)
        with pytest.raises(AssertionFailedError, match=r"unknown assertion"):
            run_preflight(spec, model)

    def test_unknown_postflight_name_raises(self) -> None:
        spec = _spec_with_assertions(unknown_check={})
        with pytest.raises(AssertionFailedError, match=r"unknown assertion"):
            run_postflight(spec, {})

    def test_preflight_skips_postflight_assertions(self) -> None:
        """Postflight names registered in spec are skipped at preflight time."""
        spec = _spec_with_assertions(
            bounded_metric={"name": "x", "min": 0.0},  # would fail at runtime
        )
        model = _fake_model(dx=10000.0)
        run_preflight(spec, model)  # no exception — bounded_metric ignored

    def test_postflight_skips_preflight_assertions(self) -> None:
        """Preflight names are skipped at postflight time."""
        spec = _spec_with_assertions(cfl={"wave_speed_m_per_s": 1e9, "max_cfl": 0.5})
        # No exception because CFL is skipped here.
        run_postflight(spec, {"x": 1.0})

    def test_dispatch_calls_check_with_params(self) -> None:
        spec = _spec_with_assertions(cfl={"wave_speed_m_per_s": 100.0, "max_cfl": 0.5})
        model = _fake_model(dx=10000.0)
        # CFL = 100 * 10 / 10000 = 0.1 → passes.
        run_preflight(spec, model)


# ----------------------------------------------------------------------
# End-to-end: opt-in CFL assertion catches the swm_jet bug at preflight
# ----------------------------------------------------------------------


def _swm_jet_with_cfl_assertion(*, dt: float, t1_seconds: float) -> RunSpec:
    return RunSpec(
        testcase=TestCaseSpec(
            name="baroclinic_instability_swm",
            grid={"nx": 64, "ny": 64, "Lx": 1.0e6, "Ly": 1.0e6},
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
            t0=0.0, t1=t1_seconds, dt=dt, save_interval=t1_seconds
        ),
        output=OutputSpec(),
        debug=DebugSpec(),
        assertions={
            "cfl": {"wave_speed_m_per_s": 221.0, "max_cfl": 0.5},
        },
    )


class TestCFLAssertionEndToEnd:
    def test_cfl_assertion_catches_swm_jet_bug_before_integration(
        self, tmp_path: Path
    ) -> None:
        """The exact original bug (dt=300, 64²) should fail at preflight.

        This must raise BEFORE any integration runs — so the failure is
        cheap (no JAX compilation, no diffrax steps) and unmistakable.
        """
        spec = _swm_jet_with_cfl_assertion(dt=300.0, t1_seconds=86400.0)

        with pytest.raises(AssertionFailedError, match=r"cfl check FAILED"):
            _run.simulate(spec, tmp_path)

        # Nothing should have been written — preflight runs before
        # integration, before any output paths get touched.
        assert not (tmp_path / "snapshots.zarr").exists()
        assert not (tmp_path / "final_state.zarr").exists()
        assert not (tmp_path / "metrics.json").exists()
        assert not (tmp_path / "resolved.yaml").exists()

    def test_cfl_assertion_passes_for_corrected_dt(self, tmp_path: Path) -> None:
        """The fix (dt=20) should pass preflight and complete normally."""
        spec = _swm_jet_with_cfl_assertion(dt=20.0, t1_seconds=600.0)
        result = _run.simulate(spec, tmp_path)
        assert result.snapshots_path is not None
        assert result.final_state_path.is_dir()


# ----------------------------------------------------------------------
# Registries are populated
# ----------------------------------------------------------------------


def test_preflight_registry_has_cfl() -> None:
    assert "cfl" in PREFLIGHT_ASSERTIONS


def test_postflight_registry_has_bounded_metric() -> None:
    assert "bounded_metric" in POSTFLIGHT_ASSERTIONS
