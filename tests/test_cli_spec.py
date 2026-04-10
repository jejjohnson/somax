"""Tests for somax-sim's RunSpec dataclass."""

from __future__ import annotations

from pathlib import Path

import pytest

from somax._src.cli.spec import (
    DebugSpec,
    RunSpec,
    TestCaseSpec,
    TimesteppingSpec,
    dump_yaml,
    load_yaml,
)


# pytest tries to collect anything named Test* as a test class. TestCaseSpec
# is a dataclass, not a test fixture — opt out of collection.
TestCaseSpec.__test__ = False


def _make_spec(**overrides) -> RunSpec:
    base = dict(
        testcase=TestCaseSpec(
            name="doublegyre_baroclinic_qg",
            grid={"nx": 128, "ny": 128, "Lx": 4.0e6, "Ly": 4.0e6},
            consts={"f0": 9.375e-5, "beta": 1.754e-11, "n_layers": 3},
            stratification={
                "H": [400.0, 1100.0, 2600.0],
                "g_prime": [9.81, 0.025, 0.0125],
            },
            params={
                "lateral_viscosity": 15.0,
                "bottom_drag": 1.0e-7,
                "wind_amplitude": 1.3e-10,
            },
        ),
        timestepping=TimesteppingSpec(
            t0=0.0,
            t1=31536000.0,
            dt=600.0,
            save_interval=86400.0,
        ),
    )
    base.update(overrides)
    return RunSpec(**base)


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------


class TestValidation:
    def test_valid_spec_passes(self) -> None:
        _make_spec().validate()  # no exception

    def test_t1_must_exceed_t0(self) -> None:
        spec = _make_spec(timestepping=TimesteppingSpec(0.0, 0.0, 1.0, 1.0))
        with pytest.raises(ValueError, match=r"t1.*must be > timestepping\.t0"):
            spec.validate()

    def test_dt_must_be_positive(self) -> None:
        spec = _make_spec(timestepping=TimesteppingSpec(0.0, 100.0, 0.0, 10.0))
        with pytest.raises(ValueError, match=r"dt.*must be > 0"):
            spec.validate()

    def test_save_interval_must_be_positive(self) -> None:
        spec = _make_spec(timestepping=TimesteppingSpec(0.0, 100.0, 1.0, 0.0))
        with pytest.raises(ValueError, match=r"save_interval.*must be > 0"):
            spec.validate()

    def test_save_interval_cannot_exceed_window(self) -> None:
        spec = _make_spec(timestepping=TimesteppingSpec(0.0, 100.0, 1.0, 200.0))
        with pytest.raises(ValueError, match="cannot exceed"):
            spec.validate()

    def test_empty_testcase_name_raises(self) -> None:
        spec = _make_spec(
            testcase=TestCaseSpec(name="", grid={"nx": 1, "ny": 1}),
        )
        with pytest.raises(ValueError, match="non-empty string"):
            spec.validate()


# ----------------------------------------------------------------------
# Debug merge
# ----------------------------------------------------------------------


class TestDebugMerge:
    def test_no_debug_returns_self(self) -> None:
        spec = _make_spec()
        assert spec.with_debug_applied() is spec

    def test_grid_override_merges_into_block(self) -> None:
        spec = _make_spec(debug=DebugSpec(testcase={"grid": {"nx": 32, "ny": 32}}))
        merged = spec.with_debug_applied()
        # nx, ny overridden; Lx, Ly preserved.
        assert merged.testcase.grid["nx"] == 32
        assert merged.testcase.grid["ny"] == 32
        assert merged.testcase.grid["Lx"] == 4.0e6
        assert merged.testcase.grid["Ly"] == 4.0e6
        # Original unchanged.
        assert spec.testcase.grid["nx"] == 128

    def test_timestepping_override_partial(self) -> None:
        spec = _make_spec(
            debug=DebugSpec(timestepping={"t1": 86400.0, "save_interval": 600.0})
        )
        merged = spec.with_debug_applied()
        assert merged.timestepping.t1 == 86400.0
        assert merged.timestepping.save_interval == 600.0
        # Untouched fields preserved.
        assert merged.timestepping.t0 == 0.0
        assert merged.timestepping.dt == 600.0

    def test_debug_merge_clears_debug_block(self) -> None:
        spec = _make_spec(debug=DebugSpec(testcase={"grid": {"nx": 32}}))
        merged = spec.with_debug_applied()
        assert merged.debug.testcase == {}
        assert merged.debug.timestepping == {}

    def test_unknown_block_raises(self) -> None:
        spec = _make_spec(debug=DebugSpec(testcase={"nonexistent_block": {"foo": 1}}))
        with pytest.raises(ValueError, match="does not match any TestCaseSpec field"):
            spec.with_debug_applied()

    def test_non_dict_block_raises(self) -> None:
        spec = _make_spec(debug=DebugSpec(testcase={"grid": "not a dict"}))
        with pytest.raises(ValueError, match="merge requires both sides to be dicts"):
            spec.with_debug_applied()


# ----------------------------------------------------------------------
# Round trip — dict / YAML
# ----------------------------------------------------------------------


class TestSerialization:
    def test_to_dict_from_dict_round_trip(self) -> None:
        spec = _make_spec()
        round_tripped = RunSpec.from_dict(spec.to_dict())
        assert round_tripped.to_dict() == spec.to_dict()

    def test_yaml_round_trip(self, tmp_path: Path) -> None:
        spec = _make_spec()
        path = tmp_path / "config.yaml"
        dump_yaml(spec, str(path))
        loaded = load_yaml(str(path))
        assert loaded.to_dict() == spec.to_dict()

    def test_from_dict_missing_required_block_raises(self) -> None:
        with pytest.raises(ValueError, match="missing required block: timestepping"):
            RunSpec.from_dict({"testcase": {"name": "x"}})

    def test_from_dict_optional_blocks_default(self) -> None:
        # No 'output' or 'debug' blocks → defaults applied.
        spec = RunSpec.from_dict(
            {
                "testcase": {"name": "x", "grid": {"nx": 1, "ny": 1}},
                "timestepping": {"t0": 0.0, "t1": 1.0, "dt": 0.1, "save_interval": 0.5},
            }
        )
        assert spec.output.write_snapshots is True
        assert spec.output.write_metrics is True
        assert spec.debug.testcase == {}
        assert spec.debug.timestepping == {}

    def test_load_yaml_rejects_non_mapping_root(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("- just a list\n")
        with pytest.raises(ValueError, match="did not parse as a top-level mapping"):
            load_yaml(str(path))

    def test_load_yaml_runs_validation(self, tmp_path: Path) -> None:
        """Loading should both parse AND validate."""
        path = tmp_path / "invalid.yaml"
        path.write_text(
            "testcase:\n"
            "  name: foo\n"
            "  grid: {nx: 1, ny: 1}\n"
            "timestepping: {t0: 10.0, t1: 5.0, dt: 1.0, save_interval: 1.0}\n"
        )
        with pytest.raises(ValueError, match=r"t1.*must be > timestepping\.t0"):
            load_yaml(str(path))
