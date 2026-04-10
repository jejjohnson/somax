"""Tests for the authored simulation configs.

Validates that every config in ``configs/_authoring/*.py`` parses
cleanly into a :class:`somax._src.cli.spec.RunSpec`, that the test-case
name resolves against the adapter registry, and that the debug overrides
also produce a valid spec.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


# Make the repo root importable so we can `import configs._authoring`.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs._authoring import (
    doublegyre_bc_qg,
    doublegyre_bt_qg,
    spinup_bc_qg,
    swm_jet,
)
from somax._src.cli._factories import TEST_CASES
from somax._src.cli.spec import RunSpec


# (display_name, raw_dict) pairs — single source of truth for parametrization.
ALL_AUTHORED_CONFIGS = [
    ("swm_jet", swm_jet.SwmJetConfig),
    ("doublegyre_bt_qg", doublegyre_bt_qg.DoubleGyreBTQGConfig),
    ("spinup_bc_qg", spinup_bc_qg.SpinupBCQGConfig),
    ("doublegyre_bc_qg", doublegyre_bc_qg.DoubleGyreBCQGConfig),
]


@pytest.mark.parametrize("name,raw", ALL_AUTHORED_CONFIGS)
def test_authored_config_parses_to_runspec(name: str, raw: dict) -> None:
    """Every authored config must parse and validate as a RunSpec."""
    spec = RunSpec.from_dict(raw)
    spec.validate()
    assert spec.testcase.name in TEST_CASES, (
        f"config {name!r} references unknown testcase {spec.testcase.name!r}; "
        f"available: {sorted(TEST_CASES)}"
    )


@pytest.mark.parametrize("name,raw", ALL_AUTHORED_CONFIGS)
def test_authored_config_debug_merge_validates(name: str, raw: dict) -> None:
    """The debug-merged variant must also be a valid RunSpec."""
    spec = RunSpec.from_dict(raw)
    merged = spec.with_debug_applied()
    merged.validate()


@pytest.mark.parametrize("name,raw", ALL_AUTHORED_CONFIGS)
def test_authored_config_round_trips_via_yaml(
    name: str, raw: dict, tmp_path: Path
) -> None:
    """Authored config → dict → YAML → dict → RunSpec round-trip."""
    from somax._src.cli.spec import dump_yaml, load_yaml

    spec = RunSpec.from_dict(raw)
    yaml_path = tmp_path / f"{name}.yaml"
    dump_yaml(spec, str(yaml_path))
    loaded = load_yaml(str(yaml_path))
    assert loaded.to_dict() == spec.to_dict()


def test_spinup_config_disables_metrics_and_snapshots() -> None:
    """Spinup configs should set output flags to spinup defaults."""
    spec = RunSpec.from_dict(spinup_bc_qg.SpinupBCQGConfig)
    assert spec.output.write_snapshots is False
    assert spec.output.write_metrics is False


def test_production_configs_enable_full_output() -> None:
    """Non-spinup configs should write everything."""
    for cfg_dict in (
        swm_jet.SwmJetConfig,
        doublegyre_bt_qg.DoubleGyreBTQGConfig,
        doublegyre_bc_qg.DoubleGyreBCQGConfig,
    ):
        spec = RunSpec.from_dict(cfg_dict)
        assert spec.output.write_snapshots is True
        assert spec.output.write_metrics is True


def test_build_configs_orchestrator_writes_all_yaml(tmp_path, monkeypatch) -> None:
    """The build_configs.py orchestrator should materialize every config."""
    # Redirect REPO_ROOT to a fresh tmp dir so the test doesn't touch the
    # real configs/simulation/ tree.
    import scripts.build_configs as build_configs

    monkeypatch.setattr(build_configs, "REPO_ROOT", tmp_path)
    rc = build_configs.main()
    assert rc == 0

    out_dir = tmp_path / "configs" / "simulation"
    for name, _ in ALL_AUTHORED_CONFIGS:
        path = out_dir / f"{name}.yaml"
        assert path.exists(), f"build_configs did not write {path}"
        # The header marker should be present at the top.
        head = path.read_text().splitlines()[0]
        assert head.startswith("#")
