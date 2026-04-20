"""Registry integrity tests for the Phase-2 scenarios registry (#76).

These are pure-logic checks — no model builds, no simulations. They
pin the scaffold so the registry stays structurally valid as
Phases 3-5 populate the stubs.
"""

from __future__ import annotations

import inspect

import pytest

from somax._src.cli.scenarios import (
    SCENARIOS,
    ScenarioEntry,
    get_scenario,
    list_scenarios,
)


EXPECTED_SCENARIOS = {
    "double_gyre",
    "north_atlantic",
    "med_sea",
    "gulf_stream",
    "southern_ocean",
    "global_ocean",
}


class TestScenariosRegistryIntegrity:
    def test_all_scenarios_registered(self):
        assert set(SCENARIOS) == EXPECTED_SCENARIOS

    def test_list_scenarios_is_sorted(self):
        assert list_scenarios() == sorted(EXPECTED_SCENARIOS)

    def test_every_entry_is_scenarioentry(self):
        for name, entry in SCENARIOS.items():
            assert isinstance(entry, ScenarioEntry), (
                f"SCENARIOS[{name!r}] is not a ScenarioEntry"
            )

    def test_entry_name_matches_key(self):
        for name, entry in SCENARIOS.items():
            assert entry.name == name

    def test_entry_has_callable_build(self):
        for name, entry in SCENARIOS.items():
            assert callable(entry.build), f"{name!r} has non-callable build"

    def test_entry_geometry_kind_is_valid(self):
        valid = {"rectangular", "real_basin", "spherical_cap"}
        for name, entry in SCENARIOS.items():
            assert entry.geometry_kind in valid, (
                f"{name!r} has unknown geometry_kind {entry.geometry_kind!r}"
            )


class TestGeometryKindAssignments:
    """Guard the Phase-2 geometry-kind assignments against silent drift."""

    def test_double_gyre_is_rectangular(self):
        assert SCENARIOS["double_gyre"].geometry_kind == "rectangular"

    @pytest.mark.parametrize("name", ["north_atlantic", "med_sea", "gulf_stream"])
    def test_real_basins_are_real_basin(self, name):
        assert SCENARIOS[name].geometry_kind == "real_basin"

    @pytest.mark.parametrize("name", ["southern_ocean", "global_ocean"])
    def test_spherical_scenarios_are_spherical_cap(self, name):
        assert SCENARIOS[name].geometry_kind == "spherical_cap"


class TestStubsRaiseWithMeaningfulMessage:
    @pytest.mark.parametrize("name", sorted(EXPECTED_SCENARIOS))
    def test_build_raises_not_implemented(self, name):
        entry = SCENARIOS[name]
        with pytest.raises(NotImplementedError) as exc_info:
            entry.build({})
        msg = str(exc_info.value)
        # Message must reference the scenario and the blocker/phase so a
        # future reader can trace why the stub is still there.
        assert name in msg, f"stub message for {name!r} lacks its own name"
        assert "phase" in msg.lower() or "blocked" in msg.lower(), (
            f"stub message for {name!r} doesn't mention phase/blocker"
        )


class TestGetScenarioLookup:
    def test_unknown_scenario_raises_keyerror_with_available(self):
        with pytest.raises(KeyError) as exc_info:
            get_scenario("no_such_scenario")
        msg = str(exc_info.value)
        assert "no_such_scenario" in msg
        # Every registered name should appear in the "available" hint.
        for name in EXPECTED_SCENARIOS:
            assert name in msg, f"error message missing registered scenario {name!r}"


class TestBuildSignatureIsUniform:
    """Every scenario's build callable takes a single dict argument."""

    @pytest.mark.parametrize("name", sorted(EXPECTED_SCENARIOS))
    def test_build_signature_has_single_dict_parameter(self, name):
        entry = SCENARIOS[name]
        sig = inspect.signature(entry.build)
        assert len(sig.parameters) == 1, (
            f"{name!r}.build should take exactly one argument (params dict)"
        )
