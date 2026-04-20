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


# Phase-3 (#77) promoted ``double_gyre`` out of stub status; Phase-4/5
# will do the same for the remaining five. Keep the stub list explicit
# so accidentally unstubbing a scenario fails loudly here.
_STUB_SCENARIOS = {
    "north_atlantic",
    "med_sea",
    "gulf_stream",
    "southern_ocean",
    "global_ocean",
}


class TestStubsRaiseWithMeaningfulMessage:
    @pytest.mark.parametrize("name", sorted(_STUB_SCENARIOS))
    def test_stub_build_raises_not_implemented(self, name):
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

    def test_double_gyre_is_not_a_stub(self):
        """Phase 3 populated ``double_gyre``; build must succeed for it."""
        bundle = SCENARIOS["double_gyre"].build(
            {
                "grid": {"nx": 8, "ny": 8, "Lx": 1.0e6, "Ly": 1.0e6},
                "consts": {"f0": 1.0e-4, "beta": 1.6e-11},
                "forcing": {"wind_amplitude": 1.0e-12},
                "initial_condition": {"type": "at_rest"},
            }
        )
        assert bundle.name == "double_gyre"
        assert bundle.geometry.kind == "rectangular"


class TestDoubleGyreForcingKeyValidation:
    """Codex review on PR #100: typos like ``wing_amplitude`` must not
    silently disable forcing. The scenario validates against an
    allowlist at build time."""

    def _base_params(self):
        return {
            "grid": {"nx": 8, "ny": 8, "Lx": 1.0e6, "Ly": 1.0e6},
            "consts": {"f0": 1.0e-4, "beta": 1.6e-11},
            "initial_condition": {"type": "at_rest"},
        }

    def test_typo_in_forcing_key_raises(self):
        params = self._base_params() | {"forcing": {"wing_amplitude": 1.0e-12}}
        with pytest.raises(ValueError, match="wing_amplitude"):
            SCENARIOS["double_gyre"].build(params)

    def test_empty_forcing_block_is_accepted(self):
        """No wind (e.g. jet-instability scenarios) is a valid regime."""
        params = self._base_params() | {"forcing": {}}
        bundle = SCENARIOS["double_gyre"].build(params)
        assert dict(bundle.forcing_params) == {}

    def test_known_keys_are_accepted(self):
        params = self._base_params() | {
            "forcing": {"wind_amplitude": 1.0e-12, "wind_profile": "doublegyre"}
        }
        bundle = SCENARIOS["double_gyre"].build(params)
        assert bundle.forcing_params["wind_amplitude"] == 1.0e-12


class TestScenarioBundleForcingParamsFrozen:
    """Copilot review on PR #100: ``ScenarioBundle`` is
    ``@dataclass(frozen=True)`` but ``forcing_params`` is a dict;
    downstream mutation would defeat the frozen guarantee."""

    def test_forcing_params_is_immutable_mapping(self):
        bundle = SCENARIOS["double_gyre"].build(
            {
                "grid": {"nx": 8, "ny": 8, "Lx": 1.0e6, "Ly": 1.0e6},
                "consts": {"f0": 1.0e-4, "beta": 1.6e-11},
                "forcing": {"wind_amplitude": 1.0e-12},
                "initial_condition": {"type": "at_rest"},
            }
        )
        with pytest.raises(TypeError):
            bundle.forcing_params["wind_amplitude"] = 2.0e-12  # type: ignore[index]

    def test_mutating_source_dict_does_not_leak_into_bundle(self):
        """The bundle takes a defensive copy, so mutating the caller's
        dict after construction doesn't reach the bundle."""
        source = {"wind_amplitude": 1.0e-12}
        bundle = SCENARIOS["double_gyre"].build(
            {
                "grid": {"nx": 8, "ny": 8, "Lx": 1.0e6, "Ly": 1.0e6},
                "consts": {"f0": 1.0e-4, "beta": 1.6e-11},
                "forcing": source,
                "initial_condition": {"type": "at_rest"},
            }
        )
        source["wind_amplitude"] = 999.0
        assert bundle.forcing_params["wind_amplitude"] == 1.0e-12


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
