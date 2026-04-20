"""Registry integrity tests for the Phase-2 models registry (#76).

Mirror of ``tests/test_scenarios_registry.py`` for the models side.
No simulations — only metadata and stub integrity.
"""

from __future__ import annotations

import inspect

import pytest

from somax._src.cli.models_registry import (
    MODELS,
    ModelEntry,
    SupportFlags,
    get_model,
    list_models,
)


EXPECTED_MODELS = {
    "linear_swm",
    "nonlinear_swm",
    "barotropic_qg",
    "multilayer_nonlinear_swm",
    "multilayer_qg",
    "reparam_multilayer_qg",
    "spherical_swm",
    "spherical_qg",
}


class TestModelsRegistryIntegrity:
    def test_eight_models_registered(self):
        assert set(MODELS) == EXPECTED_MODELS

    def test_list_models_is_sorted(self):
        assert list_models() == sorted(EXPECTED_MODELS)

    def test_every_entry_is_modelentry(self):
        for name, entry in MODELS.items():
            assert isinstance(entry, ModelEntry), (
                f"MODELS[{name!r}] is not a ModelEntry"
            )

    def test_entry_name_matches_key(self):
        for name, entry in MODELS.items():
            assert entry.name == name

    def test_entry_has_callable_build(self):
        for entry in MODELS.values():
            assert callable(entry.build)

    def test_entry_supports_is_supportflags(self):
        for name, entry in MODELS.items():
            assert isinstance(entry.supports, SupportFlags), (
                f"{name!r}.supports is not a SupportFlags"
            )

    def test_entry_family_is_valid(self):
        for name, entry in MODELS.items():
            assert entry.family in {"swm", "qg"}, (
                f"{name!r} has unknown family {entry.family!r}"
            )

    def test_entry_coordinates_is_valid(self):
        for entry in MODELS.values():
            assert entry.coordinates in {"cartesian", "spherical"}

    def test_entry_layers_is_valid(self):
        for entry in MODELS.values():
            assert entry.layers in {1, "multi"}

    def test_spherical_flag_tracks_coordinates(self):
        """``SupportFlags.spherical`` should mirror ``coordinates``."""
        for name, entry in MODELS.items():
            expected = entry.coordinates == "spherical"
            assert entry.supports.spherical is expected, (
                f"{name!r}.supports.spherical ({entry.supports.spherical}) "
                f"doesn't track coordinates ({entry.coordinates!r})"
            )


class TestMaskSupport:
    """Decision D2 in epic #72: linear SWM is deliberately unmasked."""

    def test_linear_swm_does_not_support_masks(self):
        assert MODELS["linear_swm"].supports.masks is False

    @pytest.mark.parametrize(
        "name",
        [
            "nonlinear_swm",
            "barotropic_qg",
            "multilayer_nonlinear_swm",
            "multilayer_qg",
            "reparam_multilayer_qg",
            "spherical_swm",
            "spherical_qg",
        ],
    )
    def test_other_models_support_masks(self, name):
        assert MODELS[name].supports.masks is True, (
            f"{name!r} should set supports.masks=True (see epic #72)"
        )


# Phase-3 (#77) populated the six cartesian models; the two spherical
# models stay stubs until Phase 5 (#73). Keep the stub list explicit so
# accidentally unstubbing a model fails loudly here.
_STUB_MODELS = {"spherical_swm", "spherical_qg"}

_PHASE3_MODELS = EXPECTED_MODELS - _STUB_MODELS


class TestStubsRaiseWithMeaningfulMessage:
    @pytest.mark.parametrize("name", sorted(_STUB_MODELS))
    def test_stub_build_raises_not_implemented(self, name):
        entry = MODELS[name]
        # The stub build takes (bundle, params); we can pass ``None, {}``
        # because it raises before touching either.
        with pytest.raises(NotImplementedError) as exc_info:
            entry.build(None, {})
        msg = str(exc_info.value)
        assert name in msg, f"stub message for {name!r} lacks its own name"
        assert "phase" in msg.lower() or "blocked" in msg.lower(), (
            f"stub message for {name!r} doesn't mention phase/blocker"
        )

    @pytest.mark.parametrize("name", sorted(_PHASE3_MODELS))
    def test_phase3_models_are_not_stubs(self, name):
        """Phase 3 populated every cartesian model; calling build with a
        *valid* bundle must succeed without NotImplementedError."""
        from somax._src.cli.scenarios import SCENARIOS

        bundle = SCENARIOS["double_gyre"].build(
            {
                "grid": {"nx": 8, "ny": 8, "Lx": 1.0e6, "Ly": 1.0e6},
                "consts": {"f0": 1.0e-4, "beta": 1.6e-11},
                "forcing": {"wind_amplitude": 0.0},
                "initial_condition": {"type": "at_rest"},
            }
        )
        params = _phase3_model_params(name)
        built = MODELS[name].build(bundle, params)
        assert built.model is not None
        assert built.state0 is not None


def _phase3_model_params(name: str) -> dict:
    """Minimal valid model-side params for the Phase-3 registry entries."""
    if name in ("multilayer_nonlinear_swm", "multilayer_qg", "reparam_multilayer_qg"):
        return {
            "stratification": {
                "H": [500.0, 4500.0],
                "g_prime": [9.81, 0.025],
            },
            "params": {"lateral_viscosity": 100.0, "bottom_drag": 1.0e-7},
        }
    if name == "linear_swm":
        return {"params": {"H0": 100.0}}
    return {"params": {"lateral_viscosity": 100.0, "bottom_drag": 1.0e-7}}


class TestMultilayerStratificationValidation:
    """Copilot review on PR #100: missing stratification keys raised
    a bare ``KeyError``; the adapter now raises a ``ValueError`` that
    names the missing key + the model so CLI users can trace the
    config error back to the right YAML block."""

    _MULTILAYER_MODELS = (
        "multilayer_nonlinear_swm",
        "multilayer_qg",
        "reparam_multilayer_qg",
    )

    def _bundle(self):
        from somax._src.cli.scenarios import SCENARIOS

        return SCENARIOS["double_gyre"].build(
            {
                "grid": {"nx": 8, "ny": 8, "Lx": 1.0e6, "Ly": 1.0e6},
                "consts": {"f0": 9.375e-5, "beta": 1.754e-11},
                "forcing": {"wind_amplitude": 1.0e-10},
                "initial_condition": {"type": "at_rest"},
            }
        )

    @pytest.mark.parametrize("name", _MULTILAYER_MODELS)
    def test_missing_stratification_block_raises_value_error(self, name):
        with pytest.raises(ValueError) as exc_info:
            MODELS[name].build(
                self._bundle(),
                {"params": {"lateral_viscosity": 10.0, "bottom_drag": 1.0e-7}},
            )
        msg = str(exc_info.value)
        assert name in msg
        assert "'H'" in msg
        assert "'g_prime'" in msg

    @pytest.mark.parametrize("name", _MULTILAYER_MODELS)
    def test_missing_only_g_prime_raises_value_error(self, name):
        with pytest.raises(ValueError, match=r"'g_prime'"):
            MODELS[name].build(
                self._bundle(),
                {
                    "stratification": {"H": [500.0, 4500.0]},
                    "params": {"lateral_viscosity": 10.0, "bottom_drag": 1.0e-7},
                },
            )

    @pytest.mark.parametrize("name", _MULTILAYER_MODELS)
    def test_length_mismatch_raises_value_error(self, name):
        with pytest.raises(ValueError, match="match the number of layers"):
            MODELS[name].build(
                self._bundle(),
                {
                    "stratification": {
                        "H": [500.0, 4500.0],
                        "g_prime": [9.81, 0.025, 0.01],
                    },
                    "params": {"lateral_viscosity": 10.0, "bottom_drag": 1.0e-7},
                },
            )


class TestNonlinearSWMGaussianEddyIC:
    """Copilot review on PR #100: ``gaussian_eddy`` was advertised by
    ``double_gyre`` but no model adapter implemented it. The
    ``nonlinear_swm`` adapter now handles it with a Gaussian bump on
    the thickness field."""

    def test_gaussian_eddy_produces_centered_bump(self):
        from somax._src.cli.scenarios import SCENARIOS

        bundle = SCENARIOS["double_gyre"].build(
            {
                "grid": {"nx": 16, "ny": 16, "Lx": 1.0e6, "Ly": 1.0e6},
                "consts": {"f0": 1.0e-4, "beta": 0.0},
                "forcing": {},
                "initial_condition": {
                    "type": "gaussian_eddy",
                    "params": {"amplitude": 0.5, "sigma": 1.0e5},
                },
            }
        )
        built = MODELS["nonlinear_swm"].build(
            bundle,
            {"params": {"H0": 100.0, "lateral_viscosity": 10.0}},
        )
        import jax.numpy as jnp
        import numpy as np

        h = np.asarray(built.state0.h)
        # Bump sits on top of the H0=100 rest thickness.
        assert h.max() > 100.0
        assert h.min() == pytest.approx(100.0, abs=1e-3)
        # u/v stay at rest.
        assert float(jnp.max(jnp.abs(built.state0.u))) == 0.0
        assert float(jnp.max(jnp.abs(built.state0.v))) == 0.0


class TestGetModelLookup:
    def test_unknown_model_raises_keyerror_with_available(self):
        with pytest.raises(KeyError) as exc_info:
            get_model("no_such_model")
        msg = str(exc_info.value)
        assert "no_such_model" in msg
        for name in EXPECTED_MODELS:
            assert name in msg


class TestBuildSignatureIsUniform:
    """Every model build takes ``(scenario_bundle, params_dict)``."""

    @pytest.mark.parametrize("name", sorted(EXPECTED_MODELS))
    def test_build_signature_has_two_parameters(self, name):
        entry = MODELS[name]
        sig = inspect.signature(entry.build)
        assert len(sig.parameters) == 2, (
            f"{name!r}.build should take (bundle, params) — two args"
        )
