"""Registry integrity tests for the Phase-2 models registry (#76).

Mirror of ``tests/test_scenarios_registry.py`` for the models side.
No simulations — only metadata and stub integrity.
"""

from __future__ import annotations

import inspect

import jax.numpy as jnp
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


# Phase-3 (#77) populated the six cartesian models; #73 populated the
# two spherical ones, so nothing in the registry is a stub any more.
_STUB_MODELS: set[str] = set()

_SPHERICAL_MODELS = {"spherical_swm", "spherical_qg"}

_PHASE3_MODELS = EXPECTED_MODELS - _STUB_MODELS - _SPHERICAL_MODELS


def _spherical_bundle():
    """A minimal spherical-cap bundle for the spherical model entries."""
    from somax._src.cli.scenarios._types import (
        Constants,
        ForcingFields,
        Geometry,
        InitialConditionSpec,
        ScenarioBundle,
    )

    return ScenarioBundle(
        name="test_spherical_cap",
        geometry=Geometry(
            kind="spherical_cap",
            nx=16,
            ny=8,
            lon_bounds=(0.0, 360.0),
            lat_bounds=(-70.0, -40.0),
        ),
        constants=Constants(f0=-1.2e-4, beta=1.0e-11),
        forcing=ForcingFields(),
        initial_condition=InitialConditionSpec(type="at_rest"),
    )


class TestNothingIsStubbed:
    def test_no_model_is_still_a_stub(self):
        """#73 was the last one; the list is kept so re-stubbing is loud."""
        assert not _STUB_MODELS

    @pytest.mark.parametrize("name", sorted(_SPHERICAL_MODELS))
    def test_spherical_models_build(self, name):
        """They used to raise NotImplementedError; now they build.

        The bundle is constructed here rather than taken from the
        ``southern_ocean`` scenario, which is still stubbed on the
        scenario side (#79). This keeps the model entry testable
        without waiting for it.
        """
        built = MODELS[name].build(_spherical_bundle(), {})
        assert built.model is not None
        assert built.state0 is not None

    @pytest.mark.parametrize("name", sorted(_SPHERICAL_MODELS))
    def test_spherical_models_reject_a_cartesian_geometry(self, name):
        """Without lon/lat bounds there is no sphere to build on."""
        from somax._src.cli.scenarios import SCENARIOS

        bundle = SCENARIOS["double_gyre"].build(
            {
                "grid": {"nx": 8, "ny": 8, "Lx": 1.0e6, "Ly": 1.0e6},
                "initial_condition": {"type": "at_rest"},
            }
        )
        with pytest.raises(ValueError, match="lon_bounds"):
            MODELS[name].build(bundle, {})

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


class TestSphericalAdaptersHonourTheConfig:
    """The spherical entries dropped several configured values."""

    def bundle(self, *, mask=None, g=9.81):
        from somax._src.cli.scenarios._types import (
            Constants,
            ForcingFields,
            Geometry,
            InitialConditionSpec,
            ScenarioBundle,
        )

        return ScenarioBundle(
            name="test_sphere",
            geometry=Geometry(
                kind="spherical_cap",
                nx=16,
                ny=8,
                lon_bounds=(0.0, 360.0),
                lat_bounds=(-70.0, -40.0),
                mask=mask,
            ),
            constants=Constants(f0=-1.2e-4, beta=1.0e-11, g=g),
            forcing=ForcingFields(),
            initial_condition=InitialConditionSpec(type="at_rest"),
        )

    def test_scenario_gravity_reaches_the_model(self):
        built = MODELS["spherical_swm"].build(self.bundle(g=3.71), {})
        assert float(built.model.consts.gravity) == pytest.approx(3.71)

    def test_the_default_gravity_is_unchanged(self):
        built = MODELS["spherical_swm"].build(self.bundle(), {})
        assert float(built.model.consts.gravity) == pytest.approx(9.81)

    def test_depth_comes_from_model_params(self):
        """``ModelSpec.stratification`` is empty for a single layer."""
        built = MODELS["spherical_swm"].build(self.bundle(), {"params": {"H0": 250.0}})
        assert float(built.model.consts.H0) == pytest.approx(250.0)

    def test_the_initial_state_uses_that_depth(self):
        built = MODELS["spherical_swm"].build(self.bundle(), {"params": {"H0": 250.0}})
        assert float(jnp.max(built.state0.h)) == pytest.approx(250.0)

    def test_a_stratification_block_still_works(self):
        built = MODELS["spherical_swm"].build(
            self.bundle(), {"stratification": {"H0": 400.0}}
        )
        assert float(built.model.consts.H0) == pytest.approx(400.0)

    def test_the_advection_method_is_forwarded(self):
        built = MODELS["spherical_swm"].build(
            self.bundle(), {"params": {"method": "upwind3"}}
        )
        assert built.model.method == "upwind3"

    @pytest.mark.parametrize(
        ("key", "value"), [("cg_tol", 1e-8), ("cg_max_steps", 123)]
    )
    def test_the_qg_solver_knobs_are_forwarded(self, key, value):
        built = MODELS["spherical_qg"].build(self.bundle(), {"params": {key: value}})
        assert getattr(built.model, key) == value

    def test_a_raw_scenario_mask_is_converted(self):
        """``Geometry.mask`` is a bare array; the operators want Mask2D."""
        import numpy as np
        from finitevolx import Mask2D

        wet = np.ones((8, 16), dtype=float)
        wet[2:4, 3:6] = 0.0
        built = MODELS["spherical_swm"].build(self.bundle(mask=jnp.asarray(wet)), {})
        assert isinstance(built.model.mask, Mask2D)

    def test_the_converted_mask_is_ghost_padded(self):
        import numpy as np

        wet = np.ones((8, 16), dtype=float)
        wet[2:4, 3:6] = 0.0
        built = MODELS["spherical_swm"].build(self.bundle(mask=jnp.asarray(wet)), {})
        assert built.model.mask.h.shape == (
            built.model.grid.Ny,
            built.model.grid.Nx,
        )

    def test_a_masked_model_still_evaluates(self):
        """The failure the conversion prevents: operators read mask.u/.v."""
        import numpy as np

        from somax.models import SphericalSWMState

        wet = np.ones((8, 16), dtype=float)
        wet[2:4, 3:6] = 0.0
        built = MODELS["spherical_swm"].build(self.bundle(mask=jnp.asarray(wet)), {})
        model = built.model
        tendency = model.vector_field(0.0, built.state0)
        assert isinstance(tendency, SphericalSWMState)
        assert np.isfinite(np.asarray(tendency.h)).all()

    def test_no_mask_is_still_allowed(self):
        built = MODELS["spherical_swm"].build(self.bundle(), {})
        assert built.model.mask is None


class TestCliStubListMatchesTheRegistry:
    """``list-models`` has its own stub set; it must not go stale."""

    def test_nothing_is_tagged_stub_any_more(self):
        from somax._src.cli.app import _STUB_MODELS as cli_stubs

        assert cli_stubs == frozenset()

    def test_the_two_lists_agree(self):
        from somax._src.cli.app import _STUB_MODELS as cli_stubs

        assert set(cli_stubs) == _STUB_MODELS
