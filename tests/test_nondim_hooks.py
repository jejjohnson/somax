"""Tests for the DA / IO / CLI hooks that carry a StateAffine.

Three surfaces, one idea: a transform that says what each field's
magnitude is should be usable wherever a flat vector, an exported
dataset, or a config names state.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from somax._src.core.scales import Scales
from somax._src.core.transforms import StateAffine
from somax._src.da.flatten import make_ensemble, state_to_vector
from somax._src.io.xarray import (
    SCALES_ATTR_PREFIX,
    apply_scale_metadata,
    scales_attrs,
    state_to_dataset,
    transform_attrs,
)
from somax._src.models.swm.nonlinear_2d import NonlinearSW2DState


NY, NX = 6, 5


@pytest.fixture
def scales():
    return Scales.advective(L=1e6, U=0.1, f0=1e-4, H=1000.0)


@pytest.fixture
def state(scales):
    rng = np.random.RandomState(0)
    return NonlinearSW2DState(
        h=jnp.asarray(scales.H + rng.randn(NY, NX)),
        u=jnp.asarray(0.1 * rng.randn(NY, NX)),
        v=jnp.asarray(0.1 * rng.randn(NY, NX)),
    )


@pytest.fixture
def transform(scales):
    return StateAffine.from_scales(NonlinearSW2DState, scales)


class TestStateToVector:
    def test_without_a_transform_is_unchanged(self, state):
        vector, unravel = state_to_vector(state)
        recovered = unravel(vector)
        np.testing.assert_allclose(recovered.h, state.h)

    def test_with_a_transform_round_trips(self, state, transform):
        vector, unravel = state_to_vector(state, transform)
        recovered = unravel(vector)
        for field in ("h", "u", "v"):
            np.testing.assert_allclose(
                getattr(recovered, field), getattr(state, field), rtol=1e-5
            )

    def test_the_vector_is_in_transformed_coordinates(self, state, transform):
        vector, _ = state_to_vector(state, transform)
        expected, _ = state_to_vector(transform.forward(state))
        np.testing.assert_allclose(vector, expected, rtol=1e-6)

    def test_the_transform_actually_rescales(self, state, transform):
        """Otherwise the round-trip test would be vacuous."""
        plain, _ = state_to_vector(state)
        scaled, _ = state_to_vector(state, transform)
        assert (
            np.abs(np.asarray(plain)).max() > 100.0 * np.abs(np.asarray(scaled)).max()
        )

    def test_layout_is_unchanged_by_the_transform(self, state, transform):
        plain, _ = state_to_vector(state)
        scaled, _ = state_to_vector(state, transform)
        assert plain.shape == scaled.shape

    def test_unravel_is_jit_safe(self, state, transform):
        vector, unravel = state_to_vector(state, transform)
        jitted = jax.jit(unravel)
        np.testing.assert_allclose(jitted(vector).u, unravel(vector).u, rtol=1e-6)


class TestMakeEnsemble:
    def test_shape(self, state):
        members = make_ensemble(state, jax.random.key(0), size=32, std=0.1)
        flat, _ = state_to_vector(state)
        assert members.shape == (32, flat.size)

    def test_untransformed_spread_is_isotropic_in_the_flat_vector(self, state):
        members = make_ensemble(state, jax.random.key(0), size=4000, std=0.5)
        spread = np.std(np.asarray(members), axis=0)
        np.testing.assert_allclose(spread, 0.5, rtol=0.1)

    def test_transformed_spread_is_per_field_in_physical_space(self, state, transform):
        """One std means the same thing for h and u once mapped back."""
        std = 0.25
        members = make_ensemble(
            state, jax.random.key(1), size=4000, std=std, transform=transform
        )
        _, unravel = state_to_vector(state, transform)
        physical = jax.vmap(unravel)(members)
        # Across members at fixed gridpoint: the base state's own
        # spatial variation would otherwise dominate the statistic.
        assert float(jnp.mean(jnp.std(physical.h, axis=0))) == pytest.approx(
            std * float(transform.scale.h), rel=0.1
        )
        assert float(jnp.mean(jnp.std(physical.u, axis=0))) == pytest.approx(
            std * float(transform.scale.u), rel=0.1
        )

    def test_this_is_what_untransformed_ensembles_get_wrong(self, state):
        """Without a transform, h and u get the same absolute spread."""
        members = make_ensemble(state, jax.random.key(2), size=2000, std=0.25)
        _, unravel = state_to_vector(state)
        physical = jax.vmap(unravel)(members)
        assert float(jnp.mean(jnp.std(physical.h, axis=0))) == pytest.approx(
            float(jnp.mean(jnp.std(physical.u, axis=0))), rel=0.1
        )

    def test_members_are_centred_on_the_base_state(self, state, transform):
        members = make_ensemble(
            state, jax.random.key(3), size=4000, std=0.1, transform=transform
        )
        _, unravel = state_to_vector(state, transform)
        physical = jax.vmap(unravel)(members)
        np.testing.assert_allclose(
            np.asarray(jnp.mean(physical.u, axis=0)), np.asarray(state.u), atol=5e-3
        )


class TestScaleMetadata:
    def test_scales_attrs_are_flat_primitives(self, scales):
        attrs = scales_attrs(scales)
        assert all(key.startswith(SCALES_ATTR_PREFIX) for key in attrs)
        assert all(isinstance(v, (str, float)) for v in attrs.values())

    def test_scales_attrs_carry_the_time_scale(self, scales):
        """T cannot be recomputed from L and U alone across scale sets."""
        attrs = scales_attrs(scales)
        assert attrs[f"{SCALES_ATTR_PREFIX}T"] == pytest.approx(scales.T)
        assert attrs[f"{SCALES_ATTR_PREFIX}kind"] == "advective"

    def test_transform_attrs_carry_scalar_leaves(self, transform, scales):
        attrs = transform_attrs(transform)
        assert attrs["h"]["loc"] == pytest.approx(scales.H)
        assert attrs["u"]["scale"] == pytest.approx(scales.U)

    def test_transform_attrs_skip_per_gridpoint_leaves(self, state):
        """An array loc belongs in the data, not the metadata."""
        samples = jax.tree_util.tree_map(
            lambda x: jnp.broadcast_to(x, (8, *x.shape)), state
        )
        per_point = StateAffine.from_samples(samples, per_gridpoint=True)
        assert transform_attrs(per_point) == {}

    def test_apply_writes_dataset_and_variable_attrs(self, state, scales, transform):
        ds = state_to_dataset(state)
        apply_scale_metadata(ds, scales=scales, transform=transform)
        assert ds.attrs[f"{SCALES_ATTR_PREFIX}L"] == pytest.approx(scales.L)
        assert ds["h"].attrs["loc"] == pytest.approx(scales.H)

    def test_nondimensional_mode_marks_units(self, state, scales):
        ds = state_to_dataset(state)
        apply_scale_metadata(ds, scales=scales, nondimensional=True)
        assert all(ds[name].attrs["units"] == "-" for name in ds.data_vars)

    def test_attrs_allow_offline_redimensionalisation(self, state, scales, transform):
        """The point of writing them: recover the field from the export."""
        nd = transform.forward(state)
        ds = state_to_dataset(nd)
        apply_scale_metadata(ds, scales=scales, transform=transform)
        recovered = ds["h"].values * ds["h"].attrs["scale"] + ds["h"].attrs["loc"]
        np.testing.assert_allclose(recovered, np.asarray(state.h), rtol=1e-4)

    def test_apply_is_a_noop_without_arguments(self, state):
        ds = state_to_dataset(state)
        before = dict(ds.attrs)
        apply_scale_metadata(ds)
        assert ds.attrs == before


class TestScenarioSpecNondim:
    def _spec(self, **scenario):
        from somax._src.cli.spec import RunSpec

        return RunSpec.from_dict(
            {
                "scenario": {"name": "double_gyre", **scenario},
                "model": {"name": "barotropic_qg"},
                "timestepping": {
                    "t0": 0.0,
                    "t1": 1.0,
                    "dt": 0.1,
                    "save_interval": 0.5,
                },
            }
        )

    def test_nondim_defaults_to_empty(self):
        assert self._spec().scenario.nondim == {}

    def test_nondim_block_is_parsed(self):
        spec = self._spec(nondim={"rossby": 0.02})
        assert spec.scenario.nondim == {"rossby": 0.02}

    def test_nondim_alone_validates(self):
        self._spec(nondim={"rossby": 0.02}).validate()

    def test_consts_alone_validates(self):
        self._spec(consts={"f0": 1e-4}).validate()

    def test_nondim_with_consts_is_rejected(self):
        spec = self._spec(nondim={"rossby": 0.02}, consts={"f0": 1e-4})
        with pytest.raises(ValueError, match="mutually exclusive"):
            spec.validate()

    def test_the_rejection_names_both_blocks(self):
        spec = self._spec(nondim={"rossby": 0.02}, consts={"f0": 1e-4})
        with pytest.raises(ValueError) as excinfo:
            spec.validate()
        assert "rossby" in str(excinfo.value)
        assert "f0" in str(excinfo.value)


class TestUnitsMode:
    def test_si_mode_is_the_default_mapping(self):
        from somax._src.cli._units import FIELD_UNITS, field_units

        assert field_units() is FIELD_UNITS

    def test_nondim_mode_marks_every_field(self):
        from somax._src.cli._units import field_units

        assert set(field_units("nondim").values()) == {"-"}

    def test_field_stats_use_the_supplied_mapping(self):
        from somax._src.cli._units import field_units, format_field_stats

        line = format_field_stats(
            "h",
            min_val=1.0,
            mean_val=2.0,
            max_val=3.0,
            nan_count=0,
            units=field_units("nondim"),
        )
        assert line.startswith("h[-]=")

    def test_unknown_mode_is_rejected(self):
        from somax._src.cli._units import field_units

        with pytest.raises(ValueError, match="'si' or 'nondim'"):
            field_units("metric")


class TestFactoryRouting:
    def test_nondim_block_builds_through_the_factory(self):
        from somax._src.cli._factories import build

        model, state0 = build(
            "double_gyre",
            "barotropic_qg",
            scenario_params={"grid": {"nx": 64, "ny": 64, "Lx": 1.0, "Ly": 1.0}},
            nondim={"rossby": 0.02, "beta_hat": 50.0, "munk": 0.05},
        )
        assert float(model.consts.f0) == pytest.approx(1.0 / 0.02)
        assert state0.q.shape == (model.grid.Ny, model.grid.Nx)

    def test_dimensional_path_is_unchanged(self):
        from somax._src.cli._factories import build

        model, _ = build(
            "double_gyre",
            "barotropic_qg",
            scenario_params={
                "grid": {"nx": 32, "ny": 32, "Lx": 1e6, "Ly": 1e6},
                "consts": {"f0": 1e-4, "beta": 1.6e-11},
            },
            model_params={"params": {"lateral_viscosity": 500.0}},
        )
        assert float(model.consts.f0) == pytest.approx(1e-4)

    def test_unknown_nondim_key_is_rejected(self):
        from somax._src.cli._factories import build

        with pytest.raises(ValueError, match=r"unknown scenario\.nondim key"):
            build(
                "double_gyre",
                "barotropic_qg",
                scenario_params={"grid": {"nx": 64, "ny": 64, "Lx": 1.0, "Ly": 1.0}},
                nondim={"rossby": 0.02, "beta_hat": 50.0, "munk": 0.05, "nope": 1.0},
            )

    def test_missing_required_nondim_key_is_rejected(self):
        from somax._src.cli._factories import build

        with pytest.raises(ValueError, match="missing required key"):
            build(
                "double_gyre",
                "barotropic_qg",
                scenario_params={"grid": {"nx": 64, "ny": 64, "Lx": 1.0, "Ly": 1.0}},
                nondim={"rossby": 0.02},
            )

    def test_model_without_a_nondim_factory_is_rejected(self):
        """Better than silently falling back to the dimensional path."""
        from somax._src.cli._factories import build

        with pytest.raises(ValueError, match="no nondimensional factory"):
            build(
                "double_gyre",
                "multilayer_qg",
                scenario_params={"grid": {"nx": 32, "ny": 32, "Lx": 1.0, "Ly": 1.0}},
                nondim={"rossby": 0.02},
            )


class TestExampleConfig:
    def test_the_authored_nondim_config_is_materialised(self):
        from pathlib import Path

        import yaml

        path = (
            Path(__file__).resolve().parent.parent
            / "configs"
            / "simulation"
            / "doublegyre_bt_qg_nondim.yaml"
        )
        config = yaml.safe_load(path.read_text())
        assert "nondim" in config["scenario"]
        assert "consts" not in config["scenario"]

    def test_the_authored_nondim_config_validates_and_builds(self):
        from pathlib import Path

        import yaml

        from somax._src.cli._factories import build
        from somax._src.cli.spec import RunSpec

        path = (
            Path(__file__).resolve().parent.parent
            / "configs"
            / "simulation"
            / "doublegyre_bt_qg_nondim.yaml"
        )
        spec = RunSpec.from_dict(yaml.safe_load(path.read_text()))
        spec.validate()
        model, _ = build(
            spec.scenario.name,
            spec.model.name,
            scenario_params={
                "grid": spec.scenario.grid,
                "forcing": spec.scenario.forcing,
                "initial_condition": spec.scenario.initial_condition,
            },
            nondim=spec.scenario.nondim,
        )
        assert model is not None
