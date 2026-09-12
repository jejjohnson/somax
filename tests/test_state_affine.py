"""Tests for the per-leaf affine state transform."""

from __future__ import annotations

import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from finitevolx import Mask2D
from jaxtyping import Array

from somax._src.core.scales import Scales
from somax._src.core.transforms import StateAffine
from somax._src.core.types import State
from somax._src.models.qg.barotropic import BarotropicQGState
from somax._src.models.swm.multilayer import MultilayerSW2DState
from somax._src.models.swm.nonlinear_2d import NonlinearSW2DState


NY, NX = 5, 6
NL = 3


@pytest.fixture
def scales():
    return Scales.advective(L=1e6, U=0.1, f0=1e-4, H=1000.0)


@pytest.fixture
def sw_state():
    rng = np.random.RandomState(0)
    return NonlinearSW2DState(
        h=jnp.asarray(1000.0 + rng.randn(NY, NX)),
        u=jnp.asarray(0.1 * rng.randn(NY, NX)),
        v=jnp.asarray(0.1 * rng.randn(NY, NX)),
    )


def tree_allclose(a, b, **kw):
    leaves_a = jax.tree_util.tree_leaves(a)
    leaves_b = jax.tree_util.tree_leaves(b)
    assert len(leaves_a) == len(leaves_b)
    return all(np.allclose(x, y, **kw) for x, y in zip(leaves_a, leaves_b, strict=True))


class UnknownFieldState(State):
    """A state with a field no scale rule knows about."""

    tracer: Array


class TestFromScales:
    def test_per_field_rules(self, scales, sw_state):
        t = StateAffine.from_scales(NonlinearSW2DState, scales)
        assert t.scale.u == pytest.approx(scales.U)
        assert t.scale.v == pytest.approx(scales.U)
        assert t.scale.h == pytest.approx(scales.eta)
        assert t.loc.h == pytest.approx(scales.H)
        assert t.loc.u == 0.0

    def test_vorticity_and_streamfunction_rules(self, scales):
        t = StateAffine.from_scales(BarotropicQGState, scales)
        assert t.scale.q == pytest.approx(scales.vorticity)
        assert t.loc.q == 0.0

    def test_forward_nondimensionalises(self, scales, sw_state):
        t = StateAffine.from_scales(NonlinearSW2DState, scales)
        nd = t.forward(sw_state)
        np.testing.assert_allclose(nd.u, sw_state.u / scales.U, rtol=1e-6)
        np.testing.assert_allclose(
            nd.h, (sw_state.h - scales.H) / scales.eta, rtol=1e-6
        )

    def test_loc_and_scale_are_state_instances(self, scales):
        t = StateAffine.from_scales(NonlinearSW2DState, scales)
        assert isinstance(t.loc, NonlinearSW2DState)
        assert isinstance(t.scale, NonlinearSW2DState)

    def test_unknown_field_warns_and_is_left_alone(self, scales):
        with pytest.warns(UserWarning, match="no scale rule"):
            t = StateAffine.from_scales(UnknownFieldState, scales)
        assert t.loc.tracer == 0.0
        assert t.scale.tracer == 1.0

    def test_scalar_override_sets_scale_only(self, scales):
        t = StateAffine.from_scales(NonlinearSW2DState, scales, u=7.0)
        assert t.scale.u == 7.0
        assert t.loc.u == 0.0

    def test_tuple_override_sets_both(self, scales):
        t = StateAffine.from_scales(NonlinearSW2DState, scales, h=(2.0, 3.0))
        assert t.loc.h == 2.0
        assert t.scale.h == 3.0

    def test_dict_override_sets_both(self, scales):
        t = StateAffine.from_scales(
            NonlinearSW2DState, scales, h={"loc": 2.0, "scale": 3.0}
        )
        assert t.loc.h == 2.0
        assert t.scale.h == 3.0

    def test_override_silences_the_unknown_field_warning(self, scales):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            t = StateAffine.from_scales(UnknownFieldState, scales, tracer=5.0)
        assert t.scale.tracer == 5.0

    def test_override_for_a_missing_field_is_rejected(self, scales):
        with pytest.raises(ValueError, match="no field"):
            StateAffine.from_scales(NonlinearSW2DState, scales, nope=1.0)

    def test_dict_override_with_bad_key_is_rejected(self, scales):
        with pytest.raises(ValueError, match="unexpected key"):
            StateAffine.from_scales(NonlinearSW2DState, scales, h={"mean": 1.0})

    def test_zero_scale_is_rejected(self, scales):
        with pytest.raises(ValueError, match="not be invertible"):
            StateAffine.from_scales(NonlinearSW2DState, scales, u=0.0)


class TestRoundTrip:
    def test_scalar_loc_and_scale(self, scales, sw_state):
        t = StateAffine.from_scales(NonlinearSW2DState, scales)
        assert tree_allclose(t.inverse(t.forward(sw_state)), sw_state, rtol=1e-6)

    def test_per_layer_loc_and_scale(self, scales):
        """Multilayer leaves of shape (nl, 1, 1) broadcast over (nl, Ny, Nx)."""
        rng = np.random.RandomState(1)
        g_prime = jnp.asarray([9.81, 0.05, 0.02])[:, None, None]
        dH = scales.f0 * scales.U * scales.L / g_prime
        H_k = jnp.asarray([500.0, 1000.0, 3000.0])[:, None, None]
        # Thicknesses sit near their resting values; a state of O(1)
        # thicknesses against a 3000 m loc would just be testing float32
        # cancellation, not the transform.
        state = MultilayerSW2DState(
            h=H_k + jnp.asarray(rng.randn(NL, NY, NX)),
            u=jnp.asarray(rng.randn(NL, NY, NX)),
            v=jnp.asarray(rng.randn(NL, NY, NX)),
        )
        t = StateAffine.from_scales(
            MultilayerSW2DState, scales, h={"loc": H_k, "scale": dH}
        )
        assert t.scale.h.shape == (NL, 1, 1)
        nd = t.forward(state)
        assert nd.h.shape == (NL, NY, NX)
        np.testing.assert_allclose(nd.h, (state.h - H_k) / dH, rtol=1e-6)
        assert tree_allclose(t.inverse(nd), state, rtol=1e-6)

    def test_per_gridpoint_loc_and_scale(self, sw_state):
        rng = np.random.RandomState(2)
        traj = jax.tree_util.tree_map(
            lambda x: jnp.asarray(rng.randn(8, *x.shape)), sw_state
        )
        t = StateAffine.from_samples(traj, per_gridpoint=True)
        assert t.scale.h.shape == (NY, NX)
        single = jax.tree_util.tree_map(lambda x: x[0], traj)
        assert tree_allclose(t.inverse(t.forward(single)), single, rtol=1e-6)

    def test_identity_is_a_no_op(self, sw_state):
        t = StateAffine.identity(sw_state)
        assert tree_allclose(t.forward(sw_state), sw_state)
        assert tree_allclose(t.inverse(sw_state), sw_state)


class TestCompose:
    def test_matches_sequential_application(self, scales, sw_state):
        a = StateAffine.from_scales(NonlinearSW2DState, scales)
        b = StateAffine.from_scales(
            NonlinearSW2DState, scales, h=(3.0, 2.0), u=5.0, v=4.0
        )
        composed = a.compose(b)
        assert tree_allclose(
            composed.forward(sw_state), a.forward(b.forward(sw_state)), rtol=1e-6
        )

    def test_composition_is_invertible(self, scales, sw_state):
        a = StateAffine.from_scales(NonlinearSW2DState, scales)
        b = StateAffine.from_scales(NonlinearSW2DState, scales, h=(3.0, 2.0))
        composed = a.compose(b)
        assert tree_allclose(
            composed.inverse(composed.forward(sw_state)), sw_state, rtol=1e-6
        )

    def test_compose_with_identity(self, scales, sw_state):
        a = StateAffine.from_scales(NonlinearSW2DState, scales)
        ident = StateAffine.identity(sw_state)
        assert tree_allclose(
            a.compose(ident).forward(sw_state), a.forward(sw_state), rtol=1e-6
        )

    def test_order_matters(self, scales, sw_state):
        """compose is not commutative, so the argument order is load-bearing."""
        a = StateAffine.from_scales(NonlinearSW2DState, scales)
        b = StateAffine.from_scales(NonlinearSW2DState, scales, h=(3.0, 2.0))
        assert not tree_allclose(
            a.compose(b).forward(sw_state), b.compose(a).forward(sw_state)
        )


class TestFromSamples:
    @pytest.fixture
    def traj(self, sw_state):
        rng = np.random.RandomState(3)
        return jax.tree_util.tree_map(
            lambda x: jnp.asarray(rng.randn(32, *x.shape) * 2.0 + 1.0), sw_state
        )

    def test_recovers_the_sample_moments_per_field(self, traj):
        t = StateAffine.from_samples(traj)
        assert t.loc.h.shape == ()
        assert float(t.loc.h) == pytest.approx(float(jnp.mean(traj.h)), rel=1e-6)
        assert float(t.scale.h) == pytest.approx(float(jnp.std(traj.h)), rel=1e-6)

    def test_recovers_the_sample_moments_per_gridpoint(self, traj):
        t = StateAffine.from_samples(traj, per_gridpoint=True)
        assert t.loc.h.shape == (NY, NX)
        np.testing.assert_allclose(t.loc.h, jnp.mean(traj.h, axis=0), rtol=1e-6)
        np.testing.assert_allclose(t.scale.h, jnp.std(traj.h, axis=0), rtol=1e-6)

    def test_standardised_samples_have_unit_moments(self, traj):
        t = StateAffine.from_samples(traj, per_gridpoint=True)
        nd = t.forward(traj)
        np.testing.assert_allclose(jnp.mean(nd.h, axis=0), 0.0, atol=1e-6)
        np.testing.assert_allclose(jnp.std(nd.h, axis=0), 1.0, rtol=1e-6)

    def test_constant_field_gets_the_eps_floor(self, sw_state):
        constant = jax.tree_util.tree_map(
            lambda x: jnp.broadcast_to(jnp.ones_like(x), (8, *x.shape)), sw_state
        )
        t = StateAffine.from_samples(constant, per_gridpoint=True, eps=1e-6)
        np.testing.assert_allclose(t.scale.h, 1e-6)

    def test_constant_field_still_round_trips(self, sw_state):
        constant = jax.tree_util.tree_map(
            lambda x: jnp.broadcast_to(jnp.ones_like(x), (8, *x.shape)), sw_state
        )
        t = StateAffine.from_samples(constant, per_gridpoint=True)
        single = jax.tree_util.tree_map(lambda x: x[0], constant)
        assert tree_allclose(t.inverse(t.forward(single)), single, rtol=1e-6)


class TestMaskedSamples:
    @pytest.fixture
    def mask(self):
        wet = np.ones((NY, NX), dtype=bool)
        wet[1:3, 2:4] = False
        return Mask2D.from_mask(jnp.asarray(wet))

    @pytest.fixture
    def traj(self, mask):
        rng = np.random.RandomState(4)

        def field(location):
            data = rng.randn(16, NY, NX)
            return jnp.where(getattr(mask, location), jnp.asarray(data), jnp.nan)

        return NonlinearSW2DState(h=field("h"), u=field("u"), v=field("v"))

    def test_dry_cells_get_the_identity_transform(self, traj, mask):
        t = StateAffine.from_samples(traj, per_gridpoint=True, mask=mask)
        dry = ~np.asarray(mask.h)
        assert dry.any()
        np.testing.assert_array_equal(np.asarray(t.loc.h)[dry], 0.0)
        np.testing.assert_array_equal(np.asarray(t.scale.h)[dry], 1.0)

    def test_land_sentinels_do_not_leak_into_wet_statistics(self, traj, mask):
        t = StateAffine.from_samples(traj, per_gridpoint=True, mask=mask)
        assert np.isfinite(np.asarray(t.loc.h)).all()
        assert np.isfinite(np.asarray(t.scale.h)).all()

    def test_staggered_fields_use_their_own_mask(self, traj, mask):
        t = StateAffine.from_samples(traj, per_gridpoint=True, mask=mask)
        np.testing.assert_array_equal(np.asarray(t.scale.u)[~np.asarray(mask.u)], 1.0)
        # The u-mask differs from the h-mask, so the kwarg is doing work.
        assert not np.array_equal(np.asarray(mask.u), np.asarray(mask.h))

    def test_wet_cells_match_the_plain_moments(self, traj, mask):
        t = StateAffine.from_samples(traj, per_gridpoint=True, mask=mask)
        wet = np.asarray(mask.h)
        # Average only the wet columns: the dry ones are all-NaN by
        # construction and have no plain mean to compare against.
        expected = np.asarray(traj.h)[:, wet].mean(axis=0)
        np.testing.assert_allclose(np.asarray(t.loc.h)[wet], expected, rtol=1e-6)


class TestLogDet:
    def test_equals_minus_sum_n_log_scale(self, scales, sw_state):
        t = StateAffine.from_scales(NonlinearSW2DState, scales)
        n = NY * NX
        expected = -(
            n * np.log(scales.eta) + n * np.log(scales.U) + n * np.log(scales.U)
        )
        assert float(t.log_det(sw_state)) == pytest.approx(expected, rel=1e-6)

    def test_is_constant_in_the_state(self, scales, sw_state):
        t = StateAffine.from_scales(NonlinearSW2DState, scales)
        other = jax.tree_util.tree_map(lambda x: x * 3.0 + 1.0, sw_state)
        assert float(t.log_det(sw_state)) == pytest.approx(float(t.log_det(other)))

    def test_identity_has_zero_log_det(self, sw_state):
        assert float(StateAffine.identity(sw_state).log_det(sw_state)) == pytest.approx(
            0.0
        )

    def test_flowjax_style_signatures_agree(self, scales, sw_state):
        t = StateAffine.from_scales(NonlinearSW2DState, scales)
        y, ld = t.transform_and_log_det(sw_state)
        assert tree_allclose(y, t.forward(sw_state))
        assert float(ld) == pytest.approx(float(t.log_det(sw_state)))

        x, ild = t.inverse_and_log_det(y)
        assert tree_allclose(x, sw_state, rtol=1e-6)
        assert float(ild) == pytest.approx(-float(t.log_det(sw_state)))

    def test_inverse_log_det_negates_forward(self, scales, sw_state):
        t = StateAffine.from_scales(NonlinearSW2DState, scales)
        _, fwd = t.transform_and_log_det(sw_state)
        _, inv = t.inverse_and_log_det(sw_state)
        assert float(fwd) == pytest.approx(-float(inv))


class TestTransforms:
    def test_jit(self, scales, sw_state):
        t = StateAffine.from_scales(NonlinearSW2DState, scales)
        jitted = eqx.filter_jit(lambda tr, s: tr.forward(s))
        assert tree_allclose(jitted(t, sw_state), t.forward(sw_state), rtol=1e-6)

    def test_grad_through_forward(self, scales, sw_state):
        t = StateAffine.from_scales(NonlinearSW2DState, scales)

        def loss(state):
            return jnp.sum(t.forward(state).u ** 2)

        g = eqx.filter_grad(loss)(sw_state)
        np.testing.assert_allclose(g.u, 2.0 * sw_state.u / scales.U**2, rtol=1e-6)

    def test_grad_through_inverse(self, scales, sw_state):
        t = StateAffine.from_scales(NonlinearSW2DState, scales)

        def loss(state):
            return jnp.sum(t.inverse(state).u)

        g = eqx.filter_grad(loss)(sw_state)
        np.testing.assert_allclose(g.u, scales.U, rtol=1e-6)

    def test_transform_is_a_pytree(self, scales):
        t = StateAffine.from_scales(NonlinearSW2DState, scales)
        leaves = jax.tree_util.tree_leaves(t)
        assert len(leaves) == 6  # loc and scale, three fields each


class TestFieldNamesAreOnlyAHint:
    """A field's name does not fix its scaling or its staggering.

    ``h`` is a total thickness in the nonlinear shallow-water models
    and a height anomaly in the linear ones; ``u`` is a C-grid velocity
    in the ocean models and a T-point scalar in the pde family. Each
    state says which it means.
    """

    def scales(self):
        return Scales.advective(L=1.0e6, U=0.1, f0=1.0e-4, H=500.0)

    def test_linear_shallow_water_height_is_not_offset(self):
        """A resting linear state is zero, and must map to zero."""
        from somax._src.models.swm.linear_2d import LinearSW2DState

        transform = StateAffine.from_scales(LinearSW2DState, self.scales())
        assert float(transform.loc.h) == 0.0

    def test_nonlinear_shallow_water_height_still_is(self):
        """The default is unchanged for states that do mean thickness."""
        transform = StateAffine.from_scales(NonlinearSW2DState, self.scales())
        assert float(transform.loc.h) == pytest.approx(self.scales().H)

    def test_a_resting_linear_state_maps_to_zero(self):
        from somax._src.models.swm.linear_2d import LinearSW2DState

        transform = StateAffine.from_scales(LinearSW2DState, self.scales())
        rest = LinearSW2DState(
            h=jnp.zeros((4, 4)), u=jnp.zeros((4, 4)), v=jnp.zeros((4, 4))
        )
        np.testing.assert_allclose(np.asarray(transform.forward(rest).h), 0.0)

    def test_navier_stokes_vorticity_is_recognised(self):
        """``omega`` is this family's spelling; it must not warn."""
        from somax._src.models.pde2d.navier_stokes import NSVorticityState

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            transform = StateAffine.from_scales(NSVorticityState, self.scales())
        assert float(transform.scale.omega) == pytest.approx(self.scales().vorticity)

    def test_collocated_velocities_use_the_t_point_mask(self):
        """Burgers keeps u and v at T-points, so mask.u would be wrong."""
        from somax._src.models.pde2d.burgers import Burgers2DState

        assert Burgers2DState.mask_locations == {"u": "h", "v": "h"}

    def test_qg_potential_vorticity_uses_the_t_point_mask(self):
        assert BarotropicQGState.mask_locations == {"q": "h"}

    def test_the_declared_mask_is_the_one_actually_used(self):
        """Not just declared: a dry T-cell must be excluded in practice."""
        from somax._src.models.pde2d.burgers import Burgers2DState

        wet = np.ones((6, 6), dtype=bool)
        wet[2:4, 2:4] = False
        mask = Mask2D.from_mask(jnp.asarray(wet))

        samples = jnp.asarray(np.ones((5, 6, 6)))
        # Poison the dry cells; with the T-point mask they are excluded
        # and the wet-cell mean stays exactly 1.
        samples = samples.at[:, 2:4, 2:4].set(1.0e6)
        transform = StateAffine.from_samples(
            Burgers2DState(u=samples, v=samples), mask=mask
        )
        assert float(jnp.max(transform.loc.u)) == pytest.approx(1.0, rel=1e-6)

    def test_metadata_does_not_become_a_state_field(self):
        """The ClassVars are metadata, not leaves."""
        from somax._src.models.pde2d.burgers import Burgers2DState

        transform = StateAffine.from_scales(Burgers2DState, self.scales())
        assert not hasattr(transform.loc, "mask_locations_leaf")
        assert len(jax.tree_util.tree_leaves(transform.loc)) == 2


class TestDryCellsInFieldwiseStatistics:
    """``per_gridpoint=False`` must still leave land alone.

    Reducing to one number per field would otherwise hand every dry
    cell the wet cells' statistics, moving its land sentinel and
    counting it in the log determinant.
    """

    def setup_method(self):
        wet = np.ones((6, 6), dtype=bool)
        wet[2:4, 2:4] = False
        self.wet = wet
        self.mask = Mask2D.from_mask(jnp.asarray(wet))
        rng = np.random.RandomState(0)
        field = rng.randn(5, 6, 6) * 3.0 + 7.0
        field[:, 2:4, 2:4] = np.nan  # a land sentinel
        self.samples = NonlinearSW2DState(
            h=jnp.asarray(field), u=jnp.asarray(field), v=jnp.asarray(field)
        )

    def transform(self):
        return StateAffine.from_samples(self.samples, mask=self.mask)

    def test_dry_cells_get_the_identity(self):
        transform = self.transform()
        dry = ~self.wet
        np.testing.assert_allclose(np.asarray(transform.loc.h)[dry], 0.0)
        np.testing.assert_allclose(np.asarray(transform.scale.h)[dry], 1.0)

    def test_wet_cells_share_one_number(self):
        """Fieldwise still means fieldwise — one statistic, not per cell."""
        transform = self.transform()
        wet_locs = np.asarray(transform.loc.h)[self.wet]
        assert np.allclose(wet_locs, wet_locs[0])

    def test_a_land_sentinel_is_left_untouched(self):
        transform = self.transform()
        sentinel = jnp.asarray(np.where(self.wet, 0.0, -9999.0))
        state = NonlinearSW2DState(h=sentinel, u=sentinel, v=sentinel)
        got = np.asarray(transform.forward(state).h)
        np.testing.assert_allclose(got[~self.wet], -9999.0)

    def test_the_wet_statistics_are_unaffected_by_the_sentinel(self):
        transform = self.transform()
        assert np.isfinite(float(np.asarray(transform.loc.h)[self.wet][0]))

    def test_without_a_mask_the_statistics_stay_scalar(self):
        """No mask, no land: nothing to broadcast over."""
        clean = jax.tree_util.tree_map(jnp.nan_to_num, self.samples)
        transform = StateAffine.from_samples(clean)
        assert jnp.ndim(transform.loc.h) == 0


class TestArrayValuedScaleValidation:
    """A zero anywhere in an array scale divides by zero just the same."""

    def scales(self):
        return Scales.advective(L=1.0e6, U=0.1, f0=1.0e-4, H=500.0)

    def test_a_zero_entry_in_a_per_layer_scale_is_rejected(self):
        with pytest.raises(ValueError, match="has a zero entry"):
            StateAffine.from_scales(
                MultilayerSW2DState,
                self.scales(),
                h=jnp.asarray([[[1.0]], [[0.0]], [[3.0]]]),
            )

    def test_a_scalar_zero_still_says_it_is_zero(self):
        with pytest.raises(ValueError, match="is zero"):
            StateAffine.from_scales(NonlinearSW2DState, self.scales(), h=0.0)

    def test_a_fully_nonzero_array_is_accepted(self):
        transform = StateAffine.from_scales(
            MultilayerSW2DState,
            self.scales(),
            h=jnp.asarray([[[1.0]], [[2.0]], [[3.0]]]),
        )
        assert transform.scale.h.shape == (3, 1, 1)

    def test_a_negative_entry_is_fine(self):
        """Only zero breaks invertibility; a sign flip is a valid map."""
        transform = StateAffine.from_scales(
            MultilayerSW2DState,
            self.scales(),
            h=jnp.asarray([[[1.0]], [[-2.0]], [[3.0]]]),
        )
        assert float(transform.scale.h[1, 0, 0]) == -2.0


class TestGenericPytreesAreNotMistakenForMomentPairs:
    """``from_samples`` accepts any pytree, including two-element ones.

    A generated ``(mean, std)`` pair used to be a bare 2-tuple, so a
    state that *is* a two-element container had its own structural node
    unpacked instead — sending one child's statistics to ``loc`` and
    the other's to ``scale``.
    """

    def samples(self):
        rng = np.random.RandomState(0)
        return (
            jnp.asarray(rng.randn(6, 4) + 10.0),
            jnp.asarray(rng.randn(6, 4) - 5.0),
        )

    def test_a_two_tuple_state_keeps_its_structure(self):
        transform = StateAffine.from_samples(self.samples())
        assert len(transform.loc) == 2
        assert len(transform.scale) == 2

    def test_each_child_gets_its_own_mean(self):
        first, second = self.samples()
        transform = StateAffine.from_samples((first, second))
        np.testing.assert_allclose(
            np.asarray(transform.loc[0]), np.asarray(jnp.mean(first)), rtol=1e-5
        )
        np.testing.assert_allclose(
            np.asarray(transform.loc[1]), np.asarray(jnp.mean(second)), rtol=1e-5
        )

    def test_the_scales_are_standard_deviations_not_the_other_child(self):
        first, second = self.samples()
        transform = StateAffine.from_samples((first, second))
        np.testing.assert_allclose(
            np.asarray(transform.scale[0]), np.asarray(jnp.std(first)), rtol=1e-5
        )

    def test_it_round_trips(self):
        samples = self.samples()
        transform = StateAffine.from_samples(samples)
        state = tuple(leaf[0] for leaf in samples)
        back = transform.inverse(transform.forward(state))
        for got, expected in zip(back, state, strict=True):
            np.testing.assert_allclose(np.asarray(got), np.asarray(expected), rtol=1e-4)

    def test_a_three_tuple_was_never_affected(self):
        """Only the two-element case was ambiguous."""
        rng = np.random.RandomState(1)
        samples = tuple(jnp.asarray(rng.randn(6, 4)) for _ in range(3))
        assert len(StateAffine.from_samples(samples).loc) == 3

    def test_an_ordinary_state_still_works(self):
        rng = np.random.RandomState(2)
        shape = (5, 4, 4)
        samples = NonlinearSW2DState(
            h=jnp.asarray(rng.randn(*shape)),
            u=jnp.asarray(rng.randn(*shape)),
            v=jnp.asarray(rng.randn(*shape)),
        )
        transform = StateAffine.from_samples(samples)
        assert jnp.ndim(transform.loc.h) == 0


class TestSampleFloorMustBeUsable:
    """``eps`` is the floor on the returned scale, so zero is not a floor."""

    def samples(self):
        return NonlinearSW2DState(
            h=jnp.ones((5, 4, 4)), u=jnp.ones((5, 4, 4)), v=jnp.ones((5, 4, 4))
        )

    @pytest.mark.parametrize("bad", [0.0, -1e-8, float("nan"), float("inf")])
    def test_a_bad_floor_is_rejected(self, bad):
        with pytest.raises(ValueError, match="eps must be"):
            StateAffine.from_samples(self.samples(), eps=bad)

    def test_a_constant_field_would_otherwise_get_a_zero_scale(self):
        """What the validation prevents: an inverse that divides by zero."""
        transform = StateAffine.from_samples(self.samples(), eps=1e-6)
        assert float(transform.scale.h) == pytest.approx(1e-6)

    def test_a_valid_floor_is_still_accepted(self):
        transform = StateAffine.from_samples(self.samples(), eps=1e-3)
        assert float(transform.scale.h) == pytest.approx(1e-3)


class TestNonlinearOneDimensionalCoriolisPartner:
    """``NonlinearSW1DState.v`` is a T-point field, like the linear one."""

    def test_it_declares_the_t_point_mask(self):
        from somax._src.models.swm.nonlinear_1d import NonlinearSW1DState

        assert NonlinearSW1DState.mask_locations == {"v": "h"}

    def test_it_matches_the_linear_state(self):
        from somax._src.models.swm.linear_1d import LinearSW1DState
        from somax._src.models.swm.nonlinear_1d import NonlinearSW1DState

        assert (
            NonlinearSW1DState.mask_locations["v"]
            == LinearSW1DState.mask_locations["v"]
        )

    def test_u_keeps_the_staggered_mask(self):
        """Only ``v`` is collocated; ``u`` really is at U-points."""
        from somax._src.models.swm.nonlinear_1d import NonlinearSW1DState

        assert "u" not in NonlinearSW1DState.mask_locations
