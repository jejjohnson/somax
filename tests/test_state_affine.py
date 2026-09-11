"""Tests for the per-leaf affine state transform."""

from __future__ import annotations

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
