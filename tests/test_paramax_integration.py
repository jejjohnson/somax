"""Tests for constrained parameters via paramax.

The contract: a model whose ``Params`` hold wrapped values behaves
exactly like one holding the equivalent plain arrays, gradients land on
the stored unconstrained values, and the constraint cannot be violated
by a gradient step.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import paramax
import pytest

from somax._src.core.types import frozen, interval, positive
from somax._src.models.qg.barotropic import BarotropicQG, BarotropicQGState


def build_model(viscosity, drag):
    """A small double-gyre QG model with the given parameter objects."""
    model = BarotropicQG.create(
        nx=16,
        ny=16,
        Lx=1e6,
        Ly=1e6,
        f0=1e-4,
        beta=1.6e-11,
        lateral_viscosity=100.0,
        bottom_drag=1e-7,
        wind_amplitude=1e-5,
    )
    return eqx.tree_at(
        lambda m: (m.params.lateral_viscosity, m.params.bottom_drag),
        model,
        (viscosity, drag),
        is_leaf=lambda x: x is None,
    )


@pytest.fixture
def state():
    # create(nx=16, ny=16) builds a 16-cell interior plus a ghost ring.
    grid = build_model(jnp.asarray(100.0), jnp.asarray(1e-7)).grid
    rng = np.random.RandomState(0)
    return BarotropicQGState(q=jnp.asarray(1e-6 * rng.randn(grid.Ny, grid.Nx)))


class TestHelpers:
    def test_positive_round_trips(self):
        assert float(paramax.unwrap(positive(100.0))) == pytest.approx(100.0, rel=1e-5)

    def test_positive_round_trips_for_arrays(self):
        values = jnp.asarray([1e-8, 1.0, 1e4])
        np.testing.assert_allclose(paramax.unwrap(positive(values)), values, rtol=1e-4)

    def test_positive_stays_positive_for_any_stored_value(self):
        """The point of the wrapper: no raw value maps to a negative."""
        wrapper = positive(1.0)
        for raw in (-80.0, -1.0, 0.0, 1e3):
            stepped = eqx.tree_at(lambda w: w.args[0], wrapper, jnp.asarray(raw))
            assert float(paramax.unwrap(stepped)) > 0.0

    def test_positive_never_goes_negative_even_on_underflow(self):
        """Far into the tail softplus underflows to 0.0 — never below it."""
        wrapper = positive(1.0)
        for raw in (-1e3, -200.0, -104.0):
            stepped = eqx.tree_at(lambda w: w.args[0], wrapper, jnp.asarray(raw))
            assert float(paramax.unwrap(stepped)) == 0.0

    def test_positive_rejects_non_positive(self):
        with pytest.raises(ValueError, match="strictly positive"):
            positive(0.0)
        with pytest.raises(ValueError, match="strictly positive"):
            positive(-1.0)

    def test_positive_survives_a_large_value(self):
        """``log(exp(x) - 1)`` would overflow here; the stable form does not."""
        assert np.isfinite(float(paramax.unwrap(positive(1e5))))
        assert float(paramax.unwrap(positive(1e5))) == pytest.approx(1e5, rel=1e-4)

    def test_interval_round_trips(self):
        assert float(paramax.unwrap(interval(0.25, 0.0, 1.0))) == pytest.approx(
            0.25, rel=1e-5
        )

    def test_interval_stays_inside_for_any_stored_value(self):
        wrapper = interval(4.0, 2.0, 6.0)
        # Beyond about +-16 the offset from the bound falls below float32
        # resolution at 2.0, so the value rounds onto the bound itself.
        for raw in (-15.0, 0.0, 15.0):
            stepped = eqx.tree_at(lambda w: w.args[0], wrapper, jnp.asarray(raw))
            got = float(paramax.unwrap(stepped))
            assert 2.0 < got < 6.0

    def test_interval_saturates_onto_a_bound_but_never_past_it(self):
        """Deep in either tail sigmoid reaches exactly 0 or 1."""
        wrapper = interval(4.0, 2.0, 6.0)
        for raw in (-1e3, 1e3):
            stepped = eqx.tree_at(lambda w: w.args[0], wrapper, jnp.asarray(raw))
            got = float(paramax.unwrap(stepped))
            assert 2.0 <= got <= 6.0

    def test_interval_rejects_a_value_outside_the_bounds(self):
        with pytest.raises(ValueError, match="strictly inside"):
            interval(1.5, 0.0, 1.0)

    def test_interval_rejects_unordered_bounds(self):
        with pytest.raises(ValueError, match="upper must exceed lower"):
            interval(0.5, 1.0, 0.0)

    def test_frozen_round_trips(self):
        assert float(paramax.unwrap(frozen(3.0))) == pytest.approx(3.0)

    def test_frozen_has_zero_gradient(self):
        """NonTrainable cuts the backward pass, returning exact zero."""

        def loss(x):
            return jnp.sum(paramax.unwrap(x) ** 2)

        g = eqx.filter_grad(loss)(frozen(3.0))
        leaves = jax.tree_util.tree_leaves(eqx.filter(g, eqx.is_inexact_array))
        assert leaves
        assert all(float(leaf) == 0.0 for leaf in leaves)


class TestUnwrapIsTransparent:
    def test_plain_model_is_unchanged_by_unwrap(self, state):
        model = build_model(jnp.asarray(100.0), jnp.asarray(1e-7))
        assert paramax.unwrap(model) is not None
        np.testing.assert_allclose(
            paramax.unwrap(model).vector_field(0.0, state).q,
            model.vector_field(0.0, state).q,
        )

    def test_wrapped_model_matches_the_plain_one(self, state):
        """Same physical values, wrapped or not, give the same tendency."""
        plain = build_model(jnp.asarray(100.0), jnp.asarray(1e-7))
        wrapped = build_model(positive(100.0), positive(1e-7))
        term_plain = plain.build_terms()
        term_wrapped = wrapped.build_terms()
        np.testing.assert_allclose(
            term_wrapped.vf(0.0, state, None).q,
            term_plain.vf(0.0, state, None).q,
            rtol=1e-4,
        )

    def test_trajectories_agree(self, state):
        plain = build_model(jnp.asarray(100.0), jnp.asarray(1e-7))
        wrapped = build_model(positive(100.0), positive(1e-7))
        kw = {"t0": 0.0, "t1": 200.0, "dt": 50.0}
        np.testing.assert_allclose(
            wrapped.integrate(state, **kw).ys.q,
            plain.integrate(state, **kw).ys.q,
            rtol=1e-4,
            atol=1e-12,
        )

    def test_diagnostics_agree(self, state):
        plain = build_model(jnp.asarray(100.0), jnp.asarray(1e-7))
        wrapped = build_model(positive(100.0), positive(1e-7))
        for key, value in plain.diagnostics(state).invariants().items():
            np.testing.assert_allclose(
                wrapped.diagnostics(state).invariants()[key], value, rtol=1e-4
            )

    def test_diagnostics_is_a_noop_for_plain_models(self, state):
        plain = build_model(jnp.asarray(100.0), jnp.asarray(1e-7))
        for key, value in plain.diagnose(state).invariants().items():
            np.testing.assert_allclose(
                plain.diagnostics(state).invariants()[key], value
            )


class TestGradients:
    def test_grad_flows_to_the_unconstrained_value(self, state):
        wrapped = build_model(positive(100.0), positive(1e-7))

        def loss(model):
            return jnp.sum(model.integrate(state, t0=0.0, t1=100.0, dt=50.0).ys.q ** 2)

        grads = eqx.filter_grad(loss)(wrapped)
        raw_grad = grads.params.lateral_viscosity.args[0]
        assert np.isfinite(float(raw_grad))
        assert float(raw_grad) != 0.0

    def test_chain_rule_factor_is_the_sigmoid(self, state):
        """d/dr = sigmoid(r) * d/dnu, which is what makes the bound hold."""
        nu = 100.0

        def through_wrapper(raw):
            model = build_model(
                paramax.Parameterize(jax.nn.softplus, raw), jnp.asarray(1e-7)
            )
            return jnp.sum(model.integrate(state, t0=0.0, t1=100.0, dt=50.0).ys.q ** 2)

        def through_plain(value):
            model = build_model(value, jnp.asarray(1e-7))
            return jnp.sum(model.integrate(state, t0=0.0, t1=100.0, dt=50.0).ys.q ** 2)

        raw = jnp.asarray(float(np.log(np.expm1(nu))))
        g_raw = float(jax.grad(through_wrapper)(raw))
        g_plain = float(jax.grad(through_plain)(jnp.asarray(nu)))
        expected = g_plain * float(jax.nn.sigmoid(raw))
        assert g_raw == pytest.approx(expected, rel=1e-3)

    def test_a_gradient_step_cannot_make_viscosity_negative(self, state):
        """The property the wrapper exists for."""
        wrapped = build_model(positive(1.0), positive(1e-7))

        def loss(model):
            return jnp.sum(model.integrate(state, t0=0.0, t1=100.0, dt=50.0).ys.q ** 2)

        grads = eqx.filter_grad(loss)(wrapped)
        raw = wrapped.params.lateral_viscosity.args[0]
        raw_grad = grads.params.lateral_viscosity.args[0]
        for lr in (1e-3, 1.0, 1e6):  # including an absurd learning rate
            stepped = raw - lr * raw_grad
            nu = float(jax.nn.softplus(stepped))
            assert nu > 0.0

    def test_frozen_parameter_gets_no_gradient(self, state):
        model = build_model(frozen(100.0), positive(1e-7))

        def loss(m):
            return jnp.sum(m.integrate(state, t0=0.0, t1=100.0, dt=50.0).ys.q ** 2)

        grads = eqx.filter_grad(loss)(model)
        frozen_leaves = jax.tree_util.tree_leaves(
            eqx.filter(grads.params.lateral_viscosity, eqx.is_inexact_array)
        )
        assert frozen_leaves
        assert all(float(leaf) == 0.0 for leaf in frozen_leaves)
        # The unfrozen parameter in the same model still gets a real gradient.
        assert float(grads.params.bottom_drag.args[0]) != 0.0

    def test_plain_model_gradients_still_work(self, state):
        plain = build_model(jnp.asarray(100.0), jnp.asarray(1e-7))

        def loss(model):
            return jnp.sum(model.integrate(state, t0=0.0, t1=100.0, dt=50.0).ys.q ** 2)

        grads = eqx.filter_grad(loss)(plain)
        assert np.isfinite(float(grads.params.lateral_viscosity))


class TestJit:
    def test_wrapped_model_integrates_under_jit(self, state):
        wrapped = build_model(positive(100.0), positive(1e-7))
        run = eqx.filter_jit(lambda m, s: m.integrate(s, t0=0.0, t1=100.0, dt=50.0))
        np.testing.assert_allclose(
            run(wrapped, state).ys.q,
            wrapped.integrate(state, t0=0.0, t1=100.0, dt=50.0).ys.q,
            rtol=1e-5,
        )

    def test_unwrap_does_not_leak_wrappers_into_the_tendency(self, state):
        wrapped = build_model(positive(100.0), positive(1e-7))
        out = wrapped.build_terms().vf(0.0, state, None)
        assert not paramax.contains_unwrappables(out)
