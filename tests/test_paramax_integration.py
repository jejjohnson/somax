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

from somax._src.core.types import (
    as_parameter,
    frozen,
    interval,
    positive,
    trainable_mask,
)
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


class TestNonFiniteInitialValues:
    """NaN must not slip past the range checks.

    Every comparison with NaN is false, so a check written as
    ``any(value <= 0)`` waves it through and builds a wrapper that
    unwraps to NaN — which then contaminates the integration and the
    gradients far from the call that caused it.
    """

    def test_positive_rejects_nan(self):
        with pytest.raises(ValueError, match="strictly positive"):
            positive(jnp.nan)

    def test_positive_rejects_nan_in_an_array(self):
        with pytest.raises(ValueError, match="strictly positive"):
            positive(jnp.asarray([1.0, jnp.nan, 3.0]))

    def test_positive_still_rejects_zero_and_negatives(self):
        for bad in (0.0, -1.0):
            with pytest.raises(ValueError, match="strictly positive"):
                positive(bad)

    def test_infinity_has_no_usable_pre_image(self):
        """+inf passes ``> 0`` but its softplus pre-image is not finite."""
        assert not np.isfinite(float(positive(jnp.inf).args[0]))

    def test_interval_rejects_nan(self):
        with pytest.raises(ValueError, match="strictly inside"):
            interval(jnp.nan, 0.0, 1.0)

    def test_interval_rejects_nan_in_an_array(self):
        with pytest.raises(ValueError, match="strictly inside"):
            interval(jnp.asarray([0.5, jnp.nan]), 0.0, 1.0)

    def test_interval_still_rejects_out_of_range(self):
        for bad in (0.0, 1.0, -0.5, 2.0):
            with pytest.raises(ValueError, match="strictly inside"):
                interval(bad, 0.0, 1.0)


class TestIntervalWrappersAreStructurallyCompatible:
    """Two models with the same constraint must combine into an ensemble.

    ``Parameterize`` keeps its transform as an ordinary pytree leaf, so
    it lands in the *static* half of an ``eqx.partition`` — and a
    closure built inside ``interval()`` is a fresh object every call
    that compares equal to nothing. Stacking the array halves of two
    independently built models and recombining them then picks one
    member's closure arbitrarily, which is only harmless by luck. A
    module-level frozen dataclass compares by its bounds instead.

    (The treedefs match either way: the transform is a leaf, so it does
    not appear in the structure at all. It is the static *values* that
    differ.)
    """

    def build(self, value, upper=1.0):
        model = BarotropicQG.create(nx=16, ny=16)
        return eqx.tree_at(
            lambda m: m.params.bottom_drag, model, interval(value, 0.0, upper)
        )

    def static_of(self, model):
        return eqx.partition(model, eqx.is_array)[1]

    def ensemble(self, members):
        arrays = [eqx.partition(m, eqx.is_array)[0] for m in members]
        stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *arrays)
        return eqx.combine(stacked, self.static_of(members[0]))

    def test_same_bounds_give_equal_static_parts(self):
        same = eqx.tree_equal(
            self.static_of(self.build(0.2)), self.static_of(self.build(0.4))
        )
        assert same is True

    def test_different_bounds_stay_distinguishable(self):
        """Otherwise combining would silently apply the wrong constraint."""
        same = eqx.tree_equal(
            self.static_of(self.build(0.2)),
            self.static_of(self.build(0.2, upper=2.0)),
        )
        assert same is not True

    def test_two_models_combine_into_an_ensemble(self):
        ensemble = self.ensemble([self.build(0.2), self.build(0.4)])
        assert paramax.unwrap(ensemble).params.bottom_drag.shape == (2,)

    def test_the_constraint_still_holds_after_combining(self):
        ensemble = self.ensemble([self.build(0.2), self.build(0.4)])
        got = np.asarray(paramax.unwrap(ensemble).params.bottom_drag)
        np.testing.assert_allclose(got, [0.2, 0.4], rtol=1e-5)


class TestPublicFactoriesAcceptConstrainedValues:
    """``create(lateral_viscosity=positive(...))`` must simply work.

    The factories used to call ``jnp.array`` on every parameter, which
    cannot convert a paramax wrapper, so a constrained model could only
    be built by surgery with ``eqx.tree_at``.
    """

    def test_barotropic_qg(self):
        model = BarotropicQG.create(nx=16, ny=16, lateral_viscosity=positive(100.0))
        assert float(paramax.unwrap(model).params.lateral_viscosity) == pytest.approx(
            100.0, rel=1e-5
        )

    def test_a_frozen_parameter_too(self):
        model = BarotropicQG.create(nx=16, ny=16, bottom_drag=frozen(1.0e-7))
        assert isinstance(model.params.bottom_drag, paramax.NonTrainable)

    def test_mixed_plain_and_constrained(self):
        model = BarotropicQG.create(
            nx=16, ny=16, lateral_viscosity=positive(100.0), bottom_drag=1.0e-7
        )
        unwrapped = paramax.unwrap(model)
        assert float(unwrapped.params.bottom_drag) == pytest.approx(1.0e-7)
        assert float(unwrapped.params.lateral_viscosity) == pytest.approx(
            100.0, rel=1e-5
        )

    def test_plain_values_are_unchanged(self):
        model = BarotropicQG.create(nx=16, ny=16, lateral_viscosity=100.0)
        assert isinstance(model.params.lateral_viscosity, jnp.ndarray)

    def test_as_parameter_passes_wrappers_through(self):
        wrapper = positive(1.0)
        assert as_parameter(wrapper) is wrapper

    def test_as_parameter_converts_everything_else(self):
        assert isinstance(as_parameter(2.0), jnp.ndarray)


class TestDirectVectorFieldEvaluation:
    """A constrained model must behave like its plain equivalent.

    Calling ``model.vector_field(...)`` directly is established usage —
    the differentiable-model tooling does exactly that — and used to
    hand the model implementation the wrapper itself.
    """

    def models(self):
        plain = BarotropicQG.create(nx=16, ny=16, lateral_viscosity=100.0)
        constrained = BarotropicQG.create(
            nx=16, ny=16, lateral_viscosity=positive(100.0)
        )
        return plain, constrained

    def state(self, model):
        rng = np.random.RandomState(0)
        q = jnp.asarray(1.0e-6 * rng.randn(model.grid.Ny, model.grid.Nx))
        return BarotropicQGState(q=q)

    def test_tendencies_agree(self):
        plain, constrained = self.models()
        state = self.state(plain)
        np.testing.assert_allclose(
            np.asarray(constrained.vector_field(0.0, state).q),
            np.asarray(plain.vector_field(0.0, state).q),
            rtol=1e-4,
        )

    def test_diagnose_agrees(self):
        plain, constrained = self.models()
        state = self.state(plain)
        np.testing.assert_allclose(
            float(constrained.diagnose(state).kinetic_energy),
            float(plain.diagnose(state).kinetic_energy),
            rtol=1e-5,
        )

    def test_gradients_reach_the_stored_value(self):
        _, constrained = self.models()
        state = self.state(constrained)

        def loss(model):
            return jnp.sum(model.vector_field(0.0, state).q ** 2)

        grads = eqx.filter_grad(loss)(constrained)
        assert np.isfinite(float(grads.params.lateral_viscosity.args[0]))

    def test_it_works_under_jit(self):
        _, constrained = self.models()
        state = self.state(constrained)
        jitted = eqx.filter_jit(lambda m, s: m.vector_field(0.0, s))
        got = np.asarray(jitted(constrained, state).q)
        expected = np.asarray(constrained.vector_field(0.0, state).q)
        # Relative to the field magnitude: jit may reassociate the
        # stencil sums, so a near-cancelling cell can differ in its
        # last bits while the tendency as a whole agrees.
        assert np.abs(got - expected).max() < 1e-6 * np.abs(expected).max()

    def test_a_plain_model_is_unaffected(self):
        plain, _ = self.models()
        state = self.state(plain)
        assert np.isfinite(np.asarray(plain.vector_field(0.0, state).q)).all()


class TestTrainableMask:
    """A zero gradient is not the same as no update.

    ``optax.adamw`` decouples weight decay, so it moves a parameter
    from its own value even when the gradient is exactly zero — which
    would drift a supposedly frozen physical constant during
    calibration.
    """

    def model(self):
        return BarotropicQG.create(
            nx=16, ny=16, lateral_viscosity=100.0, bottom_drag=frozen(1.0e-7)
        )

    def test_frozen_leaves_are_marked_false(self):
        mask = trainable_mask(self.model())
        assert mask.params.bottom_drag.tree is False

    def test_ordinary_leaves_are_marked_true(self):
        mask = trainable_mask(self.model())
        assert mask.params.lateral_viscosity is True

    def test_the_mask_matches_the_model_structure(self):
        model = self.model()
        mask = trainable_mask(model)
        assert jax.tree_util.tree_structure(
            eqx.filter(mask, eqx.is_array_like)
        ) == jax.tree_util.tree_structure(eqx.filter(model, eqx.is_array_like))

    def test_decoupled_weight_decay_would_move_a_frozen_value(self):
        """The failure the mask exists to prevent, shown directly."""
        optax = pytest.importorskip("optax")
        model = self.model()
        params = eqx.filter(model, eqx.is_inexact_array)
        grads = jax.tree_util.tree_map(jnp.zeros_like, params)

        unmasked = optax.adamw(1e-3, weight_decay=0.5)
        state = unmasked.init(params)
        updates, _ = unmasked.update(grads, state, params)
        moved = float(updates.params.bottom_drag.tree)
        assert moved != 0.0

    def test_the_mask_stops_it(self):
        optax = pytest.importorskip("optax")
        model = self.model()
        params = eqx.filter(model, eqx.is_inexact_array)
        grads = jax.tree_util.tree_map(jnp.zeros_like, params)

        mask = eqx.filter(trainable_mask(model), eqx.is_array_like)
        masked = optax.masked(optax.adamw(1e-3, weight_decay=0.5), mask)
        state = masked.init(params)
        updates, _ = masked.update(grads, state, params)
        assert float(updates.params.bottom_drag.tree) == 0.0
