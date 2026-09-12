"""Tests for constrained parameters via paramax.

The contract: a model whose ``Params`` hold wrapped values behaves
exactly like one holding the equivalent plain arrays, gradients land on
the stored unconstrained values, and the constraint cannot be violated
by a gradient step.
"""

from __future__ import annotations

import dataclasses
import math

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

    def test_infinity_is_rejected(self):
        """+inf passes ``> 0``, but its softplus pre-image is not finite.

        Storing it would unwrap back to ``inf`` — an overflowed
        calibration value reaching the integration rather than the
        error the docstring promises.
        """
        with pytest.raises(ValueError, match="finite"):
            positive(jnp.inf)

    def test_infinity_in_an_array_is_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            positive(jnp.asarray([1.0, jnp.inf]))

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


class TestIntervalBoundsMustBeFinite:
    """A semi-infinite interval has no logit.

    ``interval(0.5, 0.0, inf)`` passes the ordering test, but ``scaled``
    collapses to zero, the stored raw value becomes ``-inf``, and
    unwrapping evaluates ``inf * sigmoid(-inf)`` — NaN.
    """

    @pytest.mark.parametrize(
        ("lower", "upper"),
        [(0.0, float("inf")), (float("-inf"), 1.0), (float("-inf"), float("inf"))],
    )
    def test_an_infinite_bound_is_rejected(self, lower, upper):
        with pytest.raises(ValueError, match="bounds must be finite"):
            interval(0.5, lower, upper)

    def test_a_nan_bound_is_rejected(self):
        with pytest.raises(ValueError, match="bounds must be finite"):
            interval(0.5, 0.0, float("nan"))

    def test_finite_bounds_still_work(self):
        assert float(paramax.unwrap(interval(0.5, 0.0, 1.0))) == pytest.approx(0.5)

    def test_the_error_points_at_positive(self):
        """A one-sided lower bound is what ``positive`` is for."""
        with pytest.raises(ValueError, match="positive"):
            interval(0.5, 0.0, float("inf"))


class TestTermModelFactoriesAcceptConstrainedValues:
    """The term-model factories were missed by the earlier fix.

    They convert with ``jnp.asarray`` outside a ``Params``
    construction, so a repo-wide search for ``*Params(`` did not reach
    them and they still raised on a wrapper.
    """

    def test_burgers_term_model_create(self):
        from somax._src.models.pde2d.burgers_terms import Burgers2DTermModel

        model = Burgers2DTermModel.create(nx=16, ny=16, nu=positive(0.05))
        assert model is not None

    def test_burgers_term_model_from_model(self):
        from somax._src.models.pde2d.burgers import Burgers2D
        from somax._src.models.pde2d.burgers_terms import Burgers2DTermModel

        plain = Burgers2D.create(nx=16, ny=16, nu=positive(0.05))
        assert Burgers2DTermModel.from_model(plain) is not None

    def test_the_imex_split_survives_a_wrapper(self):
        import diffrax as dfx

        from somax._src.models.pde2d.burgers_terms import Burgers2DTermModel

        model = Burgers2DTermModel.create(nx=16, ny=16, nu=positive(0.05), imex=True)
        assert isinstance(model.build_terms(), dfx.MultiTerm)

    def test_linear_swm_term_model_from_model(self):
        from somax._src.models.swm.linear_2d import LinearShallowWater2D
        from somax._src.models.swm.linear_swm_terms import LinearSWM2DTermModel

        plain = LinearShallowWater2D.create(
            nx=16, ny=16, lateral_viscosity=positive(10.0)
        )
        assert LinearSWM2DTermModel.from_model(plain) is not None

    def test_the_wrapped_model_evaluates(self):
        from somax._src.models.pde2d.burgers import Burgers2DState
        from somax._src.models.pde2d.burgers_terms import Burgers2DTermModel

        model = Burgers2DTermModel.create(nx=16, ny=16, nu=positive(0.05))
        shape = (model.grid.Ny, model.grid.Nx)
        rng = np.random.RandomState(0)
        state = Burgers2DState(
            u=jnp.asarray(rng.randn(*shape)), v=jnp.asarray(rng.randn(*shape))
        )
        out = model.build_terms().vector_field(0.0, state, None)
        assert np.isfinite(np.asarray(out.u)).all()

    def test_it_matches_the_plain_equivalent(self):
        from somax._src.models.pde2d.burgers import Burgers2DState
        from somax._src.models.pde2d.burgers_terms import Burgers2DTermModel

        outs = []
        for nu in (0.05, positive(0.05)):
            model = Burgers2DTermModel.create(nx=16, ny=16, nu=nu)
            shape = (model.grid.Ny, model.grid.Nx)
            rng = np.random.RandomState(0)
            state = Burgers2DState(
                u=jnp.asarray(rng.randn(*shape)), v=jnp.asarray(rng.randn(*shape))
            )
            outs.append(
                np.asarray(model.build_terms().vector_field(0.0, state, None).u)
            )
        # Against the field magnitude, not pointwise: ``positive(0.05)``
        # stores an inverse-softplus and unwraps back through softplus,
        # which round-trips to ~1e-8 relative in float32 — enough to
        # read as 1e-5 in a cell where the tendency nearly cancels.
        assert np.abs(outs[0] - outs[1]).max() < 1e-5 * np.abs(outs[0]).max()


class TestTrainableMaskCoversOnlyArrays:
    """A ``Parameterize`` keeps its transform as an ordinary leaf.

    Marking that leaf ``True`` hands the optimiser a Python function to
    initialise state for, which it cannot do — so a model that mixes a
    constrained parameter with a frozen one was unusable with the very
    mask the docstring recommends.
    """

    @staticmethod
    def _mixed_model():
        model = BarotropicQG.create(nx=8, ny=8)
        return eqx.tree_at(
            lambda m: (m.params.lateral_viscosity, m.params.bottom_drag),
            model,
            (positive(100.0), frozen(1e-7)),
        )

    def test_the_transform_leaf_is_not_marked_trainable(self):
        mask = trainable_mask(self._mixed_model())
        flat = jax.tree_util.tree_leaves(mask)
        assert all(isinstance(m, bool) for m in flat)
        # The softplus leaf is in there; it must not be True.
        assert not all(flat)

    def test_adamw_initialises_on_the_mixed_model(self):
        optax = pytest.importorskip("optax")
        model = self._mixed_model()
        optimiser = optax.masked(optax.adamw(1e-3), trainable_mask(model))
        # Without the fix this raises while building the AdamW state
        # for the function leaf.
        optimiser.init(model)

    def test_the_frozen_leaf_is_still_masked_out(self):
        mask = trainable_mask(self._mixed_model())
        assert mask.params.bottom_drag.tree is False

    def test_a_plain_array_parameter_is_still_trainable(self):
        mask = trainable_mask(BarotropicQG.create(nx=8, ny=8))
        assert mask.params.lateral_viscosity is True


class TestIntervalSpansMustBeRepresentable:
    """Finite endpoints do not make their difference finite."""

    def test_a_span_that_overflows_float32_is_rejected(self):
        with pytest.raises(ValueError, match="overflows float32"):
            interval(0.0, -3e38, 3e38)

    def test_the_span_really_would_overflow(self):
        """Otherwise the test above would pass for the wrong reason."""
        assert math.isfinite(3e38 - -3e38)
        assert not bool(
            jnp.isfinite(
                jnp.asarray(3e38, jnp.float32) - jnp.asarray(-3e38, jnp.float32)
            )
        )

    def test_the_same_span_is_fine_in_float64(self):
        jax.config.update("jax_enable_x64", True)
        try:
            wrapped = interval(jnp.asarray(0.0, jnp.float64), -3e38, 3e38)
            assert bool(jnp.isfinite(paramax.unwrap(wrapped)))
        finally:
            jax.config.update("jax_enable_x64", False)

    def test_an_ordinary_interval_is_unaffected(self):
        assert float(paramax.unwrap(interval(0.5, 0.0, 1.0))) == pytest.approx(0.5)


class TestConstrainedModelsRoundTripThroughACheckpoint:
    """Orbax has no handler for the transform callable in ``Parameterize``."""

    @staticmethod
    def _constrained_model():
        model = BarotropicQG.create(nx=8, ny=8)
        return eqx.tree_at(lambda m: m.params.lateral_viscosity, model, positive(100.0))

    def test_it_saves_and_restores(self, tmp_path):
        from somax._src.core.checkpoint import SimulationCheckpointer

        model = self._constrained_model()
        state = BarotropicQGState(q=jnp.zeros((10, 10)))
        ckpt = SimulationCheckpointer(
            checkpoint_dir=str(tmp_path), checkpoint_interval=1
        )
        ckpt.save(1, state, model)
        _, params, step = ckpt.restore(1, state, model)

        assert step == 1
        assert float(paramax.unwrap(params).lateral_viscosity) == pytest.approx(
            100.0, rel=1e-5
        )

    def test_the_restored_parameter_is_still_a_wrapper(self, tmp_path):
        from somax._src.core.checkpoint import SimulationCheckpointer

        model = self._constrained_model()
        state = BarotropicQGState(q=jnp.zeros((10, 10)))
        ckpt = SimulationCheckpointer(
            checkpoint_dir=str(tmp_path), checkpoint_interval=1
        )
        ckpt.save(1, state, model)
        _, params, _ = ckpt.restore(1, state, model)
        assert isinstance(params.lateral_viscosity, paramax.Parameterize)

    def test_a_plain_model_still_round_trips(self, tmp_path):
        from somax._src.core.checkpoint import SimulationCheckpointer

        model = BarotropicQG.create(nx=8, ny=8)
        state = BarotropicQGState(q=jnp.zeros((10, 10)))
        ckpt = SimulationCheckpointer(
            checkpoint_dir=str(tmp_path), checkpoint_interval=1
        )
        ckpt.save(2, state, model)
        _, params, step = ckpt.restore(2, state, model)
        assert step == 2
        np.testing.assert_allclose(
            np.asarray(params.lateral_viscosity),
            np.asarray(model.params.lateral_viscosity),
        )


class _DragBoundaryQG(BarotropicQG):
    """A model whose boundary condition reads a parameter directly.

    The arithmetic against ``bottom_drag`` is what raises if the
    wrapper has not been unwrapped by the time this is called.
    """

    def apply_boundary_conditions(self, state):
        return BarotropicQGState(q=state.q + 0.0 * self.params.bottom_drag)


class TestBoundaryConditionsSeeUnwrappedParameters:
    """``integrate()`` applies them once before the solve begins.

    That first call bypassed the RHS's unwrapping, so a boundary
    condition that reads a constrained parameter raised on arithmetic
    against the wrapper.
    """

    @staticmethod
    def _retype(base):
        """Rebuild ``base`` as the subclass, field for field."""
        return _DragBoundaryQG(
            **{f.name: getattr(base, f.name) for f in dataclasses.fields(base)}
        )

    @classmethod
    def _constrained(cls):
        model = cls._retype(BarotropicQG.create(nx=8, ny=8))
        return eqx.tree_at(lambda m: m.params.bottom_drag, model, positive(1e-7))

    def test_the_method_is_in_the_unwrapped_list(self):
        from somax._src.core.model import _UNWRAPPED_METHODS

        assert "apply_boundary_conditions" in _UNWRAPPED_METHODS

    def test_it_runs_against_a_constrained_parameter(self):
        model = self._constrained()
        state = BarotropicQGState(q=jnp.ones((10, 10)))
        out = model.apply_boundary_conditions(state)
        np.testing.assert_allclose(np.asarray(out.q), np.asarray(state.q))

    def test_the_wrapper_really_is_still_in_the_model(self):
        """Otherwise the test above would pass for the wrong reason."""
        assert isinstance(self._constrained().params.bottom_drag, paramax.Parameterize)

    def test_a_plain_model_is_unaffected(self):
        model = self._retype(BarotropicQG.create(nx=8, ny=8))
        state = BarotropicQGState(q=jnp.ones((10, 10)))
        np.testing.assert_allclose(
            np.asarray(model.apply_boundary_conditions(state).q),
            np.asarray(state.q),
        )
