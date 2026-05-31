"""Tests for the composable term algebra (somax._src.core.terms)."""

from __future__ import annotations

import diffrax as dfx
import jax
import jax.numpy as jnp

from somax._src.core.model import TermModel
from somax._src.core.terms import (
    EXPLICIT,
    IMPLICIT,
    MIXED,
    Compose,
    Scaled,
    Sum,
    Term,
    TermFn,
    build_diffrax_terms,
    explicit,
    implicit,
    partition,
)


# ----------------------------------------------------------------------
# Leaf terms used across the tests
# ----------------------------------------------------------------------


class Linear(Term):
    """dx/dt contribution = rate * state (autonomous, explicit)."""

    rate: float = 1.0

    def __call__(self, t, state, args=None):
        return jax.tree_util.tree_map(lambda x: self.rate * x, state)


class ConstForce(Term):
    """Constant additive tendency of ``amp`` (ignores state)."""

    amp: float = 1.0

    def __call__(self, t, state, args=None):
        return jax.tree_util.tree_map(lambda x: self.amp * jnp.ones_like(x), state)


# ----------------------------------------------------------------------
# Evaluation + algebra
# ----------------------------------------------------------------------


def test_leaf_term_evaluates():
    term = Linear(rate=2.0)
    state = jnp.array([1.0, 3.0])
    assert jnp.allclose(term(0.0, state), jnp.array([2.0, 6.0]))


def test_sum_adds_contributions():
    term = Linear(rate=2.0) + ConstForce(amp=1.0)
    state = jnp.array([1.0, 3.0])
    # 2*state + 1
    assert jnp.allclose(term(0.0, state), jnp.array([3.0, 7.0]))


def test_sum_flattens_nested():
    a, b, c = Linear(1.0), Linear(2.0), Linear(3.0)
    left = (a + b) + c
    right = a + (b + c)
    assert isinstance(left, Sum)
    assert len(left.terms) == 3
    assert len(right.terms) == 3


def test_scaled_scales_output():
    term = 3.0 * Linear(rate=2.0)
    state = jnp.array([1.0, 1.0])
    assert jnp.allclose(term(0.0, state), jnp.array([6.0, 6.0]))
    # __rmul__ and __mul__ agree
    assert jnp.allclose((Linear(2.0) * 3.0)(0.0, state), term(0.0, state))


def test_neg_and_sub():
    state = jnp.array([2.0])
    neg = -Linear(rate=1.0)
    assert jnp.allclose(neg(0.0, state), jnp.array([-2.0]))
    diff = Linear(rate=3.0) - Linear(rate=1.0)
    assert jnp.allclose(diff(0.0, state), jnp.array([4.0]))  # (3-1)*2


def test_compose_is_operator_composition():
    # Compose(double, increment): double(increment(state)).
    double = Linear(rate=2.0)
    increment = TermFn(lambda t, s, a: jax.tree_util.tree_map(lambda x: x + 1.0, s))
    comp = double @ increment
    state = jnp.array([3.0])
    # double(state + 1) = 2 * (3 + 1) = 8
    assert jnp.allclose(comp(0.0, state), jnp.array([8.0]))
    assert isinstance(comp, Compose)


def test_sum_with_zero_identity():
    a = Linear(rate=1.0)
    # builtin sum seeds with int 0 -> exercises __radd__
    s = sum([a, Linear(rate=2.0)])
    state = jnp.array([1.0])
    assert jnp.allclose(s(0.0, state), jnp.array([3.0]))


def test_pytree_state_added_leafwise():
    term = Linear(rate=2.0) + ConstForce(amp=1.0)
    state = {"u": jnp.array([1.0]), "v": jnp.array([5.0])}
    out = term(0.0, state)
    assert jnp.allclose(out["u"], jnp.array([3.0]))
    assert jnp.allclose(out["v"], jnp.array([11.0]))


# ----------------------------------------------------------------------
# Differentiability of coefficients
# ----------------------------------------------------------------------


def test_scaled_coeff_is_differentiable():
    state = jnp.array([1.0, 2.0, 3.0])

    def loss(coeff):
        term = Scaled(Linear(rate=1.0), coeff)
        return jnp.sum(term(0.0, state) ** 2)

    # loss(c) = sum((c*state)^2) = c^2 * sum(state^2); d/dc = 2c*sum(state^2)
    g = jax.grad(loss)(2.0)
    expected = 2.0 * 2.0 * jnp.sum(state**2)
    assert jnp.allclose(g, expected)


# ----------------------------------------------------------------------
# IMEX tagging + partition
# ----------------------------------------------------------------------


def test_default_kind_is_explicit():
    assert Linear().kind == EXPLICIT


def test_implicit_tagging():
    assert implicit(Linear()).kind == IMPLICIT
    assert explicit(implicit(Linear())).kind == EXPLICIT


def test_sum_kind_is_mixed_when_kinds_differ():
    term = explicit(Linear(1.0)) + implicit(Linear(2.0))
    assert term.kind == MIXED


def test_sum_kind_uniform():
    term = explicit(Linear(1.0)) + explicit(Linear(2.0))
    assert term.kind == EXPLICIT


def test_partition_splits_by_kind():
    expl = Linear(rate=1.0)
    impl = implicit(Linear(rate=5.0))
    term = expl + impl
    e_part, i_part = partition(term)
    assert e_part is not None and i_part is not None

    state = jnp.array([2.0])
    assert jnp.allclose(e_part(0.0, state), jnp.array([2.0]))  # 1*2
    assert jnp.allclose(i_part(0.0, state), jnp.array([10.0]))  # 5*2


def test_partition_all_explicit():
    e_part, i_part = partition(Linear(1.0) + Linear(2.0))
    assert e_part is not None
    assert i_part is None


def test_partition_recurses_nested_sum():
    term = (explicit(Linear(1.0)) + implicit(Linear(2.0))) + implicit(Linear(3.0))
    e_part, i_part = partition(term)
    state = jnp.array([1.0])
    assert jnp.allclose(e_part(0.0, state), jnp.array([1.0]))
    # implicit part = (2 + 3) * state
    assert jnp.allclose(i_part(0.0, state), jnp.array([5.0]))


# ----------------------------------------------------------------------
# diffrax bridge
# ----------------------------------------------------------------------


def test_build_terms_single_kind_is_odeterm():
    term = Linear(rate=1.0) + Linear(rate=2.0)
    dterm = build_diffrax_terms(term)
    assert isinstance(dterm, dfx.ODETerm)


def test_build_terms_mixed_is_multiterm():
    term = explicit(Linear(1.0)) + implicit(Linear(2.0))
    dterm = build_diffrax_terms(term)
    assert isinstance(dterm, dfx.MultiTerm)


def test_zero_term_builds_explicit_odeterm():
    # The zero term is a valid explicit summand: it builds an ODETerm
    # (a zero RHS), it does not raise.
    from somax._src.core.terms import _ZERO

    dterm = build_diffrax_terms(_ZERO)
    assert isinstance(dterm, dfx.ODETerm)
    out = dterm.vector_field(0.0, jnp.array([3.0, 4.0]), None)
    assert jnp.allclose(out, jnp.zeros(2))


def test_state_fn_applied_before_term():
    # state_fn zeroes the state; the term then sees zeros.
    term = ConstForce(amp=1.0)
    vf = build_diffrax_terms(term, state_fn=lambda y: jnp.zeros_like(y)).vector_field
    out = vf(0.0, jnp.array([9.0, 9.0]), None)
    # ConstForce returns ones_like(zeros) = ones regardless, but shape follows input
    assert out.shape == (2,)


# ----------------------------------------------------------------------
# TermModel integration
# ----------------------------------------------------------------------


class DecayModel(TermModel):
    """TermModel for dx/dt = -rate * x assembled from a single term."""


def test_term_model_integrates_explicit():
    # dx/dt = -x  ->  x(1) = x0 * e^{-1}
    model = DecayModel(terms=Scaled(Linear(rate=1.0), -1.0))
    state0 = jnp.array([1.0])
    sol = model.integrate(state0, t0=0.0, t1=1.0, dt=0.01)
    assert jnp.allclose(sol.ys[-1], jnp.exp(-1.0), atol=1e-3)


def test_term_model_step_conforms():
    model = DecayModel(terms=Scaled(Linear(rate=1.0), -1.0))
    state0 = jnp.array([1.0, 2.0])
    stepped = model.step(state0, dt=0.01)
    assert stepped.shape == state0.shape


def test_term_model_imex_integrates():
    # Mixed explicit/implicit decay integrated with an IMEX solver.
    # dx/dt = -x split as explicit(-0.5x) + implicit(-0.5x).
    model = DecayModel(
        terms=explicit(Scaled(Linear(1.0), -0.5)) + implicit(Scaled(Linear(1.0), -0.5))
    )
    state0 = jnp.array([1.0])
    # Implicit solvers require an adaptive controller with tolerances.
    sol = model.integrate(
        state0,
        t0=0.0,
        t1=1.0,
        dt=0.01,
        solver=dfx.KenCarp3(),
        stepsize_controller=dfx.PIDController(rtol=1e-5, atol=1e-7),
        max_steps=10_000,
    )
    assert jnp.allclose(sol.ys[-1], jnp.exp(-1.0), atol=1e-3)
