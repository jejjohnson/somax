"""Tests for the SomaxModel base class."""

from __future__ import annotations

import diffrax as dfx
import jax.numpy as jnp
import pytest

from somax.core import SomaxModel


class ExponentialDecay(SomaxModel):
    """Minimal concrete model: dx/dt = -x."""

    rate: float = 1.0

    def vector_field(self, t, state, args=None):
        return -self.rate * state

    def apply_boundary_conditions(self, state):
        return state


class MissingBCs(SomaxModel):
    """Model that only implements vector_field."""

    def vector_field(self, t, state, args=None):
        return -state


def test_cannot_instantiate_abstract():
    """SomaxModel cannot be instantiated directly."""
    with pytest.raises(TypeError):
        SomaxModel()


def test_concrete_model_instantiation():
    model = ExponentialDecay(rate=2.0)
    assert model.rate == 2.0


def test_vector_field():
    model = ExponentialDecay(rate=1.0)
    state = jnp.array([1.0, 2.0, 3.0])
    rhs = model.vector_field(0.0, state)
    assert jnp.allclose(rhs, -state)


def test_build_terms_returns_ode_term():
    model = ExponentialDecay()
    terms = model.build_terms()
    assert isinstance(terms, dfx.ODETerm)


def test_integrate_produces_solution():
    model = ExponentialDecay(rate=1.0)
    state0 = jnp.array([1.0])
    sol = model.integrate(state0, t0=0.0, t1=1.0, dt=0.01)
    assert isinstance(sol, dfx.Solution)
    # After t=1 with rate=1, x ≈ e^{-1} ≈ 0.368
    assert jnp.allclose(sol.ys[-1], jnp.exp(-1.0), atol=1e-3)


def test_integrate_with_saveat():
    model = ExponentialDecay(rate=1.0)
    state0 = jnp.array([1.0])
    sol = model.integrate(
        state0,
        t0=0.0,
        t1=1.0,
        dt=0.01,
        saveat=dfx.SaveAt(ts=jnp.array([0.0, 0.5, 1.0])),
    )
    assert sol.ys.shape == (3, 1)


def test_diagnose_returns_empty_by_default():
    model = ExponentialDecay()
    state = jnp.array([1.0])
    diag = model.diagnose(state)
    assert diag == {}


def test_missing_apply_bcs_raises():
    """Model missing apply_boundary_conditions cannot be instantiated."""
    with pytest.raises(TypeError):
        MissingBCs()
