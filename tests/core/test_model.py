"""Tests for the SomaxModel base class."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import diffrax as dfx
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest

from somax.core import SomaxModel


@runtime_checkable
class _ForwardModelLike(Protocol):
    """Local mirror of ``pipekit_cycle.ForwardModel``.

    Defined here so the structural-conformance test does not pull in a
    dependency on pipekit; the contract under test is the ``step`` method.
    """

    def step(self, state: Any, dt: float) -> Any: ...


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


def test_step_returns_bare_state():
    """``step`` returns a bare state matching integrate's final state."""
    model = ExponentialDecay(rate=1.0)
    state0 = jnp.array([1.0, 2.0])

    stepped = model.step(state0, dt=0.01)

    sol = model.integrate(state0, t0=0.0, t1=0.01, dt=0.01)
    final = jtu.tree_map(lambda x: x[-1], sol.ys)

    # same pytree structure / shape as the input state (no leading time axis)
    assert jtu.tree_structure(stepped) == jtu.tree_structure(state0)
    assert stepped.shape == state0.shape
    assert jnp.allclose(stepped, final)


def test_step_composes_to_integration_window():
    """Repeated ``step`` calls match the analytic decay over the window."""
    model = ExponentialDecay(rate=1.0)
    state = jnp.array([1.0, -2.0, 0.5])

    dt = 0.05
    out = state
    for i in range(10):
        out = model.step(out, dt=dt, t0=i * dt)

    # dx/dt = -x integrated over 0.5 -> x * e^{-0.5}
    expected = state * jnp.exp(-0.5)
    assert jnp.allclose(out, expected, rtol=1e-3)


def test_model_satisfies_forward_model_protocol():
    """A somax model structurally satisfies the ForwardModel protocol.

    This is the pipekit-cycle / DA integration seam: any ``SomaxModel``
    is usable as a ``ForwardModel`` (and as filterax dynamics) purely by
    duck typing, with no subclassing.
    """
    model = ExponentialDecay(rate=1.0)
    assert isinstance(model, _ForwardModelLike)
