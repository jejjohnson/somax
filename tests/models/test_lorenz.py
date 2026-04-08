"""Tests for Lorenz dynamical system models."""

from __future__ import annotations

import jax.numpy as jnp

from somax.models import (
    L63Params,
    L63State,
    L96State,
    Lorenz63,
    Lorenz96,
)


def test_l63_state_init():
    state = L63State.init_state(noise=0.01, batchsize=1)
    assert state.x.shape == (1,)
    assert state.y.shape == (1,)
    assert state.z.shape == (1,)


def test_l63_state_init_batch():
    state = L63State.init_state(noise=0.01, batchsize=5)
    assert state.x.shape == (5, 1)


def test_l63_state_and_params():
    state, params = L63State.init_state_and_params(sigma=10.0, rho=28.0, beta=8.0 / 3.0)
    assert isinstance(state, L63State)
    assert isinstance(params, L63Params)
    assert params.sigma == 10.0
    assert params.rho == 28.0


def test_l63_state_array_property():
    state = L63State.init_state(noise=0.0, batchsize=1)
    arr = state.array
    assert arr.shape == (3,)


def test_l63_rhs():
    """Test Lorenz63 equation of motion produces finite output."""
    model = Lorenz63()
    state, params = L63State.init_state_and_params()
    rhs = model.equation_of_motion(0.0, state, params)
    assert isinstance(rhs, L63State)
    assert jnp.all(jnp.isfinite(rhs.x))
    assert jnp.all(jnp.isfinite(rhs.y))
    assert jnp.all(jnp.isfinite(rhs.z))


def test_l96_state_init():
    state = L96State.init_state(ndim=10, noise=0.01, batchsize=1)
    assert state.x.shape == (10,)


def test_l96_state_init_batch():
    state = L96State.init_state(ndim=10, noise=0.01, batchsize=5)
    assert state.x.shape == (5, 10)


def test_l96_rhs():
    """Test Lorenz96 equation of motion produces finite output."""
    model = Lorenz96()
    state, params = L96State.init_state_and_params(ndim=10)
    rhs = model.equation_of_motion(0.0, state, params)
    assert isinstance(rhs, L96State)
    assert jnp.all(jnp.isfinite(rhs.x))


def test_l63_known_fixed_point():
    """At the origin, L63 RHS should be zero (with rho < 1)."""
    from somax._src.models.lorenz63 import rhs_lorenz_63

    x_dot, y_dot, z_dot = rhs_lorenz_63(
        x=jnp.array(0.0),
        y=jnp.array(0.0),
        z=jnp.array(0.0),
        sigma=10.0,
        rho=0.5,
        beta=8.0 / 3.0,
    )
    assert jnp.isclose(x_dot, 0.0)
    assert jnp.isclose(y_dot, 0.0)
    assert jnp.isclose(z_dot, 0.0)
