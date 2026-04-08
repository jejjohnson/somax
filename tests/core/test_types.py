"""Tests for the somax core type system."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from somax.core import Diagnostics, Params, PhysConsts, State


class MyState(State):
    x: jax.Array
    y: jax.Array


class MyParams(Params):
    alpha: jax.Array
    beta: jax.Array


class MyConsts(PhysConsts):
    gravity: float = 9.81


class MyDiagnostics(Diagnostics):
    energy: jax.Array


def test_state_subclass():
    state = MyState(x=jnp.ones(3), y=jnp.zeros(3))
    assert state.x.shape == (3,)
    assert state.y.shape == (3,)


def test_params_subclass():
    params = MyParams(alpha=jnp.array(1.0), beta=jnp.array(2.0))
    assert params.alpha == 1.0
    assert params.beta == 2.0


def test_physconsts_subclass():
    consts = MyConsts()
    assert consts.gravity == 9.81


def test_diagnostics_subclass():
    diag = MyDiagnostics(energy=jnp.array(42.0))
    assert diag.energy == 42.0


def test_state_is_pytree():
    state = MyState(x=jnp.ones(3), y=jnp.zeros(3))
    leaves, treedef = jax.tree_util.tree_flatten(state)
    assert len(leaves) == 2
    reconstructed = treedef.unflatten(leaves)
    assert jnp.array_equal(reconstructed.x, state.x)
    assert jnp.array_equal(reconstructed.y, state.y)


def test_params_is_pytree():
    params = MyParams(alpha=jnp.array(1.0), beta=jnp.array(2.0))
    doubled = jax.tree_util.tree_map(lambda x: x * 2, params)
    assert doubled.alpha == 2.0
    assert doubled.beta == 4.0


def test_params_visible_to_grad():
    """Params fields should be differentiable."""

    def loss_fn(params: MyParams) -> jax.Array:
        return jnp.sum(params.alpha**2 + params.beta**2)

    params = MyParams(alpha=jnp.array(3.0), beta=jnp.array(4.0))
    grads = jax.grad(loss_fn)(params)
    assert jnp.isclose(grads.alpha, 6.0)
    assert jnp.isclose(grads.beta, 8.0)
