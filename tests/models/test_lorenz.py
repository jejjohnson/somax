"""Tests for Lorenz dynamical system models on the SomaxModel contract."""

from __future__ import annotations

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp

from somax.core import Diagnostics, Params, SomaxModel, State
from somax.models import (
    L63Diagnostics,
    L63Params,
    L63State,
    L96Diagnostics,
    L96Params,
    L96State,
    L96TState,
    Lorenz63,
    Lorenz96,
    Lorenz96t,
)


# ---------------------------------------------------------------------------
# Lorenz '63 — State
# ---------------------------------------------------------------------------


class TestL63State:
    def test_init_single(self):
        state = L63State.init_state(noise=0.01, batchsize=1)
        assert state.x.shape == (1,)
        assert state.y.shape == (1,)
        assert state.z.shape == (1,)

    def test_init_batch(self):
        state = L63State.init_state(noise=0.01, batchsize=5)
        assert state.x.shape == (5, 1)

    def test_is_state_subclass(self):
        state = L63State.init_state()
        assert isinstance(state, State)

    def test_is_pytree(self):
        state = L63State.init_state()
        leaves, treedef = jax.tree_util.tree_flatten(state)
        assert len(leaves) == 3
        reconstructed = treedef.unflatten(leaves)
        assert jnp.array_equal(reconstructed.x, state.x)

    def test_array_property(self):
        state = L63State(x=jnp.array(1.0), y=jnp.array(2.0), z=jnp.array(3.0))
        arr = state.array
        assert arr.shape == (3,)
        assert jnp.allclose(arr, jnp.array([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# Lorenz '63 — Params
# ---------------------------------------------------------------------------


class TestL63Params:
    def test_is_params_subclass(self):
        params = L63Params(
            sigma=jnp.array(10.0),
            rho=jnp.array(28.0),
            beta=jnp.array(8.0 / 3.0),
        )
        assert isinstance(params, Params)

    def test_visible_to_grad(self):
        """Params fields are differentiable (not static)."""

        def loss_fn(params: L63Params) -> jax.Array:
            return jnp.sum(params.sigma**2 + params.rho**2 + params.beta**2)

        params = L63Params(
            sigma=jnp.array(3.0),
            rho=jnp.array(4.0),
            beta=jnp.array(5.0),
        )
        grads = jax.grad(loss_fn)(params)
        assert jnp.isclose(grads.sigma, 6.0)
        assert jnp.isclose(grads.rho, 8.0)
        assert jnp.isclose(grads.beta, 10.0)


# ---------------------------------------------------------------------------
# Lorenz '63 — RHS pure function
# ---------------------------------------------------------------------------


class TestL63RHS:
    def test_finite_output(self):
        from somax._src.models.lorenz63 import rhs_lorenz_63

        x_dot, y_dot, z_dot = rhs_lorenz_63(
            x=jnp.array(1.0),
            y=jnp.array(1.0),
            z=jnp.array(1.0),
        )
        assert jnp.isfinite(x_dot)
        assert jnp.isfinite(y_dot)
        assert jnp.isfinite(z_dot)

    def test_fixed_point_origin(self):
        """At the origin with rho < 1, RHS should be zero."""
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


# ---------------------------------------------------------------------------
# Lorenz '63 — Model
# ---------------------------------------------------------------------------


class TestLorenz63Model:
    def test_is_somax_model(self):
        model = Lorenz63.create()
        assert isinstance(model, SomaxModel)

    def test_vector_field(self):
        model = Lorenz63.create()
        state = L63State(x=jnp.array(1.0), y=jnp.array(1.0), z=jnp.array(1.0))
        rhs = model.vector_field(0.0, state)
        assert isinstance(rhs, L63State)
        assert jnp.isfinite(rhs.x)
        assert jnp.isfinite(rhs.y)
        assert jnp.isfinite(rhs.z)

    def test_apply_bcs_identity(self):
        model = Lorenz63.create()
        state = L63State(x=jnp.array(1.0), y=jnp.array(2.0), z=jnp.array(3.0))
        result = model.apply_boundary_conditions(state)
        assert jnp.array_equal(result.x, state.x)

    def test_build_terms(self):
        model = Lorenz63.create()
        terms = model.build_terms()
        assert isinstance(terms, dfx.ODETerm)

    def test_integrate(self):
        model = Lorenz63.create()
        state0 = L63State(x=jnp.array(1.0), y=jnp.array(1.0), z=jnp.array(1.0))
        sol = model.integrate(state0, t0=0.0, t1=0.1, dt=0.01)
        assert isinstance(sol, dfx.Solution)
        assert jnp.all(jnp.isfinite(sol.ys.x))

    def test_integrate_saveat(self):
        model = Lorenz63.create()
        state0 = L63State(x=jnp.array(1.0), y=jnp.array(1.0), z=jnp.array(1.0))
        ts = jnp.array([0.0, 0.05, 0.1])
        sol = model.integrate(
            state0,
            t0=0.0,
            t1=0.1,
            dt=0.01,
            saveat=dfx.SaveAt(ts=ts),
        )
        assert sol.ys.x.shape == (3,)

    def test_diagnose_energy(self):
        model = Lorenz63.create()
        state = L63State(x=jnp.array(3.0), y=jnp.array(4.0), z=jnp.array(0.0))
        diag = model.diagnose(state)
        assert isinstance(diag, L63Diagnostics)
        assert isinstance(diag, Diagnostics)
        assert jnp.isclose(diag.energy, 12.5)

    def test_grad_through_params(self):
        """jax.grad flows through model params via integrate."""
        state0 = L63State(x=jnp.array(1.0), y=jnp.array(1.0), z=jnp.array(1.0))

        @eqx.filter_grad
        def grad_fn(model):
            sol = model.integrate(state0, t0=0.0, t1=0.1, dt=0.01)
            return jnp.sum(sol.ys.x**2 + sol.ys.y**2 + sol.ys.z**2)

        model = Lorenz63.create()
        grads = grad_fn(model)
        assert jnp.isfinite(grads.params.sigma)
        assert jnp.isfinite(grads.params.rho)
        assert jnp.isfinite(grads.params.beta)

    def test_vmap_over_ensemble(self):
        """vmap over a batch of initial conditions."""
        model = Lorenz63.create()
        states = L63State(
            x=jnp.array([1.0, 1.1, 0.9]),
            y=jnp.ones(3),
            z=jnp.ones(3),
        )
        rhs_batch = eqx.filter_vmap(model.vector_field, in_axes=(None, 0, None))(
            0.0, states, None
        )
        assert rhs_batch.x.shape == (3,)


# ---------------------------------------------------------------------------
# Lorenz '96 — State & Model
# ---------------------------------------------------------------------------


class TestL96State:
    def test_init_single(self):
        state = L96State.init_state(ndim=10, noise=0.01, batchsize=1)
        assert state.x.shape == (10,)

    def test_init_batch(self):
        state = L96State.init_state(ndim=10, noise=0.01, batchsize=5)
        assert state.x.shape == (5, 10)

    def test_is_state_subclass(self):
        state = L96State.init_state()
        assert isinstance(state, State)


class TestL96Params:
    def test_is_params_subclass(self):
        params = L96Params(F=jnp.array(8.0))
        assert isinstance(params, Params)

    def test_visible_to_grad(self):
        def loss_fn(params: L96Params) -> jax.Array:
            return params.F**2

        params = L96Params(F=jnp.array(3.0))
        grads = jax.grad(loss_fn)(params)
        assert jnp.isclose(grads.F, 6.0)


class TestLorenz96Model:
    def test_is_somax_model(self):
        model = Lorenz96.create()
        assert isinstance(model, SomaxModel)

    def test_vector_field(self):
        model = Lorenz96.create()
        state = L96State.init_state(ndim=10)
        rhs = model.vector_field(0.0, state)
        assert isinstance(rhs, L96State)
        assert jnp.all(jnp.isfinite(rhs.x))

    def test_integrate(self):
        model = Lorenz96.create()
        state0 = L96State.init_state(ndim=10)
        sol = model.integrate(state0, t0=0.0, t1=0.1, dt=0.01)
        assert isinstance(sol, dfx.Solution)
        assert jnp.all(jnp.isfinite(sol.ys.x))

    def test_diagnose(self):
        model = Lorenz96.create()
        state = L96State(x=jnp.ones(10))
        diag = model.diagnose(state)
        assert isinstance(diag, L96Diagnostics)
        assert jnp.isclose(diag.energy, 5.0)
        assert jnp.isclose(diag.mean, 1.0)

    def test_grad_through_F(self):
        state0 = L96State.init_state(ndim=10)

        @eqx.filter_grad
        def grad_fn(model):
            sol = model.integrate(state0, t0=0.0, t1=0.1, dt=0.01)
            return jnp.sum(sol.ys.x**2)

        model = Lorenz96.create()
        grads = grad_fn(model)
        assert jnp.isfinite(grads.params.F)

    def test_vmap_over_ensemble(self):
        model = Lorenz96.create()
        states = L96State(x=jnp.ones((4, 10)) * 8.0)
        rhs_batch = eqx.filter_vmap(model.vector_field, in_axes=(None, 0, None))(
            0.0, states, None
        )
        assert rhs_batch.x.shape == (4, 10)

    def test_no_advection(self):
        model = Lorenz96.create(advection=False)
        state = L96State(x=jnp.ones(10) * 8.0)
        rhs = model.vector_field(0.0, state)
        # Without advection: dx/dt = -x + F = -8 + 8 = 0
        assert jnp.allclose(rhs.x, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Lorenz '96 Two-Tier — State & Model
# ---------------------------------------------------------------------------


class TestL96TState:
    def test_init_single(self):
        state = L96TState.init_state(ndims=(10, 20), batchsize=1)
        assert state.x.shape == (10,)
        assert state.y.shape == (200,)

    def test_init_batch(self):
        state = L96TState.init_state(ndims=(10, 20), batchsize=3)
        assert state.x.shape == (3, 10)
        assert state.y.shape == (3, 200)

    def test_is_state_subclass(self):
        state = L96TState.init_state()
        assert isinstance(state, State)


class TestLorenz96tModel:
    def test_is_somax_model(self):
        model = Lorenz96t.create()
        assert isinstance(model, SomaxModel)

    def test_vector_field(self):
        model = Lorenz96t.create()
        state = L96TState.init_state(ndims=(10, 20))
        rhs = model.vector_field(0.0, state)
        assert isinstance(rhs, L96TState)
        assert jnp.all(jnp.isfinite(rhs.x))
        assert jnp.all(jnp.isfinite(rhs.y))

    def test_integrate(self):
        model = Lorenz96t.create()
        state0 = L96TState.init_state(ndims=(5, 10))
        sol = model.integrate(state0, t0=0.0, t1=0.01, dt=0.001)
        assert isinstance(sol, dfx.Solution)
        assert jnp.all(jnp.isfinite(sol.ys.x))
        assert jnp.all(jnp.isfinite(sol.ys.y))

    def test_grad_through_params(self):
        state0 = L96TState.init_state(ndims=(5, 10))

        @eqx.filter_grad
        def grad_fn(model):
            sol = model.integrate(state0, t0=0.0, t1=0.01, dt=0.001)
            return jnp.sum(sol.ys.x**2)

        model = Lorenz96t.create()
        grads = grad_fn(model)
        assert jnp.isfinite(grads.params.F)
        assert jnp.isfinite(grads.params.h)

    def test_vmap_over_ensemble(self):
        model = Lorenz96t.create()
        states = L96TState.init_state(ndims=(5, 10), batchsize=3)
        rhs_batch = eqx.filter_vmap(model.vector_field, in_axes=(None, 0, None))(
            0.0, states, None
        )
        assert rhs_batch.x.shape == (3, 5)
        assert rhs_batch.y.shape == (3, 50)
