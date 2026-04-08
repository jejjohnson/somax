"""Tests for 2D PDE models."""

from __future__ import annotations

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp

from somax import SomaxModel
from somax.models import (
    Burgers2D,
    Burgers2DState,
    Diffusion2D,
    Diffusion2DState,
    LinearConvection2D,
    LinearConvection2DState,
    NonlinearConvection2D,
    NonlinearConvection2DState,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gaussian_2d(x, y, mux, muy, sigma):
    return jnp.exp(-0.5 * (((x - mux) / sigma) ** 2 + ((y - muy) / sigma) ** 2))


def _make_coords(grid):
    x = jnp.arange(grid.Nx) * grid.dx
    y = jnp.arange(grid.Ny) * grid.dy
    return jnp.meshgrid(x, y)


# ---------------------------------------------------------------------------
# LinearConvection2D
# ---------------------------------------------------------------------------


class TestLinearConvection2D:
    def test_is_somax_model(self):
        model = LinearConvection2D.create(nx=16, ny=16)
        assert isinstance(model, SomaxModel)

    def test_integrate_finite(self):
        model = LinearConvection2D.create(nx=16, ny=16)
        X, Y = _make_coords(model.grid)
        u0 = _gaussian_2d(X, Y, 1.0, 1.0, 0.3)
        state0 = LinearConvection2DState(u=u0)
        sol = model.integrate(state0, t0=0.0, t1=0.01, dt=0.001)
        assert jnp.all(jnp.isfinite(sol.ys.u))

    def test_grad_through_params(self):
        model = LinearConvection2D.create(nx=16, ny=16)
        X, Y = _make_coords(model.grid)
        state0 = LinearConvection2DState(u=_gaussian_2d(X, Y, 1.0, 1.0, 0.3))

        @eqx.filter_grad
        def grad_fn(m):
            sol = m.integrate(state0, t0=0.0, t1=0.01, dt=0.001)
            return jnp.sum(sol.ys.u**2)

        grads = grad_fn(model)
        assert jnp.isfinite(grads.params.cx)
        assert jnp.isfinite(grads.params.cy)


# ---------------------------------------------------------------------------
# NonlinearConvection2D
# ---------------------------------------------------------------------------


class TestNonlinearConvection2D:
    def test_is_somax_model(self):
        model = NonlinearConvection2D.create(nx=16, ny=16)
        assert isinstance(model, SomaxModel)

    def test_integrate_finite(self):
        model = NonlinearConvection2D.create(nx=16, ny=16)
        X, Y = _make_coords(model.grid)
        g = _gaussian_2d(X, Y, 1.0, 1.0, 0.3)
        state0 = NonlinearConvection2DState(u=g, v=g)
        sol = model.integrate(state0, t0=0.0, t1=0.01, dt=0.001)
        assert jnp.all(jnp.isfinite(sol.ys.u))
        assert jnp.all(jnp.isfinite(sol.ys.v))


# ---------------------------------------------------------------------------
# Diffusion2D
# ---------------------------------------------------------------------------


class TestDiffusion2D:
    def test_is_somax_model(self):
        model = Diffusion2D.create(nx=16, ny=16)
        assert isinstance(model, SomaxModel)

    def test_gaussian_spreads(self):
        """Peak should decrease under diffusion."""
        model = Diffusion2D.create(nx=32, ny=32, Lx=4.0, Ly=4.0, nu=0.1)
        X, Y = _make_coords(model.grid)
        u0 = _gaussian_2d(X, Y, 2.0, 2.0, 0.3)
        state0 = Diffusion2DState(u=u0)
        sol = model.integrate(
            state0, t0=0.0, t1=0.2, dt=0.001, saveat=dfx.SaveAt(t1=True)
        )
        assert float(jnp.max(sol.ys.u[0])) < float(jnp.max(u0))

    def test_grad_through_params(self):
        model = Diffusion2D.create(nx=16, ny=16, nu=0.05)
        X, Y = _make_coords(model.grid)
        state0 = Diffusion2DState(u=_gaussian_2d(X, Y, 1.0, 1.0, 0.3))

        @eqx.filter_grad
        def grad_fn(m):
            sol = m.integrate(state0, t0=0.0, t1=0.01, dt=0.001)
            return jnp.sum(sol.ys.u**2)

        grads = grad_fn(model)
        assert jnp.isfinite(grads.params.nu)


# ---------------------------------------------------------------------------
# Burgers2D
# ---------------------------------------------------------------------------


class TestBurgers2D:
    def test_is_somax_model(self):
        model = Burgers2D.create(nx=16, ny=16)
        assert isinstance(model, SomaxModel)

    def test_integrate_finite(self):
        model = Burgers2D.create(nx=16, ny=16, nu=0.05)
        X, Y = _make_coords(model.grid)
        g = _gaussian_2d(X, Y, 1.0, 1.0, 0.3)
        state0 = Burgers2DState(u=g, v=g)
        sol = model.integrate(state0, t0=0.0, t1=0.01, dt=0.001)
        assert jnp.all(jnp.isfinite(sol.ys.u))
        assert jnp.all(jnp.isfinite(sol.ys.v))

    def test_grad_through_params(self):
        model = Burgers2D.create(nx=16, ny=16, nu=0.05)
        X, Y = _make_coords(model.grid)
        g = _gaussian_2d(X, Y, 1.0, 1.0, 0.3)
        state0 = Burgers2DState(u=g, v=g)

        @eqx.filter_grad
        def grad_fn(m):
            sol = m.integrate(state0, t0=0.0, t1=0.01, dt=0.001)
            return jnp.sum(sol.ys.u**2 + sol.ys.v**2)

        grads = grad_fn(model)
        assert jnp.isfinite(grads.params.nu)
