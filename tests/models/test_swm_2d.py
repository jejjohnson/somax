"""Tests for 2D shallow water models."""

from __future__ import annotations

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp

from somax import Diagnostics, Params, PhysConsts, SomaxModel, State
from somax.models import (
    LinearShallowWater2D,
    LinearSW2DDiagnostics,
    LinearSW2DParams,
    LinearSW2DPhysConsts,
    LinearSW2DState,
    NonlinearShallowWater2D,
    NonlinearSW2DDiagnostics,
    NonlinearSW2DParams,
    NonlinearSW2DPhysConsts,
    NonlinearSW2DState,
)


def _gaussian_2d(X, Y, mu_x, mu_y, sigma):
    return jnp.exp(-0.5 * (((X - mu_x) / sigma) ** 2 + ((Y - mu_y) / sigma) ** 2))


# ---------------------------------------------------------------------------
# LinearShallowWater2D
# ---------------------------------------------------------------------------


class TestLinearShallowWater2D:
    def test_is_somax_model(self):
        model = LinearShallowWater2D.create(nx=16, ny=16)
        assert isinstance(model, SomaxModel)

    def test_state_types(self):
        assert issubclass(LinearSW2DState, State)
        assert issubclass(LinearSW2DParams, Params)
        assert issubclass(LinearSW2DPhysConsts, PhysConsts)
        assert issubclass(LinearSW2DDiagnostics, Diagnostics)

    def test_vector_field_finite(self):
        model = LinearShallowWater2D.create(nx=16, ny=16, f0=1e-4)
        h0 = jnp.zeros((model.grid.Ny, model.grid.Nx))
        u0 = jnp.zeros_like(h0)
        v0 = jnp.zeros_like(h0)
        state0 = LinearSW2DState(h=h0 + 0.1, u=u0, v=v0)
        state0 = model.apply_boundary_conditions(state0)
        tendency = model.vector_field(0.0, state0)
        assert jnp.all(jnp.isfinite(tendency.h))
        assert jnp.all(jnp.isfinite(tendency.u))
        assert jnp.all(jnp.isfinite(tendency.v))

    def test_integrate(self):
        model = LinearShallowWater2D.create(nx=16, ny=16)
        h0 = jnp.zeros((model.grid.Ny, model.grid.Nx))
        state0 = LinearSW2DState(h=h0, u=jnp.zeros_like(h0), v=jnp.zeros_like(h0))
        sol = model.integrate(state0, t0=0.0, t1=100.0, dt=1.0)
        assert isinstance(sol, dfx.Solution)
        assert jnp.all(jnp.isfinite(sol.ys.h))

    def test_geostrophic_balance_small_tendency(self):
        """A geostrophically balanced state should have near-zero tendencies.

        For f-plane with h = h0 * exp(-(x-x0)^2 / (2*sigma^2)):
        Geostrophic balance: f*v = g * dh/dx, f*u = -g * dh/dy
        """
        f0, g, H0 = 1e-4, 9.81, 100.0
        nx, ny = 32, 32
        Lx = Ly = 1e6
        model = LinearShallowWater2D.create(
            nx=nx, ny=ny, Lx=Lx, Ly=Ly, g=g, f0=f0, H0=H0, bc="periodic"
        )
        x = jnp.arange(model.grid.Nx) * model.grid.dx
        y = jnp.arange(model.grid.Ny) * model.grid.dy
        X, Y = jnp.meshgrid(x, y)

        # Gaussian height perturbation
        sigma = Lx / 10.0
        h0 = 0.01 * _gaussian_2d(X, Y, Lx / 2, Ly / 2, sigma)
        # Geostrophic velocity: u = -(g/f0) * dh/dy, v = (g/f0) * dh/dx
        dh_dx = -(X - Lx / 2) / sigma**2 * h0
        dh_dy = -(Y - Ly / 2) / sigma**2 * h0
        u0 = -(g / f0) * dh_dy
        v0 = (g / f0) * dh_dx

        state0 = LinearSW2DState(h=h0, u=u0, v=v0)
        state0 = model.apply_boundary_conditions(state0)
        tendency = model.vector_field(0.0, state0)

        # Tendencies should be small relative to the fields
        s = (slice(2, -2), slice(2, -2))
        h_scale = float(jnp.max(jnp.abs(h0[s])))
        assert float(jnp.max(jnp.abs(tendency.h[s]))) < 0.5 * h_scale

    def test_diagnose(self):
        model = LinearShallowWater2D.create(nx=16, ny=16)
        h0 = 0.1 * jnp.ones((model.grid.Ny, model.grid.Nx))
        state = LinearSW2DState(h=h0, u=jnp.zeros_like(h0), v=jnp.zeros_like(h0))
        diag = model.diagnose(state)
        assert isinstance(diag, LinearSW2DDiagnostics)
        assert float(diag.energy) > 0.0
        assert diag.relative_vorticity.shape == h0.shape

    def test_grad_through_params(self):
        model = LinearShallowWater2D.create(nx=16, ny=16, lateral_viscosity=1.0)
        h0 = 0.1 * jnp.ones((model.grid.Ny, model.grid.Nx))
        state0 = LinearSW2DState(h=h0, u=jnp.zeros_like(h0), v=jnp.zeros_like(h0))

        @eqx.filter_grad
        def grad_fn(m):
            sol = m.integrate(state0, t0=0.0, t1=50.0, dt=1.0)
            return jnp.sum(sol.ys.h**2)

        grads = grad_fn(model)
        assert jnp.isfinite(grads.params.lateral_viscosity)


# ---------------------------------------------------------------------------
# NonlinearShallowWater2D
# ---------------------------------------------------------------------------


class TestNonlinearShallowWater2D:
    def test_is_somax_model(self):
        model = NonlinearShallowWater2D.create(nx=16, ny=16)
        assert isinstance(model, SomaxModel)

    def test_state_types(self):
        assert issubclass(NonlinearSW2DState, State)
        assert issubclass(NonlinearSW2DParams, Params)
        assert issubclass(NonlinearSW2DPhysConsts, PhysConsts)
        assert issubclass(NonlinearSW2DDiagnostics, Diagnostics)

    def test_integrate_finite(self):
        model = NonlinearShallowWater2D.create(
            nx=16, ny=16, H0=100.0, lateral_viscosity=100.0
        )
        h0 = jnp.full((model.grid.Ny, model.grid.Nx), 100.0)
        x = jnp.arange(model.grid.Nx) * model.grid.dx
        y = jnp.arange(model.grid.Ny) * model.grid.dy
        X, Y = jnp.meshgrid(x, y)
        h0 = h0 + 0.1 * _gaussian_2d(
            X, Y, model.grid.Lx / 2, model.grid.Ly / 2, model.grid.Lx / 5
        )
        state0 = NonlinearSW2DState(h=h0, u=jnp.zeros_like(h0), v=jnp.zeros_like(h0))
        sol = model.integrate(state0, t0=0.0, t1=10.0, dt=0.1)
        assert jnp.all(jnp.isfinite(sol.ys.h))

    def test_diagnose(self):
        model = NonlinearShallowWater2D.create(nx=16, ny=16, H0=100.0)
        h0 = jnp.full((model.grid.Ny, model.grid.Nx), 100.0)
        state = NonlinearSW2DState(h=h0, u=jnp.zeros_like(h0), v=jnp.zeros_like(h0))
        diag = model.diagnose(state)
        assert isinstance(diag, NonlinearSW2DDiagnostics)
        assert diag.potential_vorticity.shape == h0.shape
        assert diag.relative_vorticity.shape == h0.shape

    def test_grad_through_params(self):
        model = NonlinearShallowWater2D.create(
            nx=16, ny=16, H0=100.0, lateral_viscosity=100.0
        )
        h0 = jnp.full((model.grid.Ny, model.grid.Nx), 100.0)
        state0 = NonlinearSW2DState(h=h0, u=jnp.zeros_like(h0), v=jnp.zeros_like(h0))

        @eqx.filter_grad
        def grad_fn(m):
            sol = m.integrate(state0, t0=0.0, t1=5.0, dt=0.1)
            return jnp.sum(sol.ys.h**2)

        grads = grad_fn(model)
        assert jnp.isfinite(grads.params.lateral_viscosity)
