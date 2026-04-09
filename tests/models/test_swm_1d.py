"""Tests for 1D shallow water models."""

from __future__ import annotations

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp

from somax import Diagnostics, Params, PhysConsts, SomaxModel, State
from somax.models import (
    LinearShallowWater1D,
    LinearSW1DDiagnostics,
    LinearSW1DParams,
    LinearSW1DPhysConsts,
    LinearSW1DState,
    NonlinearShallowWater1D,
    NonlinearSW1DDiagnostics,
    NonlinearSW1DParams,
    NonlinearSW1DPhysConsts,
    NonlinearSW1DState,
)


def _gaussian_1d(x, mu, sigma):
    return jnp.exp(-0.5 * ((x - mu) / sigma) ** 2)


# ---------------------------------------------------------------------------
# LinearShallowWater1D
# ---------------------------------------------------------------------------


class TestLinearShallowWater1D:
    def test_is_somax_model(self):
        model = LinearShallowWater1D.create()
        assert isinstance(model, SomaxModel)

    def test_state_types(self):
        assert issubclass(LinearSW1DState, State)
        assert issubclass(LinearSW1DParams, Params)
        assert issubclass(LinearSW1DPhysConsts, PhysConsts)
        assert issubclass(LinearSW1DDiagnostics, Diagnostics)

    def test_vector_field_finite(self):
        model = LinearShallowWater1D.create(nx=50, f0=0.0)
        x = jnp.arange(model.grid.Nx) * model.grid.dx
        h0 = _gaussian_1d(x, model.grid.Lx / 2.0, model.grid.Lx / 10.0)
        state0 = LinearSW1DState(h=h0, u=jnp.zeros_like(h0), v=jnp.zeros_like(h0))
        state0 = model.apply_boundary_conditions(state0)
        tendency = model.vector_field(0.0, state0)
        assert jnp.all(jnp.isfinite(tendency.h))
        assert jnp.all(jnp.isfinite(tendency.u))

    def test_integrate(self):
        model = LinearShallowWater1D.create(nx=50, f0=0.0)
        x = jnp.arange(model.grid.Nx) * model.grid.dx
        h0 = _gaussian_1d(x, model.grid.Lx / 2.0, model.grid.Lx / 10.0)
        state0 = LinearSW1DState(h=h0, u=jnp.zeros_like(h0), v=jnp.zeros_like(h0))
        sol = model.integrate(state0, t0=0.0, t1=100.0, dt=1.0)
        assert isinstance(sol, dfx.Solution)
        assert jnp.all(jnp.isfinite(sol.ys.h))

    def test_gravity_wave_speed(self):
        """Gravity wave should propagate at c = sqrt(g*H0)."""
        g, H0 = 9.81, 100.0
        nx, Lx = 400, 2e6
        c = jnp.sqrt(g * H0)
        model = LinearShallowWater1D.create(nx=nx, Lx=Lx, g=g, f0=0.0, H0=H0)
        x = jnp.arange(model.grid.Nx) * model.grid.dx
        sigma = Lx / 20.0
        mu0 = Lx / 2.0
        h0 = _gaussian_1d(x, mu0, sigma)
        state0 = LinearSW1DState(h=h0, u=jnp.zeros_like(h0), v=jnp.zeros_like(h0))

        t_final = 5000.0
        dt = 0.5 * model.grid.dx / float(c)  # CFL < 1
        sol = model.integrate(
            state0, t0=0.0, t1=t_final, dt=dt, saveat=dfx.SaveAt(t1=True)
        )
        h_final = sol.ys.h[0]

        # The pulse splits into two waves moving at ±c.
        # Expected right-moving peak at mu0 + c*t_final
        mu_right = mu0 + float(c) * t_final
        expected_right = _gaussian_1d(x, mu_right % Lx, sigma)
        mu_left = mu0 - float(c) * t_final
        expected_left = _gaussian_1d(x, mu_left % Lx, sigma)
        expected = 0.5 * (expected_right + expected_left)

        # Correlation should be high (> 0.85)
        interior = slice(1, -1)
        corr = jnp.sum(h_final[interior] * expected[interior]) / (
            jnp.sqrt(jnp.sum(h_final[interior] ** 2) * jnp.sum(expected[interior] ** 2))
        )
        assert float(corr) > 0.85, f"Correlation {float(corr):.3f} too low"

    def test_inertial_oscillation(self):
        """With f0 and no pressure gradient, velocity should oscillate."""
        f0 = 1e-4
        model = LinearShallowWater1D.create(nx=10, Lx=1e6, f0=f0, H0=1e6)
        # Large H0 suppresses gravity waves; uniform fields
        h0 = jnp.zeros(model.grid.Nx)
        u0 = jnp.ones(model.grid.Nx)
        v0 = jnp.zeros(model.grid.Nx)
        state0 = LinearSW1DState(h=h0, u=u0, v=v0)

        period = 2.0 * jnp.pi / f0
        dt = period / 100.0
        sol = model.integrate(
            state0, t0=0.0, t1=period, dt=dt, saveat=dfx.SaveAt(t1=True)
        )
        # After one full period, u should return close to initial
        u_final = sol.ys.u[0]
        assert jnp.allclose(u_final[1:-1], 1.0, atol=0.15)

    def test_diagnose(self):
        model = LinearShallowWater1D.create(nx=50)
        x = jnp.arange(model.grid.Nx) * model.grid.dx
        h0 = _gaussian_1d(x, model.grid.Lx / 2.0, model.grid.Lx / 10.0)
        state = LinearSW1DState(h=h0, u=jnp.zeros_like(h0), v=jnp.zeros_like(h0))
        diag = model.diagnose(state)
        assert isinstance(diag, LinearSW1DDiagnostics)
        assert float(diag.energy) > 0.0

    def test_grad_through_params(self):
        model = LinearShallowWater1D.create(nx=20, lateral_viscosity=1.0)
        x = jnp.arange(model.grid.Nx) * model.grid.dx
        h0 = _gaussian_1d(x, model.grid.Lx / 2.0, model.grid.Lx / 10.0)
        state0 = LinearSW1DState(h=h0, u=jnp.zeros_like(h0), v=jnp.zeros_like(h0))

        @eqx.filter_grad
        def grad_fn(m):
            sol = m.integrate(state0, t0=0.0, t1=50.0, dt=1.0)
            return jnp.sum(sol.ys.h**2)

        grads = grad_fn(model)
        assert jnp.isfinite(grads.params.lateral_viscosity)


# ---------------------------------------------------------------------------
# NonlinearShallowWater1D
# ---------------------------------------------------------------------------


class TestNonlinearShallowWater1D:
    def test_is_somax_model(self):
        model = NonlinearShallowWater1D.create()
        assert isinstance(model, SomaxModel)

    def test_state_types(self):
        assert issubclass(NonlinearSW1DState, State)
        assert issubclass(NonlinearSW1DParams, Params)
        assert issubclass(NonlinearSW1DPhysConsts, PhysConsts)
        assert issubclass(NonlinearSW1DDiagnostics, Diagnostics)

    def test_integrate_finite(self):
        model = NonlinearShallowWater1D.create(nx=100, H0=100.0, f0=0.0)
        x = jnp.arange(model.grid.Nx) * model.grid.dx
        h0 = 100.0 + _gaussian_1d(x, model.grid.Lx / 2.0, model.grid.Lx / 10.0)
        state0 = NonlinearSW1DState(h=h0, u=jnp.zeros_like(h0), v=jnp.zeros_like(h0))
        sol = model.integrate(state0, t0=0.0, t1=100.0, dt=1.0)
        assert jnp.all(jnp.isfinite(sol.ys.h))

    def test_grad_through_params(self):
        model = NonlinearShallowWater1D.create(nx=20, H0=100.0, lateral_viscosity=1.0)
        x = jnp.arange(model.grid.Nx) * model.grid.dx
        h0 = 100.0 + 0.1 * _gaussian_1d(x, model.grid.Lx / 2.0, model.grid.Lx / 10.0)
        state0 = NonlinearSW1DState(h=h0, u=jnp.zeros_like(h0), v=jnp.zeros_like(h0))

        @eqx.filter_grad
        def grad_fn(m):
            sol = m.integrate(state0, t0=0.0, t1=50.0, dt=1.0)
            return jnp.sum(sol.ys.h**2)

        grads = grad_fn(model)
        assert jnp.isfinite(grads.params.lateral_viscosity)

    def test_diagnose(self):
        model = NonlinearShallowWater1D.create(nx=50, H0=100.0)
        x = jnp.arange(model.grid.Nx) * model.grid.dx
        h0 = 100.0 + _gaussian_1d(x, model.grid.Lx / 2.0, model.grid.Lx / 10.0)
        state = NonlinearSW1DState(h=h0, u=jnp.zeros_like(h0), v=jnp.zeros_like(h0))
        diag = model.diagnose(state)
        assert isinstance(diag, NonlinearSW1DDiagnostics)
        assert float(diag.mass) > 0.0
