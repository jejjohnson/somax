"""Tests for 1D PDE models."""

from __future__ import annotations

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp

from somax import Diagnostics, Params, SomaxModel, State
from somax.models import (
    Burgers1D,
    Burgers1DDiagnostics,
    Burgers1DParams,
    Burgers1DState,
    Diffusion1D,
    Diffusion1DDiagnostics,
    Diffusion1DParams,
    Diffusion1DState,
    LinearConvection1D,
    LinearConvection1DDiagnostics,
    LinearConvection1DParams,
    LinearConvection1DState,
    NonlinearConvection1D,
    NonlinearConvection1DDiagnostics,
    NonlinearConvection1DState,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gaussian_1d(x, mu, sigma):
    """Normalized Gaussian."""
    return jnp.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _make_x(grid):
    """Cell-centre coordinates including ghost cells."""
    return jnp.arange(grid.Nx) * grid.dx


# ---------------------------------------------------------------------------
# LinearConvection1D
# ---------------------------------------------------------------------------


class TestLinearConvection1D:
    def test_is_somax_model(self):
        model = LinearConvection1D.create()
        assert isinstance(model, SomaxModel)

    def test_state_types(self):
        assert issubclass(LinearConvection1DState, State)
        assert issubclass(LinearConvection1DParams, Params)
        assert issubclass(LinearConvection1DDiagnostics, Diagnostics)

    def test_vector_field_finite(self):
        model = LinearConvection1D.create(nx=50)
        x = _make_x(model.grid)
        u0 = _gaussian_1d(x, mu=1.0, sigma=0.2)
        state0 = LinearConvection1DState(u=u0)
        state0 = model.apply_boundary_conditions(state0)
        tendency = model.vector_field(0.0, state0)
        assert jnp.all(jnp.isfinite(tendency.u))

    def test_integrate(self):
        model = LinearConvection1D.create(nx=50)
        x = _make_x(model.grid)
        state0 = LinearConvection1DState(u=_gaussian_1d(x, 1.0, 0.2))
        sol = model.integrate(state0, t0=0.0, t1=0.1, dt=0.001)
        assert isinstance(sol, dfx.Solution)
        assert jnp.all(jnp.isfinite(sol.ys.u))

    def test_gaussian_translates(self):
        """Gaussian pulse should translate at speed c over short time."""
        c, nx, Lx = 1.0, 200, 4.0
        model = LinearConvection1D.create(nx=nx, Lx=Lx, c=c)
        x = _make_x(model.grid)
        mu0 = 2.0
        state0 = LinearConvection1DState(u=_gaussian_1d(x, mu0, 0.3))
        dt, t_final = 0.002, 0.5
        jnp.arange(0.0, t_final, dt)
        sol = model.integrate(
            state0, t0=0.0, t1=t_final, dt=dt, saveat=dfx.SaveAt(t1=True)
        )
        u_final = sol.ys.u[0]
        # Expected: shifted Gaussian
        u_exact = _gaussian_1d(x, mu0 + c * t_final, 0.3)
        # Correlation should be high (> 0.95) even with numerical diffusion
        corr = jnp.sum(u_final[1:-1] * u_exact[1:-1]) / (
            jnp.sqrt(jnp.sum(u_final[1:-1] ** 2) * jnp.sum(u_exact[1:-1] ** 2))
        )
        assert float(corr) > 0.90

    def test_grad_through_params(self):
        model = LinearConvection1D.create(nx=50)
        x = _make_x(model.grid)
        state0 = LinearConvection1DState(u=_gaussian_1d(x, 1.0, 0.2))

        @eqx.filter_grad
        def grad_fn(m):
            sol = m.integrate(state0, t0=0.0, t1=0.05, dt=0.001)
            return jnp.sum(sol.ys.u**2)

        grads = grad_fn(model)
        assert jnp.isfinite(grads.params.c)

    def test_diagnose(self):
        model = LinearConvection1D.create(nx=50)
        x = _make_x(model.grid)
        state = LinearConvection1DState(u=_gaussian_1d(x, 1.0, 0.2))
        diag = model.diagnose(state)
        assert isinstance(diag, LinearConvection1DDiagnostics)
        assert float(diag.energy) > 0.0


# ---------------------------------------------------------------------------
# NonlinearConvection1D
# ---------------------------------------------------------------------------


class TestNonlinearConvection1D:
    def test_is_somax_model(self):
        model = NonlinearConvection1D.create()
        assert isinstance(model, SomaxModel)

    def test_state_types(self):
        assert issubclass(NonlinearConvection1DState, State)
        assert issubclass(NonlinearConvection1DDiagnostics, Diagnostics)

    def test_integrate_finite(self):
        model = NonlinearConvection1D.create(nx=100)
        x = _make_x(model.grid)
        state0 = NonlinearConvection1DState(u=_gaussian_1d(x, 1.0, 0.2))
        sol = model.integrate(state0, t0=0.0, t1=0.05, dt=0.001)
        assert jnp.all(jnp.isfinite(sol.ys.u))


# ---------------------------------------------------------------------------
# Diffusion1D
# ---------------------------------------------------------------------------


class TestDiffusion1D:
    def test_is_somax_model(self):
        model = Diffusion1D.create()
        assert isinstance(model, SomaxModel)

    def test_state_types(self):
        assert issubclass(Diffusion1DState, State)
        assert issubclass(Diffusion1DParams, Params)
        assert issubclass(Diffusion1DDiagnostics, Diagnostics)

    def test_gaussian_spreads(self):
        """Gaussian should spread: variance increases over time."""
        nx, Lx, nu = 200, 4.0, 0.05
        model = Diffusion1D.create(nx=nx, Lx=Lx, nu=nu)
        x = _make_x(model.grid)
        state0 = Diffusion1DState(u=_gaussian_1d(x, 2.0, 0.3))
        sol = model.integrate(
            state0, t0=0.0, t1=0.5, dt=0.001, saveat=dfx.SaveAt(t1=True)
        )
        u_final = sol.ys.u[0]
        # Peak should decrease (diffusion broadens the profile)
        assert float(jnp.max(u_final)) < float(jnp.max(state0.u))

    def test_energy_decreases(self):
        """Diffusion dissipates energy with periodic BCs."""
        model = Diffusion1D.create(nx=100, nu=0.1)
        x = _make_x(model.grid)
        state0 = Diffusion1DState(u=_gaussian_1d(x, 1.0, 0.2))
        e0 = model.diagnose(state0).energy
        sol = model.integrate(
            state0, t0=0.0, t1=0.2, dt=0.001, saveat=dfx.SaveAt(t1=True)
        )
        state_f = Diffusion1DState(u=sol.ys.u[0])
        ef = model.diagnose(state_f).energy
        assert float(ef) < float(e0)

    def test_grad_through_params(self):
        model = Diffusion1D.create(nx=50, nu=0.05)
        x = _make_x(model.grid)
        state0 = Diffusion1DState(u=_gaussian_1d(x, 1.0, 0.2))

        @eqx.filter_grad
        def grad_fn(m):
            sol = m.integrate(state0, t0=0.0, t1=0.05, dt=0.001)
            return jnp.sum(sol.ys.u**2)

        grads = grad_fn(model)
        assert jnp.isfinite(grads.params.nu)


# ---------------------------------------------------------------------------
# Burgers1D
# ---------------------------------------------------------------------------


class TestBurgers1D:
    def test_is_somax_model(self):
        model = Burgers1D.create()
        assert isinstance(model, SomaxModel)

    def test_state_types(self):
        assert issubclass(Burgers1DState, State)
        assert issubclass(Burgers1DParams, Params)
        assert issubclass(Burgers1DDiagnostics, Diagnostics)

    def test_integrate_finite(self):
        model = Burgers1D.create(nx=100, nu=0.05)
        x = _make_x(model.grid)
        state0 = Burgers1DState(u=_gaussian_1d(x, 1.0, 0.2))
        sol = model.integrate(state0, t0=0.0, t1=0.1, dt=0.001)
        assert jnp.all(jnp.isfinite(sol.ys.u))

    def test_grad_through_params(self):
        model = Burgers1D.create(nx=50, nu=0.05)
        x = _make_x(model.grid)
        state0 = Burgers1DState(u=_gaussian_1d(x, 1.0, 0.2))

        @eqx.filter_grad
        def grad_fn(m):
            sol = m.integrate(state0, t0=0.0, t1=0.05, dt=0.001)
            return jnp.sum(sol.ys.u**2)

        grads = grad_fn(model)
        assert jnp.isfinite(grads.params.nu)

    def test_viscosity_smooths(self):
        """Higher viscosity should produce a smoother solution."""
        nx, Lx = 100, 2.0
        x_low = _make_x(Burgers1D.create(nx=nx, Lx=Lx, nu=0.01).grid)
        model_low = Burgers1D.create(nx=nx, Lx=Lx, nu=0.01)
        model_high = Burgers1D.create(nx=nx, Lx=Lx, nu=0.2)
        u0 = _gaussian_1d(x_low, 1.0, 0.2)
        state0 = Burgers1DState(u=u0)
        sol_low = model_low.integrate(
            state0, t0=0.0, t1=0.1, dt=0.001, saveat=dfx.SaveAt(t1=True)
        )
        sol_high = model_high.integrate(
            state0, t0=0.0, t1=0.1, dt=0.001, saveat=dfx.SaveAt(t1=True)
        )
        # Higher viscosity should have smaller gradient magnitude
        grad_low = jnp.sum(jnp.diff(sol_low.ys.u[0, 1:-1]) ** 2)
        grad_high = jnp.sum(jnp.diff(sol_high.ys.u[0, 1:-1]) ** 2)
        assert float(grad_high) < float(grad_low)
