"""Tests for 2D incompressible Navier-Stokes."""

from __future__ import annotations

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp

from somax import Diagnostics, Params, SomaxModel, State
from somax.models import (
    IncompressibleNS2D,
    NSDiagnostics,
    NSParams,
    NSVorticityState,
)


class TestIncompressibleNS2D:
    def test_is_somax_model(self):
        model = IncompressibleNS2D.create(nx=16, ny=16, nu=0.1)
        assert isinstance(model, SomaxModel)

    def test_state_types(self):
        assert issubclass(NSVorticityState, State)
        assert issubclass(NSParams, Params)
        assert issubclass(NSDiagnostics, Diagnostics)

    def test_vector_field_finite(self):
        model = IncompressibleNS2D.create(nx=16, ny=16, nu=0.1)
        omega0 = jnp.zeros((model.grid.Ny, model.grid.Nx))
        state0 = NSVorticityState(omega=omega0)
        state0 = model.apply_boundary_conditions(state0)
        tendency = model.vector_field(0.0, state0)
        assert jnp.all(jnp.isfinite(tendency.omega))

    def test_integrate_cavity(self):
        """Lid-driven cavity should integrate without NaN."""
        model = IncompressibleNS2D.create(nx=16, ny=16, nu=0.1, problem="cavity")
        omega0 = jnp.zeros((model.grid.Ny, model.grid.Nx))
        state0 = NSVorticityState(omega=omega0)
        sol = model.integrate(
            state0,
            t0=0.0,
            t1=0.05,
            dt=0.001,
            saveat=dfx.SaveAt(t1=True),
        )
        assert jnp.all(jnp.isfinite(sol.ys.omega))

    def test_integrate_channel(self):
        """Channel flow should integrate without NaN and use periodic-x BCs."""
        model = IncompressibleNS2D.create(
            nx=16,
            ny=16,
            nu=0.1,
            problem="channel",
            body_force=1.0,
            u_lid=0.0,
        )
        # Verify channel configuration
        assert model.problem == "channel"
        omega0 = jnp.zeros((model.grid.Ny, model.grid.Nx))
        state0 = NSVorticityState(omega=omega0)
        sol = model.integrate(
            state0,
            t0=0.0,
            t1=0.05,
            dt=0.001,
            saveat=dfx.SaveAt(t1=True),
        )
        assert jnp.all(jnp.isfinite(sol.ys.omega))
        # Channel BCs enforce periodicity in x: left/right ghost cells should match
        omega_final = sol.ys.omega[0]
        state_bc = model.apply_boundary_conditions(NSVorticityState(omega=omega_final))
        interior_rows = slice(1, -1)
        assert jnp.allclose(
            state_bc.omega[interior_rows, 0], state_bc.omega[interior_rows, -2]
        )

    def test_diagnose(self):
        model = IncompressibleNS2D.create(nx=16, ny=16, nu=0.1)
        omega0 = jnp.zeros((model.grid.Ny, model.grid.Nx))
        state0 = NSVorticityState(omega=omega0)
        diag = model.diagnose(state0)
        assert isinstance(diag, NSDiagnostics)
        assert diag.psi.shape == omega0.shape
        assert diag.u.shape == omega0.shape
        assert jnp.isfinite(diag.kinetic_energy)

    def test_grad_through_nu(self):
        """jax.grad should flow through viscosity."""
        model = IncompressibleNS2D.create(nx=16, ny=16, nu=0.1)
        omega0 = 0.1 * jnp.ones((model.grid.Ny, model.grid.Nx))
        state0 = NSVorticityState(omega=omega0)

        @eqx.filter_grad
        def grad_fn(m):
            sol = m.integrate(state0, t0=0.0, t1=0.02, dt=0.001)
            return jnp.sum(sol.ys.omega**2)

        grads = grad_fn(model)
        assert jnp.isfinite(grads.params.nu)

    def test_cavity_develops_vorticity(self):
        """Lid-driven cavity should develop non-zero vorticity."""
        model = IncompressibleNS2D.create(
            nx=16, ny=16, nu=0.1, problem="cavity", u_lid=1.0
        )
        omega0 = jnp.zeros((model.grid.Ny, model.grid.Nx))
        state0 = NSVorticityState(omega=omega0)
        sol = model.integrate(
            state0,
            t0=0.0,
            t1=0.1,
            dt=0.001,
            saveat=dfx.SaveAt(t1=True),
        )
        omega_final = sol.ys.omega[0]
        # Vorticity should be non-zero (lid drives flow)
        assert float(jnp.max(jnp.abs(omega_final[1:-1, 1:-1]))) > 0.01
