"""Tests for barotropic quasi-geostrophic model."""

from __future__ import annotations

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp

from somax import Diagnostics, Params, PhysConsts, SomaxModel, State
from somax.models import (
    BarotropicQG,
    BarotropicQGDiagnostics,
    BarotropicQGParams,
    BarotropicQGPhysConsts,
    BarotropicQGState,
)


class TestBarotropicQG:
    def test_is_somax_model(self):
        model = BarotropicQG.create(nx=16, ny=16)
        assert isinstance(model, SomaxModel)

    def test_state_types(self):
        assert issubclass(BarotropicQGState, State)
        assert issubclass(BarotropicQGParams, Params)
        assert issubclass(BarotropicQGPhysConsts, PhysConsts)
        assert issubclass(BarotropicQGDiagnostics, Diagnostics)

    def test_vector_field_finite(self):
        model = BarotropicQG.create(nx=16, ny=16, lateral_viscosity=100.0)
        q0 = 0.01 * jnp.ones((model.grid.Ny, model.grid.Nx))
        state0 = BarotropicQGState(q=q0)
        state0 = model.apply_boundary_conditions(state0)
        tendency = model.vector_field(0.0, state0)
        assert jnp.all(jnp.isfinite(tendency.q))

    def test_integrate(self):
        model = BarotropicQG.create(
            nx=16, ny=16, lateral_viscosity=100.0, wind_amplitude=1e-5
        )
        q0 = jnp.zeros((model.grid.Ny, model.grid.Nx))
        state0 = BarotropicQGState(q=q0)
        sol = model.integrate(state0, t0=0.0, t1=100.0, dt=1.0)
        assert isinstance(sol, dfx.Solution)
        assert jnp.all(jnp.isfinite(sol.ys.q))

    def test_wind_forcing_develops_pv(self):
        """Wind forcing should drive non-zero PV from rest."""
        model = BarotropicQG.create(
            nx=16, ny=16, lateral_viscosity=100.0, wind_amplitude=1e-5
        )
        q0 = jnp.zeros((model.grid.Ny, model.grid.Nx))
        state0 = BarotropicQGState(q=q0)
        sol = model.integrate(
            state0, t0=0.0, t1=1000.0, dt=1.0, saveat=dfx.SaveAt(t1=True)
        )
        q_final = sol.ys.q[0]
        # PV should be non-zero in the interior
        assert float(jnp.max(jnp.abs(q_final[2:-2, 2:-2]))) > 1e-10

    def test_diagnose(self):
        model = BarotropicQG.create(nx=16, ny=16)
        q0 = 0.01 * jnp.ones((model.grid.Ny, model.grid.Nx))
        state = BarotropicQGState(q=q0)
        diag = model.diagnose(state)
        assert isinstance(diag, BarotropicQGDiagnostics)
        assert diag.psi.shape == q0.shape
        assert diag.u.shape == q0.shape
        assert jnp.isfinite(diag.kinetic_energy)
        assert jnp.isfinite(diag.enstrophy)

    def test_grad_through_params(self):
        model = BarotropicQG.create(nx=16, ny=16, lateral_viscosity=100.0)
        q0 = 0.01 * jnp.ones((model.grid.Ny, model.grid.Nx))
        state0 = BarotropicQGState(q=q0)

        @eqx.filter_grad
        def grad_fn(m):
            sol = m.integrate(state0, t0=0.0, t1=50.0, dt=1.0)
            return jnp.sum(sol.ys.q**2)

        grads = grad_fn(model)
        assert jnp.isfinite(grads.params.lateral_viscosity)

    def test_zero_pv_zero_tendency(self):
        """Zero PV with no forcing should give zero tendency."""
        model = BarotropicQG.create(
            nx=16, ny=16, lateral_viscosity=0.0, wind_amplitude=0.0
        )
        q0 = jnp.zeros((model.grid.Ny, model.grid.Nx))
        state0 = BarotropicQGState(q=q0)
        state0 = model.apply_boundary_conditions(state0)
        tendency = model.vector_field(0.0, state0)
        assert float(jnp.max(jnp.abs(tendency.q))) < 1e-12
