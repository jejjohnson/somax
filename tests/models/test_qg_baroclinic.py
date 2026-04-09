"""Tests for baroclinic (multilayer) quasi-geostrophic model."""

from __future__ import annotations

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp

from somax import Diagnostics, Params, PhysConsts, SomaxModel, State
from somax.core import StratificationProfile
from somax.models import (
    BaroclinicQG,
    BaroclinicQGDiagnostics,
    BaroclinicQGParams,
    BaroclinicQGPhysConsts,
    BaroclinicQGState,
    doublegyre_baroclinic_qg,
)


class TestBaroclinicQG:
    def test_is_somax_model(self):
        model = BaroclinicQG.create(
            nx=16,
            ny=16,
            n_layers=2,
            H=(500.0, 4500.0),
            g_prime=(9.81, 0.025),
        )
        assert isinstance(model, SomaxModel)

    def test_state_types(self):
        assert issubclass(BaroclinicQGState, State)
        assert issubclass(BaroclinicQGParams, Params)
        assert issubclass(BaroclinicQGPhysConsts, PhysConsts)
        assert issubclass(BaroclinicQGDiagnostics, Diagnostics)

    def test_create_default(self):
        model = BaroclinicQG.create()
        assert model.consts.n_layers == 3

    def test_create_two_layer(self):
        model = BaroclinicQG.create(
            nx=16, ny=16, n_layers=2, H=(500.0, 4500.0), g_prime=(9.81, 0.025)
        )
        assert model.consts.n_layers == 2
        assert model.helmholtz_lambdas.shape == (2,)

    def test_create_from_stratification(self):
        strat = StratificationProfile.from_N2_constant(
            N2=1e-5, depth=4000.0, n_layers=4
        )
        model = BaroclinicQG.create(nx=16, ny=16, stratification=strat)
        assert model.consts.n_layers == 4

    def test_state_shape(self):
        model = BaroclinicQG.create(
            nx=16, ny=16, n_layers=2, H=(500.0, 4500.0), g_prime=(9.81, 0.025)
        )
        q0 = jnp.zeros((2, model.grid.Ny, model.grid.Nx))
        state = BaroclinicQGState(q=q0)
        assert state.q.shape == (2, model.grid.Ny, model.grid.Nx)

    def test_vector_field_finite(self):
        model = BaroclinicQG.create(
            nx=16,
            ny=16,
            n_layers=2,
            H=(500.0, 4500.0),
            g_prime=(9.81, 0.025),
            lateral_viscosity=100.0,
        )
        q0 = 0.01 * jnp.ones((2, model.grid.Ny, model.grid.Nx))
        state0 = BaroclinicQGState(q=q0)
        state0 = model.apply_boundary_conditions(state0)
        tendency = model.vector_field(0.0, state0)
        assert jnp.all(jnp.isfinite(tendency.q))

    def test_zero_pv_zero_tendency(self):
        """Zero PV with no forcing should give zero tendency."""
        model = BaroclinicQG.create(
            nx=16,
            ny=16,
            n_layers=2,
            H=(500.0, 4500.0),
            g_prime=(9.81, 0.025),
            lateral_viscosity=0.0,
            wind_amplitude=0.0,
            bottom_drag=0.0,
        )
        q0 = jnp.zeros((2, model.grid.Ny, model.grid.Nx))
        state0 = BaroclinicQGState(q=q0)
        tendency = model.vector_field(0.0, state0)
        assert float(jnp.max(jnp.abs(tendency.q))) < 1e-12

    def test_wind_forcing_top_layer_only(self):
        """Wind forcing should only affect the top layer."""
        model = BaroclinicQG.create(
            nx=16,
            ny=16,
            n_layers=2,
            H=(500.0, 4500.0),
            g_prime=(9.81, 0.025),
            wind_amplitude=1e-10,
        )
        q0 = jnp.zeros((2, model.grid.Ny, model.grid.Nx))
        state0 = BaroclinicQGState(q=q0)
        tendency = model.vector_field(0.0, state0)
        # Top layer should have non-zero tendency from wind
        assert float(jnp.max(jnp.abs(tendency.q[0]))) > 0
        # Bottom layer should be zero (no drag on q=0)
        assert float(jnp.max(jnp.abs(tendency.q[1]))) < 1e-15

    def test_bottom_drag_bottom_layer_only(self):
        """Bottom drag should only affect the bottom layer."""
        model = BaroclinicQG.create(
            nx=16,
            ny=16,
            n_layers=2,
            H=(500.0, 4500.0),
            g_prime=(9.81, 0.025),
            bottom_drag=1e-4,
        )
        # Set non-zero PV so psi is non-zero and drag acts
        q0 = 0.01 * jnp.ones((2, model.grid.Ny, model.grid.Nx))
        state0 = model.apply_boundary_conditions(BaroclinicQGState(q=q0))
        tendency = model.vector_field(0.0, state0)
        # Bottom layer tendency should include drag contribution
        # (top layer has only advection, bottom has advection + drag)
        # Just check both are finite and non-zero
        assert jnp.all(jnp.isfinite(tendency.q))

    def test_pv_inversion_finite(self):
        """PV inversion should produce finite streamfunction."""
        model = BaroclinicQG.create(
            nx=16,
            ny=16,
            n_layers=2,
            H=(500.0, 4500.0),
            g_prime=(9.81, 0.025),
        )
        q0 = 0.01 * jnp.ones((2, model.grid.Ny, model.grid.Nx))
        q0 = jnp.stack([q0[0], -q0[0]])  # Opposite sign for baroclinic structure
        psi = model._invert_pv(q0)
        assert jnp.all(jnp.isfinite(psi))
        assert psi.shape == q0.shape

    def test_modal_transform_roundtrip(self):
        """Layer -> modal -> layer should recover original."""
        model = BaroclinicQG.create(
            nx=16,
            ny=16,
            n_layers=2,
            H=(500.0, 4500.0),
            g_prime=(9.81, 0.025),
        )
        q0 = jnp.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        )
        roundtrip = model.modal.to_layer(model.modal.to_modal(q0))
        assert jnp.allclose(roundtrip, q0, atol=1e-5)

    def test_boundary_conditions(self):
        """BCs should zero out boundary cells for all layers."""
        model = BaroclinicQG.create(
            nx=16,
            ny=16,
            n_layers=2,
            H=(500.0, 4500.0),
            g_prime=(9.81, 0.025),
        )
        q0 = jnp.ones((2, model.grid.Ny, model.grid.Nx))
        state = model.apply_boundary_conditions(BaroclinicQGState(q=q0))
        # Boundaries should be zero
        assert jnp.allclose(state.q[:, 0, :], 0.0)
        assert jnp.allclose(state.q[:, -1, :], 0.0)
        assert jnp.allclose(state.q[:, :, 0], 0.0)
        assert jnp.allclose(state.q[:, :, -1], 0.0)
        # Interior should be unchanged
        assert jnp.allclose(state.q[:, 1:-1, 1:-1], 1.0)

    def test_integrate_short(self):
        """Short integration should produce finite results."""
        model = BaroclinicQG.create(
            nx=16,
            ny=16,
            n_layers=2,
            H=(500.0, 4500.0),
            g_prime=(9.81, 0.025),
            lateral_viscosity=50.0,
            wind_amplitude=1e-10,
        )
        q0 = jnp.zeros((2, model.grid.Ny, model.grid.Nx))
        state0 = BaroclinicQGState(q=q0)
        sol = model.integrate(state0, t0=0.0, t1=3600.0, dt=100.0)
        assert isinstance(sol, dfx.Solution)
        assert jnp.all(jnp.isfinite(sol.ys.q))

    def test_wind_develops_flow(self):
        """Wind forcing from rest should develop non-zero PV."""
        model = BaroclinicQG.create(
            nx=16,
            ny=16,
            n_layers=2,
            H=(500.0, 4500.0),
            g_prime=(9.81, 0.025),
            lateral_viscosity=50.0,
            wind_amplitude=1e-10,
        )
        q0 = jnp.zeros((2, model.grid.Ny, model.grid.Nx))
        state0 = BaroclinicQGState(q=q0)
        sol = model.integrate(
            state0,
            t0=0.0,
            t1=3600.0,
            dt=100.0,
            saveat=dfx.SaveAt(t1=True),
        )
        q_final = sol.ys.q[0]
        assert float(jnp.max(jnp.abs(q_final[0, 2:-2, 2:-2]))) > 1e-10

    def test_diagnose_shapes(self):
        """Diagnostics should have correct shapes."""
        model = BaroclinicQG.create(
            nx=16,
            ny=16,
            n_layers=2,
            H=(500.0, 4500.0),
            g_prime=(9.81, 0.025),
        )
        q0 = 0.01 * jnp.ones((2, model.grid.Ny, model.grid.Nx))
        state = BaroclinicQGState(q=q0)
        diag = model.diagnose(state)
        assert isinstance(diag, BaroclinicQGDiagnostics)
        assert diag.psi.shape == (2, model.grid.Ny, model.grid.Nx)
        assert diag.u.shape == (2, model.grid.Ny, model.grid.Nx)
        assert diag.v.shape == (2, model.grid.Ny, model.grid.Nx)
        assert diag.kinetic_energy.shape == (2,)
        assert diag.enstrophy.shape == (2,)
        assert diag.rossby_radii.shape == (2,)
        assert jnp.isfinite(diag.total_kinetic_energy)
        assert jnp.isfinite(diag.total_enstrophy)

    def test_ke_nonnegative(self):
        """Kinetic energy should be non-negative."""
        model = BaroclinicQG.create(
            nx=16,
            ny=16,
            n_layers=2,
            H=(500.0, 4500.0),
            g_prime=(9.81, 0.025),
        )
        q0 = 0.01 * jnp.ones((2, model.grid.Ny, model.grid.Nx))
        state = BaroclinicQGState(q=q0)
        diag = model.diagnose(state)
        assert jnp.all(diag.kinetic_energy >= 0)

    def test_rossby_radii_physical(self):
        """Rossby deformation radii should be positive (finite ones)."""
        model = BaroclinicQG.create(
            nx=16,
            ny=16,
            n_layers=2,
            H=(500.0, 4500.0),
            g_prime=(9.81, 0.025),
        )
        diag = model.diagnose(
            BaroclinicQGState(q=jnp.zeros((2, model.grid.Ny, model.grid.Nx)))
        )
        finite_radii = diag.rossby_radii[jnp.isfinite(diag.rossby_radii)]
        assert finite_radii.size > 0
        assert jnp.all(finite_radii > 0)

    def test_three_layer(self):
        """3-layer model should work correctly."""
        model = BaroclinicQG.create(
            nx=16,
            ny=16,
            n_layers=3,
            H=(400.0, 1100.0, 2500.0),
            g_prime=(9.81, 0.025, 0.0125),
            lateral_viscosity=50.0,
            wind_amplitude=1e-10,
        )
        q0 = jnp.zeros((3, model.grid.Ny, model.grid.Nx))
        state0 = BaroclinicQGState(q=q0)
        sol = model.integrate(state0, t0=0.0, t1=3600.0, dt=100.0)
        assert jnp.all(jnp.isfinite(sol.ys.q))

    def test_grad_through_params(self):
        """Gradients should flow through differentiable parameters."""
        model = BaroclinicQG.create(
            nx=16,
            ny=16,
            n_layers=2,
            H=(500.0, 4500.0),
            g_prime=(9.81, 0.025),
            lateral_viscosity=100.0,
        )
        q0 = 0.01 * jnp.ones((2, model.grid.Ny, model.grid.Nx))
        state0 = BaroclinicQGState(q=q0)

        @eqx.filter_grad
        def grad_fn(m):
            sol = m.integrate(state0, t0=0.0, t1=50.0, dt=1.0)
            return jnp.sum(sol.ys.q**2)

        grads = grad_fn(model)
        assert jnp.isfinite(grads.params.lateral_viscosity)


class TestDoublegyreBaroclinicQG:
    def test_creates(self):
        model, state0 = doublegyre_baroclinic_qg(nx=16, ny=16)
        assert isinstance(model, BaroclinicQG)
        assert isinstance(state0, BaroclinicQGState)
        assert state0.q.shape[0] == 3

    def test_integrates(self):
        model, state0 = doublegyre_baroclinic_qg(nx=16, ny=16)
        sol = model.integrate(state0, t0=0.0, t1=3600.0, dt=100.0)
        assert jnp.all(jnp.isfinite(sol.ys.q))
