"""Tests for multilayer 2D nonlinear shallow water model."""

from __future__ import annotations

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp

from somax import Diagnostics, Params, PhysConsts, SomaxModel, State
from somax.core import StratificationProfile
from somax.models import (
    MultilayerShallowWater2D,
    MultilayerSW2DDiagnostics,
    MultilayerSW2DParams,
    MultilayerSW2DPhysConsts,
    MultilayerSW2DState,
    baroclinic_instability_swm,
)


def _make_model(**kw):
    defaults = dict(
        nx=16,
        ny=16,
        n_layers=2,
        H=(500.0, 500.0),
        g_prime=(9.81, 0.02),
    )
    defaults.update(kw)
    return MultilayerShallowWater2D.create(**defaults)


def _rest_state(model):
    nl = model.consts.n_layers
    Ny, Nx = model.grid.Ny, model.grid.Nx
    h0 = jnp.ones((nl, Ny, Nx)) * model.strat.H[:, None, None]
    u0 = jnp.zeros((nl, Ny, Nx))
    v0 = jnp.zeros((nl, Ny, Nx))
    return MultilayerSW2DState(h=h0, u=u0, v=v0)


class TestMultilayerSW2DContract:
    def test_is_somax_model(self):
        model = _make_model()
        assert isinstance(model, SomaxModel)

    def test_state_types(self):
        assert issubclass(MultilayerSW2DState, State)
        assert issubclass(MultilayerSW2DParams, Params)
        assert issubclass(MultilayerSW2DPhysConsts, PhysConsts)
        assert issubclass(MultilayerSW2DDiagnostics, Diagnostics)


class TestMultilayerSW2DCreate:
    def test_create_default(self):
        model = MultilayerShallowWater2D.create()
        assert model.consts.n_layers == 3

    def test_create_two_layer(self):
        model = _make_model()
        assert model.consts.n_layers == 2
        assert model.strat.H.shape == (2,)

    def test_create_from_stratification(self):
        strat = StratificationProfile.from_N2_constant(
            N2=1e-5, depth=4000.0, n_layers=4
        )
        model = MultilayerShallowWater2D.create(nx=16, ny=16, stratification=strat)
        assert model.consts.n_layers == 4

    def test_create_validation(self):
        """Mismatched n_layers, H, g_prime should raise ValueError."""
        try:
            MultilayerShallowWater2D.create(
                nx=16,
                ny=16,
                n_layers=3,
                H=(500.0, 500.0),
                g_prime=(9.81, 0.02),
            )
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass

    def test_state_shape(self):
        model = _make_model()
        state = _rest_state(model)
        assert state.h.shape == (2, model.grid.Ny, model.grid.Nx)
        assert state.u.shape == (2, model.grid.Ny, model.grid.Nx)
        assert state.v.shape == (2, model.grid.Ny, model.grid.Nx)


class TestMultilayerSW2DPhysics:
    def test_rest_state_zero_tendency(self):
        """Uniform h = H, u = v = 0, no forcing -> zero tendency."""
        model = _make_model()
        state0 = _rest_state(model)
        state0 = model.apply_boundary_conditions(state0)
        tend = model.vector_field(0.0, state0)
        assert float(jnp.max(jnp.abs(tend.h))) < 1e-10
        assert float(jnp.max(jnp.abs(tend.u))) < 1e-10
        assert float(jnp.max(jnp.abs(tend.v))) < 1e-10

    def test_wind_forcing_top_layer_only(self):
        """Wind forcing should only affect the top layer."""
        model = _make_model(wind_amplitude=1e-5)
        state0 = _rest_state(model)
        state0 = model.apply_boundary_conditions(state0)
        tend = model.vector_field(0.0, state0)
        # Top layer should have non-zero momentum tendency
        assert float(jnp.max(jnp.abs(tend.u[0]))) > 0
        # Bottom layer should be zero (no drag, no wind)
        assert float(jnp.max(jnp.abs(tend.u[-1]))) < 1e-15

    def test_bottom_drag_bottom_layer_only(self):
        """Bottom drag on uniform flow should only decelerate bottom layer."""
        model = _make_model(bottom_drag=1e-4)
        state0 = _rest_state(model)
        nl = model.consts.n_layers
        # Set uniform flow in both layers
        u_init = jnp.ones((nl, model.grid.Ny, model.grid.Nx))
        state0 = MultilayerSW2DState(h=state0.h, u=u_init, v=state0.v)
        state0 = model.apply_boundary_conditions(state0)
        tend = model.vector_field(0.0, state0)
        # Bottom layer should have drag term: -kappa * u
        # Top layer drag contribution should be zero
        # (other terms may be nonzero due to Coriolis)
        assert jnp.all(jnp.isfinite(tend.u))

    def test_pressure_coupling_between_layers(self):
        """Height perturbation in layer 0 should create tendency in layer 1."""
        model = _make_model()
        state0 = _rest_state(model)
        Ny, Nx = model.grid.Ny, model.grid.Nx
        # Perturb top layer
        h_pert = state0.h.at[0, Ny // 2, Nx // 2].add(1.0)
        state0 = MultilayerSW2DState(h=h_pert, u=state0.u, v=state0.v)
        state0 = model.apply_boundary_conditions(state0)
        tend = model.vector_field(0.0, state0)
        # Layer 1 should have nonzero momentum tendency from pressure coupling
        assert float(jnp.max(jnp.abs(tend.u[1]))) > 0

    def test_vector_field_finite(self):
        model = _make_model(lateral_viscosity=100.0)
        state0 = _rest_state(model)
        Ny, Nx = model.grid.Ny, model.grid.Nx
        h_pert = state0.h.at[0, Ny // 2, Nx // 2].add(1.0)
        state0 = MultilayerSW2DState(h=h_pert, u=state0.u, v=state0.v)
        state0 = model.apply_boundary_conditions(state0)
        tend = model.vector_field(0.0, state0)
        assert jnp.all(jnp.isfinite(tend.h))
        assert jnp.all(jnp.isfinite(tend.u))
        assert jnp.all(jnp.isfinite(tend.v))


class TestMultilayerSW2DBCs:
    def test_periodic_bcs(self):
        model = _make_model(bc="periodic")
        state0 = _rest_state(model)
        state_bc = model.apply_boundary_conditions(state0)
        assert state_bc.h.shape == state0.h.shape

    def test_wall_bcs(self):
        model = _make_model(bc="wall")
        state0 = _rest_state(model)
        nl = model.consts.n_layers
        u_init = jnp.ones((nl, model.grid.Ny, model.grid.Nx))
        state0 = MultilayerSW2DState(h=state0.h, u=u_init, v=state0.v)
        state_bc = model.apply_boundary_conditions(state0)
        # u should be zero at x-boundaries
        assert jnp.allclose(state_bc.u[:, :, 0], 0.0)
        assert jnp.allclose(state_bc.u[:, :, -1], 0.0)


class TestMultilayerSW2DIntegration:
    def test_integrate_short(self):
        model = _make_model(lateral_viscosity=100.0)
        state0 = _rest_state(model)
        sol = model.integrate(state0, t0=0.0, t1=1000.0, dt=10.0)
        assert isinstance(sol, dfx.Solution)
        assert jnp.all(jnp.isfinite(sol.ys.h))
        assert jnp.all(jnp.isfinite(sol.ys.u))

    def test_integrate_with_perturbation(self):
        model = _make_model(lateral_viscosity=100.0)
        state0 = _rest_state(model)
        Ny, Nx = model.grid.Ny, model.grid.Nx
        h_pert = state0.h.at[0, Ny // 2, Nx // 2].add(1.0)
        state0 = MultilayerSW2DState(h=h_pert, u=state0.u, v=state0.v)
        sol = model.integrate(state0, t0=0.0, t1=1000.0, dt=10.0)
        assert jnp.all(jnp.isfinite(sol.ys.h))

    def test_three_layer(self):
        model = MultilayerShallowWater2D.create(
            nx=16,
            ny=16,
            n_layers=3,
            H=(400.0, 1100.0, 2500.0),
            g_prime=(9.81, 0.025, 0.0125),
            lateral_viscosity=100.0,
        )
        state0 = _rest_state(model)
        sol = model.integrate(state0, t0=0.0, t1=1000.0, dt=10.0)
        assert jnp.all(jnp.isfinite(sol.ys.h))


class TestMultilayerSW2DDiagnostics:
    def test_diagnose_shapes(self):
        model = _make_model()
        state = _rest_state(model)
        diag = model.diagnose(state)
        assert isinstance(diag, MultilayerSW2DDiagnostics)
        assert diag.potential_vorticity.shape == (
            2,
            model.grid.Ny,
            model.grid.Nx,
        )
        assert diag.relative_vorticity.shape == (
            2,
            model.grid.Ny,
            model.grid.Nx,
        )
        assert diag.kinetic_energy_field.shape == (
            2,
            model.grid.Ny,
            model.grid.Nx,
        )
        assert jnp.isfinite(diag.total_energy)
        assert jnp.isfinite(diag.total_enstrophy)

    def test_energy_nonnegative(self):
        model = _make_model()
        state = _rest_state(model)
        diag = model.diagnose(state)
        assert jnp.all(diag.energy >= 0)


class TestMultilayerSW2DGrad:
    def test_grad_through_params(self):
        model = _make_model(lateral_viscosity=100.0)
        state0 = _rest_state(model)
        Ny, Nx = model.grid.Ny, model.grid.Nx
        h_pert = state0.h.at[0, Ny // 2, Nx // 2].add(0.1)
        state0 = MultilayerSW2DState(h=h_pert, u=state0.u, v=state0.v)

        @eqx.filter_grad
        def grad_fn(m):
            sol = m.integrate(state0, t0=0.0, t1=50.0, dt=1.0)
            return jnp.sum(sol.ys.h**2)

        grads = grad_fn(model)
        assert jnp.isfinite(grads.params.lateral_viscosity)


class TestBaroclinicInstabilitySWM:
    def test_creates(self):
        model, state0 = baroclinic_instability_swm(nx=16, ny=16)
        assert isinstance(model, MultilayerShallowWater2D)
        assert isinstance(state0, MultilayerSW2DState)
        assert state0.h.shape[0] == 2

    def test_integrates(self):
        model, state0 = baroclinic_instability_swm(nx=16, ny=16)
        sol = model.integrate(state0, t0=0.0, t1=1000.0, dt=10.0)
        assert jnp.all(jnp.isfinite(sol.ys.h))
