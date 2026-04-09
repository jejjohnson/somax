"""Tests for reparameterized quasi-geostrophic model."""

from __future__ import annotations

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp

from somax import Diagnostics, SomaxModel
from somax.models import (
    MultilayerShallowWater2D,
    MultilayerSW2DState,
    ReparameterizedQG,
    ReparamQGDiagnostics,
    doublegyre_reparameterized_qg,
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
    return ReparameterizedQG.create(**defaults)


def _rest_state(model):
    nl = model.consts.n_layers
    Ny, Nx = model.grid.Ny, model.grid.Nx
    h0 = jnp.ones((nl, Ny, Nx)) * model.strat.H[:, None, None]
    u0 = jnp.zeros((nl, Ny, Nx))
    v0 = jnp.zeros((nl, Ny, Nx))
    return MultilayerSW2DState(h=h0, u=u0, v=v0)


class TestReparamQGContract:
    def test_is_somax_model(self):
        model = _make_model()
        assert isinstance(model, SomaxModel)

    def test_wraps_multilayer_swm(self):
        model = _make_model()
        assert isinstance(model.swm, MultilayerShallowWater2D)

    def test_diagnostics_type(self):
        assert issubclass(ReparamQGDiagnostics, Diagnostics)

    def test_uses_swm_state(self):
        """QG and SWM share the same state type."""
        model = _make_model()
        state = _rest_state(model)
        assert isinstance(state, MultilayerSW2DState)


class TestReparamQGProjection:
    def test_rest_state_fixed_point(self):
        """At rest (u=v=0, h=H), projection should be identity."""
        model = _make_model()
        state = _rest_state(model)
        projected = model.project(state)
        assert jnp.allclose(projected.h, state.h, atol=1e-6)
        assert jnp.allclose(projected.u, state.u, atol=1e-10)
        assert jnp.allclose(projected.v, state.v, atol=1e-10)

    def test_idempotent(self):
        """Projection should be approximately idempotent: P(P(x)) ~ P(x).

        Residuals are O(dx^2) due to discrete operator stencil differences
        between the Q and G operators at boundaries.
        """
        model = _make_model()
        state = _rest_state(model)
        Ny, Nx = model.grid.Ny, model.grid.Nx
        # Add perturbation
        h_pert = state.h.at[0, Ny // 2, Nx // 2].add(1.0)
        state = MultilayerSW2DState(h=h_pert, u=state.u, v=state.v)

        proj1 = model.project(state)
        proj2 = model.project(proj1)
        # Discrete operators introduce O(dx^2) residuals
        assert jnp.allclose(proj2.h, proj1.h, atol=0.1)
        assert jnp.allclose(proj2.u, proj1.u, atol=0.1)
        assert jnp.allclose(proj2.v, proj1.v, atol=0.1)

    def test_projection_produces_geostrophic_balance(self):
        """Projected state should satisfy approximate geostrophic balance."""
        model = _make_model()
        state = _rest_state(model)
        Ny, Nx = model.grid.Ny, model.grid.Nx
        h_pert = state.h.at[0, Ny // 2, Nx // 2].add(1.0)
        state = MultilayerSW2DState(h=h_pert, u=state.u, v=state.v)

        projected = model.project(state)
        # Projected state should have nonzero velocity (geostrophic)
        assert float(jnp.max(jnp.abs(projected.u))) > 0
        assert float(jnp.max(jnp.abs(projected.v))) > 0

    def test_projection_finite(self):
        """Projection should produce finite results."""
        model = _make_model()
        state = _rest_state(model)
        Ny, Nx = model.grid.Ny, model.grid.Nx
        h_pert = state.h.at[0, Ny // 2, Nx // 2].add(1.0)
        state = MultilayerSW2DState(h=h_pert, u=state.u, v=state.v)

        projected = model.project(state)
        assert jnp.all(jnp.isfinite(projected.h))
        assert jnp.all(jnp.isfinite(projected.u))
        assert jnp.all(jnp.isfinite(projected.v))


class TestReparamQGPhysics:
    def test_rest_state_zero_tendency(self):
        """Uniform h=H, u=v=0, no forcing -> zero tendency."""
        model = _make_model()
        state = _rest_state(model)
        state = model.apply_boundary_conditions(state)
        tend = model.vector_field(0.0, state)
        assert float(jnp.max(jnp.abs(tend.h))) < 1e-10
        assert float(jnp.max(jnp.abs(tend.u))) < 1e-10
        assert float(jnp.max(jnp.abs(tend.v))) < 1e-10

    def test_vector_field_finite(self):
        model = _make_model(lateral_viscosity=100.0)
        state = _rest_state(model)
        Ny, Nx = model.grid.Ny, model.grid.Nx
        h_pert = state.h.at[0, Ny // 2, Nx // 2].add(1.0)
        state = MultilayerSW2DState(h=h_pert, u=state.u, v=state.v)
        state = model.apply_boundary_conditions(state)
        tend = model.vector_field(0.0, state)
        assert jnp.all(jnp.isfinite(tend.h))
        assert jnp.all(jnp.isfinite(tend.u))
        assert jnp.all(jnp.isfinite(tend.v))


class TestReparamQGIntegration:
    def test_integrate_short(self):
        model = _make_model(lateral_viscosity=100.0)
        state0 = _rest_state(model)
        sol = model.integrate(state0, t0=0.0, t1=1000.0, dt=10.0)
        assert isinstance(sol, dfx.Solution)
        assert jnp.all(jnp.isfinite(sol.ys.h))

    def test_integrate_with_perturbation(self):
        model = _make_model(lateral_viscosity=100.0)
        state0 = _rest_state(model)
        Ny, Nx = model.grid.Ny, model.grid.Nx
        h_pert = state0.h.at[0, Ny // 2, Nx // 2].add(1.0)
        state0 = MultilayerSW2DState(h=h_pert, u=state0.u, v=state0.v)
        sol = model.integrate(state0, t0=0.0, t1=1000.0, dt=10.0)
        assert jnp.all(jnp.isfinite(sol.ys.h))

    def test_three_layer(self):
        model = ReparameterizedQG.create(
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


class TestReparamQGDiagnostics:
    def test_diagnose_shapes(self):
        model = _make_model()
        state = _rest_state(model)
        diag = model.diagnose(state)
        assert isinstance(diag, ReparamQGDiagnostics)
        nl = model.consts.n_layers
        Ny, Nx = model.grid.Ny, model.grid.Nx
        assert diag.psi.shape == (nl, Ny, Nx)
        assert diag.u_ageostrophic.shape == (nl, Ny, Nx)
        assert diag.v_ageostrophic.shape == (nl, Ny, Nx)

    def test_ageostrophic_zero_at_rest(self):
        """Ageostrophic velocity should be zero at rest."""
        model = _make_model()
        state = _rest_state(model)
        diag = model.diagnose(state)
        assert float(jnp.max(jnp.abs(diag.u_ageostrophic))) < 1e-10
        assert float(jnp.max(jnp.abs(diag.v_ageostrophic))) < 1e-10


class TestReparamQGGrad:
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
        assert jnp.isfinite(grads.swm.params.lateral_viscosity)


class TestDoublegyreReparamQG:
    def test_creates(self):
        model, state0 = doublegyre_reparameterized_qg(nx=16, ny=16)
        assert isinstance(model, ReparameterizedQG)
        assert isinstance(state0, MultilayerSW2DState)
        assert state0.h.shape[0] == 3

    def test_integrates(self):
        model, state0 = doublegyre_reparameterized_qg(nx=16, ny=16)
        sol = model.integrate(state0, t0=0.0, t1=1000.0, dt=10.0)
        assert jnp.all(jnp.isfinite(sol.ys.h))
