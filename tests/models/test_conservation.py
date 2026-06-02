"""Conservation-drift tests over short integrations.

These integrate a model for a few steps and check the conserved invariants,
so they are explicitly marked ``slow`` (excluded from the fast PR lane).
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from somax._src.models.swm.multilayer import (
    MultilayerShallowWater2D,
    MultilayerSW2DState,
)


@pytest.mark.slow
def test_multilayer_mass_conserved_tightly() -> None:
    """Flux-form mass is conserved to near machine precision (periodic basin).

    No forcing / drag / viscosity, periodic BCs — the continuity equation is
    in flux form, so total mass per layer should be conserved up to the
    time-integration error, far tighter than the energy/enstrophy budget.
    """
    model = MultilayerShallowWater2D.create(
        nx=32,
        ny=32,
        n_layers=2,
        H=(500.0, 1500.0),
        g_prime=(9.81, 0.02),
        lateral_viscosity=0.0,
        bottom_drag=0.0,
        wind_amplitude=0.0,
        bc="periodic",
    )
    sh = (2, model.grid.Ny, model.grid.Nx)
    # A small geostrophic-ish thickness perturbation to make the run nontrivial.
    key = jnp.linspace(0.0, 1.0, model.grid.Nx)
    bump = 1.0 * jnp.sin(2 * jnp.pi * key)[None, None, :]
    h0 = model.strat.H[:, None, None] * jnp.ones(sh) + bump
    state0 = model.apply_boundary_conditions(
        MultilayerSW2DState(h=h0, u=jnp.zeros(sh), v=jnp.zeros(sh))
    )

    mass0 = jnp.sum(model.diagnose(state0).invariants()["mass"])
    sol = model.integrate(state0, t0=0.0, t1=200.0, dt=10.0)
    final = MultilayerSW2DState(h=sol.ys.h[-1], u=sol.ys.u[-1], v=sol.ys.v[-1])
    mass1 = jnp.sum(model.diagnose(final).invariants()["mass"])

    rel_drift = float(jnp.abs(mass1 - mass0) / jnp.abs(mass0))
    assert rel_drift < 1e-6


@pytest.mark.slow
def test_multilayer_energy_bounded_no_growth() -> None:
    """Unforced energy should not grow (implicit dissipation only decays it)."""
    model = MultilayerShallowWater2D.create(
        nx=32,
        ny=32,
        n_layers=2,
        H=(500.0, 1500.0),
        g_prime=(9.81, 0.02),
        lateral_viscosity=0.0,
        bottom_drag=0.0,
        wind_amplitude=0.0,
        bc="periodic",
    )
    sh = (2, model.grid.Ny, model.grid.Nx)
    key = jnp.linspace(0.0, 1.0, model.grid.Nx)
    bump = 1.0 * jnp.sin(2 * jnp.pi * key)[None, None, :]
    h0 = model.strat.H[:, None, None] * jnp.ones(sh) + bump
    state0 = model.apply_boundary_conditions(
        MultilayerSW2DState(h=h0, u=jnp.zeros(sh), v=jnp.zeros(sh))
    )

    e0 = float(model.diagnose(state0).invariants()["total_energy"])
    sol = model.integrate(state0, t0=0.0, t1=200.0, dt=10.0)
    final = MultilayerSW2DState(h=sol.ys.h[-1], u=sol.ys.u[-1], v=sol.ys.v[-1])
    e1 = float(model.diagnose(final).invariants()["total_energy"])

    # Allow a small relative tolerance for time-truncation wiggle; the point
    # is the unforced run does not *grow* energy.
    assert e1 <= e0 * (1.0 + 1e-3)
