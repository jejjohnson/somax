"""Tests for the reference-free evaluation metrics in ``somax.eval``.

States are built from a real ``LinearShallowWater2D`` so the metrics run
against genuine finitevolx ``Difference2D`` / ``Coriolis2D`` operators and
grid, not mocks.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from somax._src.models.swm.linear_2d import LinearShallowWater2D, LinearSW2DState
from somax.eval import (
    compute_eval_metrics,
    geostrophic_imbalance,
    kinetic_energy,
    rms_divergence,
    total_enstrophy,
)


@pytest.fixture
def model() -> LinearShallowWater2D:
    """A small f-plane shallow-water model (β=0 → constant Coriolis)."""
    return LinearShallowWater2D.create(nx=32, ny=32, Lx=1e6, Ly=1e6, f0=1e-4, beta=0.0)


def _state(model: LinearShallowWater2D, h, u, v) -> LinearSW2DState:
    shape = (model.grid.Ny, model.grid.Nx)
    return LinearSW2DState(
        h=jnp.broadcast_to(jnp.asarray(h, dtype=float), shape),
        u=jnp.broadcast_to(jnp.asarray(u, dtype=float), shape),
        v=jnp.broadcast_to(jnp.asarray(v, dtype=float), shape),
    )


# ----------------------------------------------------------------------
# rms_divergence
# ----------------------------------------------------------------------


def test_rms_divergence_zero_for_uniform_flow(model):
    """A spatially constant velocity field is divergence-free."""
    state = _state(model, h=0.0, u=1.3, v=-0.7)
    assert float(rms_divergence(state.u, state.v, model.diff)) == pytest.approx(
        0.0, abs=1e-12
    )


def test_rms_divergence_positive_for_diverging_flow(model):
    """A flow that ramps in x has nonzero divergence."""
    x = jnp.arange(model.grid.Nx, dtype=float) * model.grid.dx
    u = jnp.broadcast_to(x[None, :], (model.grid.Ny, model.grid.Nx))
    state = LinearSW2DState(h=jnp.zeros_like(u), u=u, v=jnp.zeros_like(u))
    assert float(rms_divergence(state.u, state.v, model.diff)) > 1e-8


# ----------------------------------------------------------------------
# total_enstrophy
# ----------------------------------------------------------------------


def test_total_enstrophy_zero_for_irrotational_flow(model):
    """A spatially constant velocity field has zero vorticity."""
    state = _state(model, h=0.0, u=2.0, v=3.0)
    enstrophy = float(total_enstrophy(state.u, state.v, model.diff, model.grid))
    assert enstrophy == pytest.approx(0.0, abs=1e-9)


def test_total_enstrophy_positive_for_sheared_flow(model):
    """A flow sheared in y carries vorticity ⇒ positive enstrophy."""
    y = jnp.arange(model.grid.Ny, dtype=float) * model.grid.dy
    u = jnp.broadcast_to(y[:, None], (model.grid.Ny, model.grid.Nx))
    state = LinearSW2DState(h=jnp.zeros_like(u), u=u, v=jnp.zeros_like(u))
    assert float(total_enstrophy(state.u, state.v, model.diff, model.grid)) > 0.0


# ----------------------------------------------------------------------
# kinetic_energy
# ----------------------------------------------------------------------


def test_kinetic_energy_matches_closed_form(model):
    """KE of a uniform field equals 0.5·(u²+v²)·(interior area)."""
    u0, v0 = 2.0, 1.0
    state = _state(model, h=0.0, u=u0, v=v0)
    n_interior = (model.grid.Ny - 2) * (model.grid.Nx - 2)
    cell_area = model.grid.dx * model.grid.dy
    expected = 0.5 * (u0**2 + v0**2) * n_interior * cell_area
    got = float(kinetic_energy(state.u, state.v, model.grid))
    assert got == pytest.approx(expected, rel=1e-6)


def test_kinetic_energy_zero_at_rest(model):
    state = _state(model, h=0.0, u=0.0, v=0.0)
    assert float(kinetic_energy(state.u, state.v, model.grid)) == 0.0


# ----------------------------------------------------------------------
# geostrophic_imbalance
# ----------------------------------------------------------------------


def test_geostrophic_imbalance_unity_for_pure_inertial_flow(model):
    """With no pressure gradient (h=0) the residual *is* the Coriolis term,
    so the ageostrophic fraction is exactly 1."""
    state = _state(model, h=0.0, u=0.5, v=-0.3)
    assert float(geostrophic_imbalance(model, state)) == pytest.approx(1.0, rel=1e-6)


def test_geostrophic_imbalance_near_zero_for_balanced_jet(model):
    """A uniform geostrophic jet is in exact discrete balance.

    Height linear in x (so ∂ₓη is an exact constant and ∂_yη ≡ 0) balanced
    by a constant along-jet velocity ``v = (g/f)·∂ₓη`` and ``u = 0`` satisfies
    the C-grid momentum balance term-for-term, independent of staggering.
    """
    g, f0 = model.consts.gravity, model.consts.f0
    ny, nx = model.grid.Ny, model.grid.Nx
    slope = 1e-3  # ∂η/∂x  (m height per m)
    x = jnp.arange(nx, dtype=float) * model.grid.dx
    h = jnp.broadcast_to(slope * x[None, :], (ny, nx))
    v = jnp.full((ny, nx), (g / f0) * slope)
    u = jnp.zeros((ny, nx))
    state = LinearSW2DState(h=h, u=u, v=v)
    imbalance = float(geostrophic_imbalance(model, state))
    assert imbalance < 1e-6


def test_geostrophic_imbalance_scale_invariant(model):
    """The imbalance is a ratio, so scaling the whole state leaves it fixed."""
    base = _state(model, h=2.0, u=0.5, v=-0.3)
    scaled = LinearSW2DState(h=3.0 * base.h, u=3.0 * base.u, v=3.0 * base.v)
    assert float(geostrophic_imbalance(model, scaled)) == pytest.approx(
        float(geostrophic_imbalance(model, base)), rel=1e-5
    )


# ----------------------------------------------------------------------
# compute_eval_metrics dispatcher
# ----------------------------------------------------------------------


def test_compute_eval_metrics_fluid_model(model):
    state = _state(model, h=1.0, u=0.5, v=0.25)
    metrics = compute_eval_metrics(model, state)
    assert set(metrics) == {
        "rms_divergence",
        "total_enstrophy",
        "kinetic_energy",
        "geostrophic_imbalance",
    }
    assert all(isinstance(value, float) for value in metrics.values())
    assert all(jnp.isfinite(jnp.asarray(value)) for value in metrics.values())


def test_compute_eval_metrics_empty_for_non_fluid_model():
    """A model without C-grid velocity fields yields no metrics."""

    class _Scalar:  # stand-in for a Lorenz-style state (no u/v)
        x = jnp.array(1.0)

    class _NonFluidModel:  # no .diff / .grid
        pass

    assert compute_eval_metrics(_NonFluidModel(), _Scalar()) == {}
