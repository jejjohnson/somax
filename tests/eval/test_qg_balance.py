"""Tests for the QG balance residual and invariant wiring in eval metrics.

Model construction + a few diagnose/inversion calls — no integration, fast
PR lane.
"""

from __future__ import annotations

import jax.numpy as jnp

from somax._src.models.qg.barotropic import BarotropicQG, BarotropicQGState
from somax._src.models.swm.multilayer import (
    MultilayerShallowWater2D,
    MultilayerSW2DState,
)
from somax.eval import compute_eval_metrics, qg_balance_residual


def _barotropic():
    return BarotropicQG.create(nx=32, ny=32, lateral_viscosity=100.0)


class TestQGBalanceResidual:
    def test_zero_pv_is_zero_residual(self) -> None:
        model = _barotropic()
        state = BarotropicQGState(q=jnp.zeros((model.grid.Ny, model.grid.Nx)))
        assert float(qg_balance_residual(model, state)) == 0.0

    def test_inverted_state_closes_round_trip(self) -> None:
        """psi from inversion, then q = laplacian(psi) is self-consistent."""
        model = _barotropic()
        # A nonzero interior PV bump.
        q = jnp.zeros((model.grid.Ny, model.grid.Nx)).at[16, 16].set(1e-5)
        state = BarotropicQGState(q=q)
        # The residual measures ||laplacian(invert(q)) - q|| / ||q||; for a
        # consistent elliptic operator this is small.
        assert float(qg_balance_residual(model, state)) < 1e-3


class TestComputeEvalMetricsQG:
    def test_qg_model_reports_balance_and_invariants(self) -> None:
        model = _barotropic()
        q = jnp.zeros((model.grid.Ny, model.grid.Nx)).at[16, 16].set(1e-5)
        metrics = compute_eval_metrics(model, BarotropicQGState(q=q))
        assert "qg_balance_residual" in metrics
        assert "invariant_kinetic_energy" in metrics
        assert "invariant_enstrophy" in metrics
        # QG models do NOT report the velocity-divergence metrics.
        assert "rms_divergence" not in metrics

    def test_swm_model_reports_invariants(self) -> None:
        model = MultilayerShallowWater2D.create(
            nx=16, ny=16, n_layers=2, H=(500.0, 1500.0), g_prime=(9.81, 0.02)
        )
        sh = (2, model.grid.Ny, model.grid.Nx)
        h = model.strat.H[:, None, None] * jnp.ones(sh)
        state = model.apply_boundary_conditions(
            MultilayerSW2DState(h=h, u=jnp.zeros(sh), v=jnp.zeros(sh))
        )
        metrics = compute_eval_metrics(model, state)
        assert "invariant_mass" in metrics
        assert "invariant_total_energy" in metrics
        # 3D multilayer state -> velocity-divergence metrics do not apply.
        assert "rms_divergence" not in metrics
