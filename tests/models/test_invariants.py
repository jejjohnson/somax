"""Unit tests for Diagnostics.invariants() across model families.

Model construction + a single diagnose() call — no integration, so these
stay in the fast PR lane.
"""

from __future__ import annotations

import jax.numpy as jnp

from somax._src.models.qg.barotropic import BarotropicQG, BarotropicQGState
from somax._src.models.swm.multilayer import (
    MultilayerShallowWater2D,
    MultilayerSW2DState,
)
from somax._src.models.swm.nonlinear_2d import (
    NonlinearShallowWater2D,
    NonlinearSW2DState,
)
from somax.core import Diagnostics


def _ml_model_state():
    model = MultilayerShallowWater2D.create(
        nx=16, ny=16, n_layers=2, H=(500.0, 1500.0), g_prime=(9.81, 0.02)
    )
    sh = (2, model.grid.Ny, model.grid.Nx)
    h = model.strat.H[:, None, None] * jnp.ones(sh)
    state = model.apply_boundary_conditions(
        MultilayerSW2DState(h=h, u=jnp.zeros(sh), v=jnp.zeros(sh))
    )
    return model, state


class TestBaseInvariants:
    def test_base_diagnostics_empty(self) -> None:
        assert Diagnostics().invariants() == {}


class TestMultilayerInvariants:
    def test_keys_present(self) -> None:
        model, state = _ml_model_state()
        inv = model.diagnose(state).invariants()
        assert set(inv) == {
            "mass",
            "total_energy",
            "potential_enstrophy",
            "casimir_q3",
        }

    def test_mass_positive_and_finite(self) -> None:
        model, state = _ml_model_state()
        inv = model.diagnose(state).invariants()
        assert float(inv["mass"]) > 0
        assert all(bool(jnp.isfinite(v)) for v in inv.values())

    def test_mass_matches_thickness_integral(self) -> None:
        model, state = _ml_model_state()
        diag = model.diagnose(state)
        s = (slice(None), slice(1, -1), slice(1, -1))
        cell_area = model.grid.dx * model.grid.dy
        expected = float(jnp.sum(state.h[s]) * cell_area)
        assert float(jnp.sum(diag.mass)) == jnp.asarray(expected)


class TestNonlinearSWInvariants:
    def test_keys_present(self) -> None:
        model = NonlinearShallowWater2D.create(nx=16, ny=16)
        sh = (model.grid.Ny, model.grid.Nx)
        H0 = model.consts.H0 if hasattr(model.consts, "H0") else 1000.0
        state = model.apply_boundary_conditions(
            NonlinearSW2DState(h=H0 * jnp.ones(sh), u=jnp.zeros(sh), v=jnp.zeros(sh))
        )
        inv = model.diagnose(state).invariants()
        assert set(inv) == {
            "mass",
            "total_energy",
            "potential_enstrophy",
            "casimir_q3",
        }


class TestBarotropicQGInvariants:
    def test_energy_enstrophy_keys(self) -> None:
        model = BarotropicQG.create(nx=16, ny=16)
        state = BarotropicQGState(q=jnp.zeros((model.grid.Ny, model.grid.Nx)))
        inv = model.diagnose(state).invariants()
        assert set(inv) == {"kinetic_energy", "enstrophy"}
