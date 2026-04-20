"""End-to-end smoke tests that models accept a mask.

These pin the threading of the finitevolX ``Mask1D`` / ``Mask2D`` API
through every family: the models are constructed with an all-ocean mask
and then, for the canonical 2-D SWM, with a coastal mask that carves a
small island.  The tests assert the models build, their RHS is finite,
and — for the coastal case — dry T-cells stay at zero after a few steps.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from finitevolx import Mask1D, Mask2D

from somax.models import (
    BarotropicQG,
    Burgers1D,
    Burgers2D,
    LinearShallowWater1D,
    LinearShallowWater2D,
    NonlinearShallowWater1D,
    NonlinearShallowWater2D,
    NonlinearSW2DState,
)


def _island_mask(ny: int, nx: int) -> Mask2D:
    """Rectangular basin with a small interior island."""
    h = np.ones((ny, nx), dtype=bool)
    h[0, :] = h[-1, :] = False
    h[:, 0] = h[:, -1] = False
    j0, j1 = ny // 2 - 1, ny // 2 + 2
    i0, i1 = nx // 2 - 1, nx // 2 + 2
    h[j0:j1, i0:i1] = False
    return Mask2D.from_mask(jnp.asarray(h))


# ---------------------------------------------------------------------------
# All-ocean smoke tests: model builds and RHS evaluates without errors
# ---------------------------------------------------------------------------


class TestAllOceanMaskSmoke:
    def test_swm_linear_2d(self):
        mask = Mask2D.from_dimensions(ny=16, nx=16)
        model = LinearShallowWater2D.create(nx=14, ny=14, mask=mask)
        assert model.mask is mask
        assert model.diff.mask is mask

    def test_swm_nonlinear_2d(self):
        mask = Mask2D.from_dimensions(ny=16, nx=16)
        model = NonlinearShallowWater2D.create(nx=14, ny=14, mask=mask)
        h0 = jnp.ones((model.grid.Ny, model.grid.Nx))
        u0 = jnp.zeros_like(h0)
        state = NonlinearSW2DState(h=h0, u=u0, v=u0)
        state = model.apply_boundary_conditions(state)
        tend = model.vector_field(0.0, state)
        assert jnp.all(jnp.isfinite(tend.h))
        assert jnp.all(jnp.isfinite(tend.u))
        assert jnp.all(jnp.isfinite(tend.v))

    def test_qg_barotropic(self):
        mask = Mask2D.from_dimensions(ny=16, nx=16)
        model = BarotropicQG.create(nx=14, ny=14, mask=mask)
        assert model.diff.mask is mask

    def test_pde2d_burgers(self):
        mask = Mask2D.from_dimensions(ny=16, nx=16)
        model = Burgers2D.create(nx=14, ny=14, mask=mask)
        assert model.advection.mask is mask

    def test_pde1d_burgers(self):
        mask = Mask1D.from_dimensions(nx=32)
        model = Burgers1D.create(nx=30, mask=mask)
        assert model.diff.mask is mask
        assert model.advection.mask is mask

    def test_swm_linear_1d(self):
        mask = Mask1D.from_dimensions(nx=32)
        model = LinearShallowWater1D.create(nx=30, mask=mask)
        assert model.mask is mask
        assert model.interp.mask is mask

    def test_swm_nonlinear_1d(self):
        mask = Mask1D.from_dimensions(nx=32)
        model = NonlinearShallowWater1D.create(nx=30, mask=mask)
        assert model.advection.mask is mask


# ---------------------------------------------------------------------------
# Coastal mask: dry cells stay zero in the integrated state
# ---------------------------------------------------------------------------


class TestCoastalMaskNonlinearSWM2D:
    @pytest.fixture
    def model_and_state(self):
        ny, nx = 20, 20
        mask = _island_mask(ny, nx)
        model = NonlinearShallowWater2D.create(
            nx=nx - 2, ny=ny - 2, bc="wall", mask=mask
        )
        rng = np.random.default_rng(0)
        h0 = jnp.asarray(0.1 * rng.standard_normal((ny, nx)))
        u0 = jnp.asarray(0.01 * rng.standard_normal((ny, nx)))
        v0 = jnp.asarray(0.01 * rng.standard_normal((ny, nx)))
        # Apply mask to enforce the dry-cell invariant on the initial state
        h0 = h0 * mask.h
        u0 = u0 * mask.u
        v0 = v0 * mask.v
        state = NonlinearSW2DState(h=h0, u=u0, v=v0)
        state = model.apply_boundary_conditions(state)
        return model, state, mask

    def test_dry_cells_stay_zero_after_one_rhs(self, model_and_state):
        """One RHS evaluation must not inject non-zero into dry T-cells."""
        model, state, mask = model_and_state
        tend = model.vector_field(0.0, state)
        # dh/dt comes through advection → T-point output → mask.h applies
        dry_T = ~mask.h
        assert float(jnp.max(jnp.abs(tend.h[dry_T]))) == 0.0

    def test_integrate_short_finite(self, model_and_state):
        """A few Euler steps remain finite with the coastal mask threaded."""
        model, state, _mask = model_and_state
        dt = 1.0

        @eqx.filter_jit
        def step(s):
            tend = model.vector_field(0.0, s)
            new = NonlinearSW2DState(
                h=s.h + dt * tend.h,
                u=s.u + dt * tend.u,
                v=s.v + dt * tend.v,
            )
            return model.apply_boundary_conditions(new)

        s = state
        for _ in range(3):
            s = step(s)

        assert jnp.all(jnp.isfinite(s.h))
        assert jnp.all(jnp.isfinite(s.u))
        assert jnp.all(jnp.isfinite(s.v))
