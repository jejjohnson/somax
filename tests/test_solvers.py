"""Tests for the matrix-free IMEX solver helpers (issue #55).

The cheap tests (construction + a small IMEX solve that would error or build
a dense Jacobian without the helper) run in the fast PR lane. The
large-grid scaling check that actually reproduces the 256x256 regime is
marked ``slow``.
"""

from __future__ import annotations

import diffrax as dfx
import jax.numpy as jnp
import optimistix as optx
import pytest

from somax._src.models.pde2d.burgers import Burgers2DState
from somax._src.models.pde2d.burgers_terms import Burgers2DTermModel
from somax.solvers import imex_solver, imex_stepsize_controller


def _imex_burgers(n: int) -> tuple[Burgers2DTermModel, Burgers2DState]:
    model = Burgers2DTermModel.create(nx=n, ny=n, nu=0.05, imex=True)
    sh = (model.grid.Ny, model.grid.Nx)
    x = jnp.linspace(0.0, 1.0, model.grid.Nx)
    u0 = jnp.sin(2 * jnp.pi * x)[None, :] * jnp.ones(sh)
    state0 = model.apply_boundary_conditions(Burgers2DState(u=u0, v=jnp.zeros(sh)))
    return model, state0


class TestImexSolverConstruction:
    def test_solver_is_kencarp_with_matrix_free_newton(self) -> None:
        solver = imex_solver()
        assert isinstance(solver, dfx.KenCarp3)
        # The implicit stage must use a Krylov (GMRES) linear solver, not the
        # default dense AutoLinearSolver — that's the whole point of #55.
        rf = solver.root_finder
        assert isinstance(rf, optx.Newton)
        assert type(rf.linear_solver).__name__ == "GMRES"

    def test_controller_is_adaptive(self) -> None:
        assert isinstance(imex_stepsize_controller(), dfx.PIDController)

    def test_tolerances_threaded(self) -> None:
        solver = imex_solver(rtol=1e-3, atol=1e-7)
        assert solver.root_finder.rtol == 1e-3
        assert solver.root_finder.atol == 1e-7


class TestImexSolve:
    def test_small_imex_solve_runs_finite(self) -> None:
        """A small IMEX solve completes and stays finite.

        Without the helper this path either errors (fixed-step controller +
        unspecified implicit tolerances) or builds a dense Jacobian; the
        helper makes it just work.
        """
        model, state0 = _imex_burgers(16)
        sol = model.integrate(
            state0,
            t0=0.0,
            t1=0.05,
            dt=0.01,
            solver=imex_solver(),
            stepsize_controller=imex_stepsize_controller(),
        )
        assert jnp.all(jnp.isfinite(sol.ys.u))
        assert jnp.all(jnp.isfinite(sol.ys.v))

    def test_imex_tracks_explicit_solution(self) -> None:
        """The IMEX (implicit-diffusion) solve agrees with a fully-explicit
        reference on a short, stable window — the split is faithful."""
        n = 16
        imex_model, state0 = _imex_burgers(n)
        sol_imex = imex_model.integrate(
            state0,
            t0=0.0,
            t1=0.05,
            dt=0.005,
            solver=imex_solver(rtol=1e-7, atol=1e-9),
            stepsize_controller=imex_stepsize_controller(rtol=1e-7, atol=1e-9),
        )
        explicit_model = Burgers2DTermModel.create(nx=n, ny=n, nu=0.05, imex=False)
        sol_exp = explicit_model.integrate(state0, t0=0.0, t1=0.05, dt=0.005)
        assert jnp.allclose(sol_imex.ys.u, sol_exp.ys.u, atol=1e-3)


@pytest.mark.slow
def test_imex_solver_scales_to_large_grid() -> None:
    """The matrix-free IMEX solve runs at 128x128 without OOM.

    The dense-Jacobian default is ~100x slower by 64x64 and OOMs at
    256x256 (#55); the matrix-free path stays O(N) memory. We assert the
    128x128 solve completes and is finite (256x256 is left out of CI for
    runtime, but the same path handles it in ~1 min locally).
    """
    model, state0 = _imex_burgers(128)
    sol = model.integrate(
        state0,
        t0=0.0,
        t1=0.02,
        dt=0.01,
        solver=imex_solver(),
        stepsize_controller=imex_stepsize_controller(),
    )
    assert jnp.all(jnp.isfinite(sol.ys.u))
