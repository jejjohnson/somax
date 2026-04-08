"""Tests for elliptic solvers (Poisson, Laplace, Helmholtz)."""

from __future__ import annotations

import jax.numpy as jnp

from somax.models import HelmholtzSolver2D, LaplaceSolver2D, PoissonSolver2D


class TestPoissonSolver2D:
    def test_sinusoidal_dirichlet(self):
        r"""Verify sin(pi*x)*sin(pi*y) with Dirichlet BCs.

        nabla^2 phi = -2*pi^2 * sin(pi*x)*sin(pi*y)
        Exact: phi = sin(pi*x)*sin(pi*y)
        """
        nx, ny = 64, 64
        solver = PoissonSolver2D.create(nx=nx, ny=ny, Lx=1.0, Ly=1.0, bc="dirichlet")
        x = jnp.arange(solver.grid.Nx) * solver.grid.dx
        y = jnp.arange(solver.grid.Ny) * solver.grid.dy
        X, Y = jnp.meshgrid(x, y)
        rhs = -2.0 * jnp.pi**2 * jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)
        phi = solver.solve(rhs)
        exact = jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)
        # Check interior L2 error
        error = jnp.sqrt(jnp.mean((phi[1:-1, 1:-1] - exact[1:-1, 1:-1]) ** 2))
        assert float(error) < 0.1, f"L2 error {float(error):.6f} too large"

    def test_zero_rhs_gives_zero(self):
        """Zero RHS with homogeneous Dirichlet BCs should give zero."""
        solver = PoissonSolver2D.create(nx=32, ny=32, bc="dirichlet")
        rhs = jnp.zeros((solver.grid.Ny, solver.grid.Nx))
        phi = solver.solve(rhs)
        assert float(jnp.max(jnp.abs(phi[1:-1, 1:-1]))) < 1e-10

    def test_periodic_bc(self):
        """Periodic Poisson solve should be consistent."""
        solver = PoissonSolver2D.create(nx=32, ny=32, bc="periodic")
        x = jnp.arange(solver.grid.Nx) * solver.grid.dx
        y = jnp.arange(solver.grid.Ny) * solver.grid.dy
        X, Y = jnp.meshgrid(x, y)
        # Use a periodic RHS: sin(2*pi*x)*sin(2*pi*y)
        k = 2.0 * jnp.pi
        rhs = -2.0 * k**2 * jnp.sin(k * X) * jnp.sin(k * Y)
        phi = solver.solve(rhs)
        exact = jnp.sin(k * X) * jnp.sin(k * Y)
        error = jnp.sqrt(jnp.mean((phi[1:-1, 1:-1] - exact[1:-1, 1:-1]) ** 2))
        assert float(error) < 0.1


class TestLaplaceSolver2D:
    def test_creates_and_solves(self):
        solver = LaplaceSolver2D.create(nx=32, ny=32)
        phi = solver.solve()
        assert phi.shape == (solver.grid.Ny, solver.grid.Nx)
        # With homogeneous Dirichlet and zero RHS, solution should be zero
        assert float(jnp.max(jnp.abs(phi[1:-1, 1:-1]))) < 1e-10


class TestHelmholtzSolver2D:
    def test_creates_and_solves(self):
        solver = HelmholtzSolver2D.create(nx=32, ny=32, lambda_=1.0)
        rhs = jnp.zeros((solver.grid.Ny, solver.grid.Nx))
        phi = solver.solve(rhs)
        assert phi.shape == (solver.grid.Ny, solver.grid.Nx)

    def test_lambda_zero_matches_poisson(self):
        """Helmholtz with lambda=0 should match Poisson."""
        nx, ny = 32, 32
        poisson = PoissonSolver2D.create(nx=nx, ny=ny, bc="dirichlet")
        helmholtz = HelmholtzSolver2D.create(nx=nx, ny=ny, lambda_=0.0, bc="dirichlet")
        x = jnp.arange(poisson.grid.Nx) * poisson.grid.dx
        y = jnp.arange(poisson.grid.Ny) * poisson.grid.dy
        X, Y = jnp.meshgrid(x, y)
        rhs = -2.0 * jnp.pi**2 * jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)
        phi_p = poisson.solve(rhs)
        phi_h = helmholtz.solve(rhs)
        assert jnp.allclose(phi_p, phi_h, atol=1e-10)

    def test_screening_reduces_amplitude(self):
        """Larger lambda should reduce the solution amplitude."""
        h1 = HelmholtzSolver2D.create(nx=32, ny=32, lambda_=1.0)
        h10 = HelmholtzSolver2D.create(nx=32, ny=32, lambda_=10.0)
        x = jnp.arange(h1.grid.Nx) * h1.grid.dx
        y = jnp.arange(h1.grid.Ny) * h1.grid.dy
        X, Y = jnp.meshgrid(x, y)
        rhs = jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)
        phi1 = h1.solve(rhs)
        phi10 = h10.solve(rhs)
        assert float(jnp.max(jnp.abs(phi10))) < float(jnp.max(jnp.abs(phi1)))
