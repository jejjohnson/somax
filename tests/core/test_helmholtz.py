"""Tests for HelmholtzCache (precomputed solver wrappers)."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import pytest

from somax._src.core.helmholtz import (
    DirichletHelmholtzCache,
    HelmholtzCache,
    MultimodalHelmholtzCache,
    NeumannHelmholtzCache,
    PeriodicHelmholtzCache,
)


# ---------------------------------------------------------------------------
# Stub solvers (avoid hard dep on spectraldiffx in unit tests)
# ---------------------------------------------------------------------------


class StubPeriodicSolver(eqx.Module):
    """Minimal stub mimicking SpectralHelmholtzSolver2D.solve()."""

    def solve(self, rhs, alpha: float = 0.0, zero_mean: bool = True):
        # Identity solve: just return rhs scaled by -1/(1+alpha)
        return -rhs / (1.0 + alpha)


class StubDirichletSolver(eqx.Module):
    """Minimal stub mimicking DirichletHelmholtzSolver2D.__call__()."""

    dx: float = 1.0
    dy: float = 1.0
    alpha: float = 0.0

    def __call__(self, rhs):
        return -rhs / (1.0 + self.alpha)


class StubNeumannSolver(eqx.Module):
    """Minimal stub mimicking NeumannHelmholtzSolver2D.__call__()."""

    dx: float = 1.0
    dy: float = 1.0
    alpha: float = 0.0

    def __call__(self, rhs):
        return -rhs / (1.0 + self.alpha)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


def test_helmholtz_cache_is_abstract():
    with pytest.raises(TypeError):
        HelmholtzCache()


# ---------------------------------------------------------------------------
# PeriodicHelmholtzCache
# ---------------------------------------------------------------------------


class TestPeriodicHelmholtzCache:
    def test_solve(self):
        solver = StubPeriodicSolver()
        cache = PeriodicHelmholtzCache(solver=solver, lambda_=2.0)
        rhs = jnp.ones((4, 4))
        result = cache.solve(rhs)
        expected = -rhs / 3.0  # -1/(1+2)
        assert jnp.allclose(result, expected)

    def test_zero_lambda_is_poisson(self):
        solver = StubPeriodicSolver()
        cache = PeriodicHelmholtzCache(solver=solver, lambda_=0.0)
        rhs = jnp.ones((4, 4))
        result = cache.solve(rhs)
        expected = -rhs  # -1/(1+0)
        assert jnp.allclose(result, expected)


# ---------------------------------------------------------------------------
# DirichletHelmholtzCache
# ---------------------------------------------------------------------------


class TestDirichletHelmholtzCache:
    def test_solve(self):
        solver = StubDirichletSolver(alpha=1.5)
        cache = DirichletHelmholtzCache(solver=solver)
        rhs = jnp.ones((4, 4))
        result = cache.solve(rhs)
        expected = -rhs / 2.5
        assert jnp.allclose(result, expected)


# ---------------------------------------------------------------------------
# NeumannHelmholtzCache
# ---------------------------------------------------------------------------


class TestNeumannHelmholtzCache:
    def test_solve(self):
        solver = StubNeumannSolver(alpha=0.5)
        cache = NeumannHelmholtzCache(solver=solver)
        rhs = jnp.ones((4, 4))
        result = cache.solve(rhs)
        expected = -rhs / 1.5
        assert jnp.allclose(result, expected)


# ---------------------------------------------------------------------------
# MultimodalHelmholtzCache
# ---------------------------------------------------------------------------


class TestMultimodalHelmholtzCache:
    def test_n_modes(self):
        caches = tuple(
            PeriodicHelmholtzCache(solver=StubPeriodicSolver(), lambda_=float(k))
            for k in range(3)
        )
        multi = MultimodalHelmholtzCache(caches=caches)
        assert multi.n_modes == 3

    def test_solve_shape(self):
        caches = tuple(
            PeriodicHelmholtzCache(solver=StubPeriodicSolver(), lambda_=float(k))
            for k in range(2)
        )
        multi = MultimodalHelmholtzCache(caches=caches)
        rhs = jnp.ones((2, 4, 4))
        result = multi.solve(rhs)
        assert result.shape == (2, 4, 4)

    def test_per_mode_lambda(self):
        """Each mode should use its own lambda."""
        caches = (
            PeriodicHelmholtzCache(solver=StubPeriodicSolver(), lambda_=0.0),
            PeriodicHelmholtzCache(solver=StubPeriodicSolver(), lambda_=1.0),
        )
        multi = MultimodalHelmholtzCache(caches=caches)
        rhs = jnp.ones((2, 3, 3))
        result = multi.solve(rhs)
        # Mode 0: -1/(1+0) = -1, Mode 1: -1/(1+1) = -0.5
        assert jnp.allclose(result[0], -1.0)
        assert jnp.allclose(result[1], -0.5)
