"""Precomputed Helmholtz solver cache for spectral PDE inversion."""

from __future__ import annotations

import abc

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class HelmholtzCache(eqx.Module):
    """Cached Helmholtz solver for repeated (lap - lambda) psi = f solves.

    Wraps a spectraldiffx Helmholtz solver together with the Helmholtz
    parameter so that models can call ``cache.solve(rhs)`` without
    re-specifying grid or BC details each time.

    This is the base class. Use one of the concrete subclasses below.
    """

    @abc.abstractmethod
    def solve(self, rhs: Array) -> Array:
        """Solve the Helmholtz equation for the given RHS."""
        ...


class PeriodicHelmholtzCache(HelmholtzCache):
    """Helmholtz cache for periodic (FFT-based) domains.

    Wraps a ``SpectralHelmholtzSolver2D`` from spectraldiffx.

    Attributes:
        solver: A ``SpectralHelmholtzSolver2D`` instance.
        lambda_: Helmholtz parameter (>= 0).
        zero_mean: Whether to project out the zero mode (default True).
    """

    solver: eqx.Module
    lambda_: float
    zero_mean: bool = True

    def solve(self, rhs: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """Solve (lap - lambda) psi = rhs on a periodic domain."""
        return self.solver.solve(rhs, alpha=self.lambda_, zero_mean=self.zero_mean)


class DirichletHelmholtzCache(HelmholtzCache):
    """Helmholtz cache for Dirichlet (DST-based) domains.

    Wraps a ``DirichletHelmholtzSolver2D`` from spectraldiffx.
    The solver's ``alpha`` field is set at construction time, so
    ``solve`` just forwards the RHS.

    Attributes:
        solver: A ``DirichletHelmholtzSolver2D`` instance (stores dx, dy, alpha).
    """

    solver: eqx.Module

    def solve(self, rhs: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """Solve (lap - alpha) psi = rhs with Dirichlet BCs."""
        return self.solver(rhs)


class NeumannHelmholtzCache(HelmholtzCache):
    """Helmholtz cache for Neumann (DCT-based) domains.

    Wraps a ``NeumannHelmholtzSolver2D`` from spectraldiffx.

    Attributes:
        solver: A ``NeumannHelmholtzSolver2D`` instance (stores dx, dy, alpha).
    """

    solver: eqx.Module

    def solve(self, rhs: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """Solve (lap - alpha) psi = rhs with Neumann BCs."""
        return self.solver(rhs)


class MultimodalHelmholtzCache(eqx.Module):
    """Batched Helmholtz cache for multilayer modal solves.

    Stores one Helmholtz cache per vertical mode, enabling efficient
    per-mode PV inversion for each mode k.

    Attributes:
        caches: Tuple of ``HelmholtzCache`` instances, one per mode.
    """

    caches: tuple[HelmholtzCache, ...]

    @property
    def n_modes(self) -> int:
        return len(self.caches)

    def solve(self, rhs: Float[Array, "nl Ny Nx"]) -> Float[Array, "nl Ny Nx"]:
        """Solve the Helmholtz equation independently for each mode.

        Args:
            rhs: Right-hand side with shape (n_modes, Ny, Nx).

        Returns:
            Solution with shape (n_modes, Ny, Nx).
        """
        solutions = [self.caches[k].solve(rhs[k]) for k in range(self.n_modes)]
        return jnp.stack(solutions, axis=0)
