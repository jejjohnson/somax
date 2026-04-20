"""2D diffusion equation: du/dt = nu * laplacian(u)."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from finitevolx import CartesianGrid2D, Difference2D, Mask2D, enforce_periodic
from jaxtyping import Array, PyTree

from somax._src.core.model import SomaxModel
from somax._src.core.types import Diagnostics, Params, State


class Diffusion2DParams(Params):
    """Differentiable parameters for 2D diffusion.

    Args:
        nu: Kinematic viscosity (diffusion coefficient).
    """

    nu: Array


class Diffusion2DState(State):
    """State for 2D diffusion.

    Args:
        u: Scalar field on T-points, shape ``(Ny, Nx)``.
    """

    u: Array


class Diffusion2DDiagnostics(Diagnostics):
    """Diagnostics for 2D diffusion.

    Args:
        energy: Integrated energy.
    """

    energy: Array


class Diffusion2D(SomaxModel):
    r"""2D diffusion equation on an Arakawa C-grid.

    Solves ``du/dt = nu * (d²u/dx² + d²u/dy²)``.

    Args:
        params: Differentiable parameters (viscosity ``nu``).
        grid: 2D Arakawa C-grid.
        diff: Difference operators.
        mask: Optional Arakawa C-grid mask (``None`` = all-ocean).
    """

    params: Diffusion2DParams
    grid: CartesianGrid2D = eqx.field(static=True)
    diff: Difference2D
    mask: Mask2D | None

    def vector_field(
        self, t: float, state: PyTree, args: PyTree | None = None
    ) -> Diffusion2DState:
        r"""Compute tendency: du/dt = nu * laplacian(u)."""
        du_dt = self.params.nu * self.diff.laplacian(state.u)
        return Diffusion2DState(u=du_dt)

    def apply_boundary_conditions(self, state: PyTree) -> Diffusion2DState:
        """Apply periodic boundary conditions."""
        return Diffusion2DState(u=enforce_periodic(state.u))

    def diagnose(self, state: PyTree) -> Diffusion2DDiagnostics:
        """Compute energy diagnostic."""
        interior = state.u[1:-1, 1:-1]
        energy = 0.5 * jnp.sum(interior**2) * self.grid.dx * self.grid.dy
        return Diffusion2DDiagnostics(energy=energy)

    @staticmethod
    def create(
        nx: int = 64,
        ny: int = 64,
        Lx: float = 2.0,
        Ly: float = 2.0,
        nu: float = 0.01,
        mask: Mask2D | None = None,
    ) -> Diffusion2D:
        """Convenience factory.

        Args:
            nx: Number of interior cells in x.
            ny: Number of interior cells in y.
            Lx: Domain length in x.
            Ly: Domain length in y.
            nu: Kinematic viscosity.
            mask: Optional Arakawa C-grid mask (``None`` = all-ocean).

        Returns:
            A ``Diffusion2D`` model instance.
        """
        grid = CartesianGrid2D.from_interior(nx, ny, Lx, Ly)
        params = Diffusion2DParams(nu=jnp.array(nu))
        diff = Difference2D(grid=grid, mask=mask)
        return Diffusion2D(params=params, grid=grid, diff=diff, mask=mask)
