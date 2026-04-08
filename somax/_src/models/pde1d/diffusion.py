"""1D diffusion equation: du/dt = nu * d2u/dx2."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from finitevolx import ArakawaCGrid1D, Difference1D
from jaxtyping import Array, PyTree

from somax._src.core.model import SomaxModel
from somax._src.core.types import Diagnostics, Params, State


class Diffusion1DParams(Params):
    """Differentiable parameters for 1D diffusion.

    Args:
        nu: Kinematic viscosity (diffusion coefficient).
    """

    nu: Array


class Diffusion1DState(State):
    """State for 1D diffusion.

    Args:
        u: Scalar field on T-points, shape ``(Nx,)`` including ghost cells.
    """

    u: Array


class Diffusion1DDiagnostics(Diagnostics):
    """Diagnostics for 1D diffusion.

    Args:
        energy: Integrated energy 0.5 * sum(u^2).
    """

    energy: Array


class Diffusion1D(SomaxModel):
    r"""1D diffusion equation on an Arakawa C-grid.

    Solves ``du/dt = nu * d²u/dx²`` where ``nu`` is a learnable
    viscosity visible to ``jax.grad``.

    Args:
        params: Differentiable parameters (viscosity ``nu``).
        grid: 1D Arakawa C-grid.
        diff: Difference operators.
        periodic: Whether to use periodic boundary conditions.
    """

    params: Diffusion1DParams
    grid: ArakawaCGrid1D = eqx.field(static=True)
    diff: Difference1D = eqx.field(static=True)
    periodic: bool = eqx.field(static=True, default=True)

    def vector_field(
        self, t: float, state: PyTree, args: PyTree | None = None
    ) -> Diffusion1DState:
        r"""Compute tendency: du/dt = nu * d²u/dx²."""
        du_dt = self.params.nu * self.diff.laplacian(state.u)
        return Diffusion1DState(u=du_dt)

    def apply_boundary_conditions(self, state: PyTree) -> Diffusion1DState:
        """Apply boundary conditions to the state."""
        u = state.u
        if self.periodic:
            u = u.at[0].set(u[-2])
            u = u.at[-1].set(u[1])
        else:
            # Dirichlet: u = 0 at boundaries
            u = u.at[0].set(-u[1])
            u = u.at[-1].set(-u[-2])
        return Diffusion1DState(u=u)

    def diagnose(self, state: PyTree) -> Diffusion1DDiagnostics:
        """Compute energy diagnostic."""
        energy = 0.5 * jnp.sum(state.u[1:-1] ** 2) * self.grid.dx
        return Diffusion1DDiagnostics(energy=energy)

    @staticmethod
    def create(
        nx: int = 100,
        Lx: float = 2.0,
        nu: float = 0.01,
        periodic: bool = True,
    ) -> Diffusion1D:
        """Convenience factory.

        Args:
            nx: Number of interior grid cells.
            Lx: Domain length.
            nu: Kinematic viscosity.
            periodic: Use periodic BCs if True, Dirichlet if False.

        Returns:
            A ``Diffusion1D`` model instance.
        """
        grid = ArakawaCGrid1D.from_interior(nx, Lx)
        params = Diffusion1DParams(nu=jnp.array(nu))
        diff = Difference1D(grid=grid)
        return Diffusion1D(
            params=params,
            grid=grid,
            diff=diff,
            periodic=periodic,
        )
