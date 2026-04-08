"""1D Burgers equation: du/dt + u du/dx = nu * d2u/dx2."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from finitevolx import Advection1D, ArakawaCGrid1D, Difference1D
from jaxtyping import Array, PyTree

from somax._src.core.model import SomaxModel
from somax._src.core.types import Diagnostics, Params, State


class Burgers1DParams(Params):
    """Differentiable parameters for 1D Burgers equation.

    Args:
        nu: Kinematic viscosity (diffusion coefficient).
    """

    nu: Array


class Burgers1DState(State):
    """State for 1D Burgers equation.

    Args:
        u: Scalar field on T-points, shape ``(Nx,)`` including ghost cells.
    """

    u: Array


class Burgers1DDiagnostics(Diagnostics):
    """Diagnostics for 1D Burgers equation.

    Args:
        energy: Integrated energy 0.5 * sum(u^2).
    """

    energy: Array


class Burgers1D(SomaxModel):
    r"""1D Burgers equation on an Arakawa C-grid.

    Solves ``du/dt + u * du/dx = nu * d²u/dx²``, combining nonlinear
    advection with viscous diffusion. The viscosity ``nu`` is learnable
    and visible to ``jax.grad``.

    Args:
        params: Differentiable parameters (viscosity ``nu``).
        grid: 1D Arakawa C-grid.
        diff: Difference operators (for diffusion).
        advection: Advection operator (for nonlinear convection).
        periodic: Whether to use periodic boundary conditions.
        method: Reconstruction method for advection (default ``"upwind1"``).
    """

    params: Burgers1DParams
    grid: ArakawaCGrid1D = eqx.field(static=True)
    diff: Difference1D = eqx.field(static=True)
    advection: Advection1D = eqx.field(static=True)
    periodic: bool = eqx.field(static=True, default=True)
    method: str = eqx.field(static=True, default="upwind1")

    def vector_field(
        self, t: float, state: PyTree, args: PyTree | None = None
    ) -> Burgers1DState:
        r"""Compute tendency: du/dt = -u * du/dx + nu * d²u/dx²."""
        advection_term = self.advection(state.u, state.u, method=self.method)
        diffusion_term = self.params.nu * self.diff.laplacian(state.u)
        return Burgers1DState(u=advection_term + diffusion_term)

    def apply_boundary_conditions(self, state: PyTree) -> Burgers1DState:
        """Apply boundary conditions to the state."""
        u = state.u
        if self.periodic:
            u = u.at[0].set(u[-2])
            u = u.at[-1].set(u[1])
        else:
            u = u.at[0].set(-u[1])
            u = u.at[-1].set(-u[-2])
        return Burgers1DState(u=u)

    def diagnose(self, state: PyTree) -> Burgers1DDiagnostics:
        """Compute energy diagnostic."""
        energy = 0.5 * jnp.sum(state.u[1:-1] ** 2) * self.grid.dx
        return Burgers1DDiagnostics(energy=energy)

    @staticmethod
    def create(
        nx: int = 100,
        Lx: float = 2.0,
        nu: float = 0.01,
        periodic: bool = True,
        method: str = "upwind1",
    ) -> Burgers1D:
        """Convenience factory.

        Args:
            nx: Number of interior grid cells.
            Lx: Domain length.
            nu: Kinematic viscosity.
            periodic: Use periodic BCs if True, Dirichlet if False.
            method: Advection reconstruction method.

        Returns:
            A ``Burgers1D`` model instance.
        """
        grid = ArakawaCGrid1D.from_interior(nx, Lx)
        params = Burgers1DParams(nu=jnp.array(nu))
        diff = Difference1D(grid=grid)
        advection = Advection1D(grid=grid)
        return Burgers1D(
            params=params,
            grid=grid,
            diff=diff,
            advection=advection,
            periodic=periodic,
            method=method,
        )
