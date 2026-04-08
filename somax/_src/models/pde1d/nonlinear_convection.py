"""1D nonlinear convection (inviscid Burgers): du/dt + u du/dx = 0."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from finitevolx import Advection1D, ArakawaCGrid1D
from jaxtyping import Array, PyTree

from somax._src.core.model import SomaxModel
from somax._src.core.types import Diagnostics, State


class NonlinearConvection1DState(State):
    """State for 1D nonlinear convection.

    Args:
        u: Scalar field on T-points, shape ``(Nx,)`` including ghost cells.
    """

    u: Array


class NonlinearConvection1DDiagnostics(Diagnostics):
    """Diagnostics for 1D nonlinear convection.

    Args:
        energy: Integrated energy 0.5 * sum(u^2).
    """

    energy: Array


class NonlinearConvection1D(SomaxModel):
    """1D nonlinear convection (inviscid Burgers) on an Arakawa C-grid.

    Solves ``du/dt + u * du/dx = 0`` using upwind flux reconstruction.

    Args:
        grid: 1D Arakawa C-grid.
        advection: Advection operator.
        periodic: Whether to use periodic boundary conditions.
        method: Reconstruction method for advection (default ``"upwind1"``).
    """

    grid: ArakawaCGrid1D = eqx.field(static=True)
    advection: Advection1D = eqx.field(static=True)
    periodic: bool = eqx.field(static=True, default=True)
    method: str = eqx.field(static=True, default="upwind1")

    def vector_field(
        self, t: float, state: PyTree, args: PyTree | None = None
    ) -> NonlinearConvection1DState:
        """Compute tendency: du/dt = -d(u*u)/dx via upwind flux."""
        du_dt = self.advection(state.u, state.u, method=self.method)
        return NonlinearConvection1DState(u=du_dt)

    def apply_boundary_conditions(self, state: PyTree) -> NonlinearConvection1DState:
        """Apply boundary conditions to the state."""
        u = state.u
        if self.periodic:
            u = u.at[0].set(u[-2])
            u = u.at[-1].set(u[1])
        else:
            u = u.at[0].set(-u[1])
            u = u.at[-1].set(-u[-2])
        return NonlinearConvection1DState(u=u)

    def diagnose(self, state: PyTree) -> NonlinearConvection1DDiagnostics:
        """Compute energy diagnostic."""
        energy = 0.5 * jnp.sum(state.u[1:-1] ** 2) * self.grid.dx
        return NonlinearConvection1DDiagnostics(energy=energy)

    @staticmethod
    def create(
        nx: int = 100,
        Lx: float = 2.0,
        periodic: bool = True,
        method: str = "upwind1",
    ) -> NonlinearConvection1D:
        """Convenience factory.

        Args:
            nx: Number of interior grid cells.
            Lx: Domain length.
            periodic: Use periodic BCs if True, Dirichlet if False.
            method: Advection reconstruction method.

        Returns:
            A ``NonlinearConvection1D`` model instance.
        """
        grid = ArakawaCGrid1D.from_interior(nx, Lx)
        advection = Advection1D(grid=grid)
        return NonlinearConvection1D(
            grid=grid,
            advection=advection,
            periodic=periodic,
            method=method,
        )
