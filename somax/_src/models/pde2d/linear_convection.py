"""2D linear convection: du/dt + cx du/dx + cy du/dy = 0."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from finitevolx import ArakawaCGrid2D, Difference2D, Interpolation2D, enforce_periodic
from jaxtyping import Array, PyTree

from somax._src.core.model import SomaxModel
from somax._src.core.types import Diagnostics, Params, State


class LinearConvection2DParams(Params):
    """Differentiable parameters for 2D linear convection.

    Args:
        cx: Wave speed in x-direction.
        cy: Wave speed in y-direction.
    """

    cx: Array
    cy: Array


class LinearConvection2DState(State):
    """State for 2D linear convection.

    Args:
        u: Scalar field on T-points, shape ``(Ny, Nx)`` including ghost cells.
    """

    u: Array


class LinearConvection2DDiagnostics(Diagnostics):
    """Diagnostics for 2D linear convection.

    Args:
        energy: Integrated energy.
    """

    energy: Array


class LinearConvection2D(SomaxModel):
    """2D linear convection on an Arakawa C-grid.

    Solves ``du/dt + cx * du/dx + cy * du/dy = 0``.

    Args:
        params: Differentiable parameters (wave speeds ``cx``, ``cy``).
        grid: 2D Arakawa C-grid.
        diff: Difference operators.
        interp: Interpolation operators.
    """

    params: LinearConvection2DParams
    grid: ArakawaCGrid2D = eqx.field(static=True)
    diff: Difference2D = eqx.field(static=True)
    interp: Interpolation2D = eqx.field(static=True)

    def vector_field(
        self, t: float, state: PyTree, args: PyTree | None = None
    ) -> LinearConvection2DState:
        """Compute tendency: du/dt = -cx * du/dx - cy * du/dy."""
        # Flux in x: c_x * u at U-points, then divergence back to T
        flux_x = self.params.cx * self.interp.T_to_U(state.u)
        # Flux in y: c_y * u at V-points, then divergence back to T
        flux_y = self.params.cy * self.interp.T_to_V(state.u)
        du_dt = -(self.diff.diff_x_U_to_T(flux_x) + self.diff.diff_y_V_to_T(flux_y))
        return LinearConvection2DState(u=du_dt)

    def apply_boundary_conditions(self, state: PyTree) -> LinearConvection2DState:
        """Apply periodic boundary conditions."""
        return LinearConvection2DState(u=enforce_periodic(state.u))

    def diagnose(self, state: PyTree) -> LinearConvection2DDiagnostics:
        """Compute energy diagnostic."""
        interior = state.u[1:-1, 1:-1]
        energy = 0.5 * jnp.sum(interior**2) * self.grid.dx * self.grid.dy
        return LinearConvection2DDiagnostics(energy=energy)

    @staticmethod
    def create(
        nx: int = 64,
        ny: int = 64,
        Lx: float = 2.0,
        Ly: float = 2.0,
        cx: float = 1.0,
        cy: float = 1.0,
    ) -> LinearConvection2D:
        """Convenience factory.

        Args:
            nx: Number of interior cells in x.
            ny: Number of interior cells in y.
            Lx: Domain length in x.
            Ly: Domain length in y.
            cx: Wave speed in x.
            cy: Wave speed in y.

        Returns:
            A ``LinearConvection2D`` model instance.
        """
        grid = ArakawaCGrid2D.from_interior(nx, ny, Lx, Ly)
        params = LinearConvection2DParams(cx=jnp.array(cx), cy=jnp.array(cy))
        diff = Difference2D(grid=grid)
        interp = Interpolation2D(grid=grid)
        return LinearConvection2D(params=params, grid=grid, diff=diff, interp=interp)
