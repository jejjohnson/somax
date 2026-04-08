"""2D Burgers equation: du/dt + u*grad(u) = nu * laplacian(u)."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from finitevolx import (
    Advection2D as FVXAdvection2D,
    ArakawaCGrid2D,
    Difference2D,
    Interpolation2D,
    enforce_periodic,
)
from jaxtyping import Array, PyTree

from somax._src.core.model import SomaxModel
from somax._src.core.types import Diagnostics, Params, State


class Burgers2DParams(Params):
    """Differentiable parameters for 2D Burgers equation.

    Args:
        nu: Kinematic viscosity (diffusion coefficient).
    """

    nu: Array


class Burgers2DState(State):
    """State for 2D Burgers equation.

    Args:
        u: x-velocity on T-points, shape ``(Ny, Nx)``.
        v: y-velocity on T-points, shape ``(Ny, Nx)``.
    """

    u: Array
    v: Array


class Burgers2DDiagnostics(Diagnostics):
    """Diagnostics for 2D Burgers equation.

    Args:
        kinetic_energy: Integrated KE.
    """

    kinetic_energy: Array


class Burgers2D(SomaxModel):
    r"""2D Burgers equation on an Arakawa C-grid.

    Solves the system::

        du/dt + u * du/dx + v * du/dy = nu * laplacian(u)
        dv/dt + u * dv/dx + v * dv/dy = nu * laplacian(v)

    Args:
        params: Differentiable parameters (viscosity ``nu``).
        grid: 2D Arakawa C-grid.
        diff: Difference operators.
        advection: Advection operator.
        interp: Interpolation operators.
        method: Reconstruction method for advection (default ``"upwind1"``).
    """

    params: Burgers2DParams
    grid: ArakawaCGrid2D = eqx.field(static=True)
    diff: Difference2D = eqx.field(static=True)
    advection: FVXAdvection2D = eqx.field(static=True)
    interp: Interpolation2D = eqx.field(static=True)
    method: str = eqx.field(static=True, default="upwind1")

    def vector_field(
        self, t: float, state: PyTree, args: PyTree | None = None
    ) -> Burgers2DState:
        r"""Compute tendency: advection + diffusion for (u, v)."""
        # Transport velocities at faces
        u_on_U = self.interp.T_to_U(state.u)
        v_on_V = self.interp.T_to_V(state.v)
        # Advection
        adv_u = self.advection(state.u, u_on_U, v_on_V, method=self.method)
        adv_v = self.advection(state.v, u_on_U, v_on_V, method=self.method)
        # Diffusion
        diff_u = self.params.nu * self.diff.laplacian(state.u)
        diff_v = self.params.nu * self.diff.laplacian(state.v)
        return Burgers2DState(u=adv_u + diff_u, v=adv_v + diff_v)

    def apply_boundary_conditions(self, state: PyTree) -> Burgers2DState:
        """Apply periodic boundary conditions."""
        return Burgers2DState(
            u=enforce_periodic(state.u),
            v=enforce_periodic(state.v),
        )

    def diagnose(self, state: PyTree) -> Burgers2DDiagnostics:
        """Compute kinetic energy diagnostic."""
        ui, vi = state.u[1:-1, 1:-1], state.v[1:-1, 1:-1]
        ke = 0.5 * jnp.sum(ui**2 + vi**2) * self.grid.dx * self.grid.dy
        return Burgers2DDiagnostics(kinetic_energy=ke)

    @staticmethod
    def create(
        nx: int = 64,
        ny: int = 64,
        Lx: float = 2.0,
        Ly: float = 2.0,
        nu: float = 0.01,
        method: str = "upwind1",
    ) -> Burgers2D:
        """Convenience factory.

        Args:
            nx: Number of interior cells in x.
            ny: Number of interior cells in y.
            Lx: Domain length in x.
            Ly: Domain length in y.
            nu: Kinematic viscosity.
            method: Advection reconstruction method.

        Returns:
            A ``Burgers2D`` model instance.
        """
        grid = ArakawaCGrid2D.from_interior(nx, ny, Lx, Ly)
        params = Burgers2DParams(nu=jnp.array(nu))
        diff = Difference2D(grid=grid)
        advection = FVXAdvection2D(grid=grid)
        interp = Interpolation2D(grid=grid)
        return Burgers2D(
            params=params,
            grid=grid,
            diff=diff,
            advection=advection,
            interp=interp,
            method=method,
        )
