"""1D linear convection: du/dt + c du/dx = 0."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from finitevolx import ArakawaCGrid1D, Difference1D, Interpolation1D
from jaxtyping import Array, PyTree

from somax._src.core.model import SomaxModel
from somax._src.core.types import Diagnostics, Params, State


class LinearConvection1DParams(Params):
    """Differentiable parameters for 1D linear convection.

    Args:
        c: Wave speed.
    """

    c: Array


class LinearConvection1DState(State):
    """State for 1D linear convection.

    Args:
        u: Scalar field on T-points, shape ``(Nx,)`` including ghost cells.
    """

    u: Array


class LinearConvection1DDiagnostics(Diagnostics):
    """Diagnostics for 1D linear convection.

    Args:
        energy: Integrated energy 0.5 * sum(u^2).
    """

    energy: Array


class LinearConvection1D(SomaxModel):
    """1D linear convection equation on an Arakawa C-grid.

    Solves ``du/dt + c * du/dx = 0`` where ``c`` is a learnable wave
    speed visible to ``jax.grad``.

    Args:
        params: Differentiable parameters (wave speed ``c``).
        grid: 1D Arakawa C-grid.
        diff: Difference operators.
        interp: Interpolation operators.
        periodic: Whether to use periodic boundary conditions.
    """

    params: LinearConvection1DParams
    grid: ArakawaCGrid1D = eqx.field(static=True)
    diff: Difference1D = eqx.field(static=True)
    interp: Interpolation1D = eqx.field(static=True)
    periodic: bool = eqx.field(static=True, default=True)

    def vector_field(
        self, t: float, state: PyTree, args: PyTree | None = None
    ) -> LinearConvection1DState:
        """Compute tendency: du/dt = -c * du/dx."""
        # Flux form: -d(c * u)/dx at T-points
        # 1. Interpolate u to U-points, multiply by c
        flux = self.params.c * self.interp.T_to_U(state.u)
        # 2. Backward difference of flux: U -> T
        du_dt = -self.diff.diff_x_U_to_T(flux)
        return LinearConvection1DState(u=du_dt)

    def apply_boundary_conditions(self, state: PyTree) -> LinearConvection1DState:
        """Apply boundary conditions to the state."""
        u = state.u
        if self.periodic:
            u = u.at[0].set(u[-2])
            u = u.at[-1].set(u[1])
        else:
            # Dirichlet: zero at boundaries
            u = u.at[0].set(-u[1])
            u = u.at[-1].set(-u[-2])
        return LinearConvection1DState(u=u)

    def diagnose(self, state: PyTree) -> LinearConvection1DDiagnostics:
        """Compute energy diagnostic."""
        energy = 0.5 * jnp.sum(state.u[1:-1] ** 2) * self.grid.dx
        return LinearConvection1DDiagnostics(energy=energy)

    @staticmethod
    def create(
        nx: int = 100,
        Lx: float = 2.0,
        c: float = 1.0,
        periodic: bool = True,
    ) -> LinearConvection1D:
        """Convenience factory.

        Args:
            nx: Number of interior grid cells.
            Lx: Domain length.
            c: Wave speed.
            periodic: Use periodic BCs if True, Dirichlet if False.

        Returns:
            A ``LinearConvection1D`` model instance.
        """
        grid = ArakawaCGrid1D.from_interior(nx, Lx)
        params = LinearConvection1DParams(c=jnp.array(c))
        diff = Difference1D(grid=grid)
        interp = Interpolation1D(grid=grid)
        return LinearConvection1D(
            params=params,
            grid=grid,
            diff=diff,
            interp=interp,
            periodic=periodic,
        )
