"""2D nonlinear convection: du/dt + u*grad(u) = 0."""

from __future__ import annotations

from typing import Any, ClassVar

import equinox as eqx
import jax.numpy as jnp
from finitevolx import (
    Advection2D as FVXAdvection2D,
    CartesianGrid2D,
    Interpolation2D,
    Mask2D,
    enforce_periodic,
)
from jaxtyping import Array, PyTree

from somax._src.core.model import SomaxModel
from somax._src.core.scales import Scales
from somax._src.core.types import Diagnostics, State
from somax._src.models._nondim import require_positive


class NonlinearConvection2DState(State):
    """State for 2D nonlinear convection.

    Args:
        u: x-velocity on T-points, shape ``(Ny, Nx)``.
        v: y-velocity on T-points, shape ``(Ny, Nx)``.
    """

    u: Array
    v: Array

    # Both components are collocated at T-points, not staggered.
    mask_locations: ClassVar[dict[str, str]] = {"u": "h", "v": "h"}


class NonlinearConvection2DDiagnostics(Diagnostics):
    """Diagnostics for 2D nonlinear convection.

    Args:
        kinetic_energy: Integrated KE = 0.5 * (u^2 + v^2).
    """

    kinetic_energy: Array


class NonlinearConvection2D(SomaxModel):
    """2D nonlinear convection on an Arakawa C-grid.

    Solves the system::

        du/dt + u * du/dx + v * du/dy = 0
        dv/dt + u * dv/dx + v * dv/dy = 0

    using upwind flux reconstruction.

    Args:
        grid: 2D Arakawa C-grid.
        advection: Advection operator.
        interp: Interpolation operators.
        mask: Optional Arakawa C-grid mask (``None`` = all-ocean).
        method: Reconstruction method (default ``"upwind1"``).
    """

    grid: CartesianGrid2D = eqx.field(static=True)
    advection: FVXAdvection2D
    interp: Interpolation2D
    mask: Mask2D | None
    method: str = eqx.field(static=True, default="upwind1")

    def vector_field(
        self, t: float, state: PyTree, args: PyTree | None = None
    ) -> NonlinearConvection2DState:
        """Compute tendency via upwind advection."""
        # Interpolate T-point velocities to U/V face points for transport
        u_on_U = self.interp.T_to_U(state.u)
        v_on_V = self.interp.T_to_V(state.v)
        # Advect each component
        du_dt = self.advection(state.u, u_on_U, v_on_V, method=self.method)
        dv_dt = self.advection(state.v, u_on_U, v_on_V, method=self.method)
        return NonlinearConvection2DState(u=du_dt, v=dv_dt)

    def apply_boundary_conditions(self, state: PyTree) -> NonlinearConvection2DState:
        """Apply periodic boundary conditions."""
        return NonlinearConvection2DState(
            u=enforce_periodic(state.u),
            v=enforce_periodic(state.v),
        )

    def diagnose(self, state: PyTree) -> NonlinearConvection2DDiagnostics:
        """Compute kinetic energy diagnostic."""
        ui, vi = state.u[1:-1, 1:-1], state.v[1:-1, 1:-1]
        ke = 0.5 * jnp.sum(ui**2 + vi**2) * self.grid.dx * self.grid.dy
        return NonlinearConvection2DDiagnostics(kinetic_energy=ke)

    @staticmethod
    def from_nondimensional(
        *,
        nx: int = 64,
        ny: int = 64,
        aspect: float = 1.0,
        **create_kw: Any,
    ) -> tuple[NonlinearConvection2D, Scales]:
        r"""Build the model at unit scales instead of SI coefficients.

        Non-dimensional form
        --------------------
        Advective scale set (:meth:`somax.Scales.advective`) with
        ``L = U = 1``, so ``T = 1`` and the equation reads
        ``d_t u + u . grad u = 0``. Inviscid nonlinear convection has no
        coefficient to scale out, so this factory only fixes the domain
        and reports the scale set; it exists so the family presents one
        interface.


        Args:
            nx: Interior cells in x.
            ny: Interior cells in y.
            aspect: ``Ly / Lx``; the domain is ``Lx = 1``.
            **create_kw: Forwarded to :meth:`create` (``method``, ``mask``).

        Returns:
            ``(model, scales)``. ``scales.dt_from_cfl(C, (nx, ny),
            extent=(1.0, aspect))`` gives a step in the same time unit;
            pass both cell counts and the aspect ratio, since the bound
            depends on the *smaller* spacing.
        """
        context = "NonlinearConvection2D.from_nondimensional"
        require_positive(context, aspect=aspect)
        model = NonlinearConvection2D.create(
            nx=nx,
            ny=ny,
            Lx=1.0,
            Ly=aspect,
            **create_kw,
        )
        return model, Scales.advective(L=1.0, U=1.0)

    @staticmethod
    def create(
        nx: int = 64,
        ny: int = 64,
        Lx: float = 2.0,
        Ly: float = 2.0,
        method: str = "upwind1",
        mask: Mask2D | None = None,
    ) -> NonlinearConvection2D:
        """Convenience factory.

        Args:
            nx: Number of interior cells in x.
            ny: Number of interior cells in y.
            Lx: Domain length in x.
            Ly: Domain length in y.
            method: Advection reconstruction method.
            mask: Optional Arakawa C-grid mask (``None`` = all-ocean).

        Returns:
            A ``NonlinearConvection2D`` model instance.
        """
        grid = CartesianGrid2D.from_interior(nx, ny, Lx, Ly)
        advection = FVXAdvection2D(grid=grid, mask=mask)
        interp = Interpolation2D(grid=grid, mask=mask)
        return NonlinearConvection2D(
            grid=grid, advection=advection, interp=interp, mask=mask, method=method
        )
