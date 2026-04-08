"""2D incompressible Navier-Stokes in vorticity-streamfunction form."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from finitevolx import (
    Advection2D as FVXAdvection2D,
    ArakawaCGrid2D,
    Difference2D,
    Interpolation2D,
    enforce_periodic,
    streamfunction_from_vorticity,
)
from jaxtyping import Array, Float, PyTree

from somax._src.core.model import SomaxModel
from somax._src.core.types import Diagnostics, Params, State


class NSVorticityState(State):
    """State for the vorticity-streamfunction NS formulation.

    Args:
        omega: Vorticity field on T-points, shape ``(Ny, Nx)``.
    """

    omega: Array


class NSParams(Params):
    """Differentiable parameters for incompressible Navier-Stokes.

    Args:
        nu: Kinematic viscosity.
    """

    nu: Array


class NSDiagnostics(Diagnostics):
    """Diagnostics for incompressible Navier-Stokes.

    Args:
        psi: Streamfunction.
        u: x-velocity.
        v: y-velocity.
        kinetic_energy: Domain-averaged kinetic energy.
        enstrophy: Domain-averaged enstrophy.
    """

    psi: Float[Array, "Ny Nx"]
    u: Float[Array, "Ny Nx"]
    v: Float[Array, "Ny Nx"]
    kinetic_energy: Array
    enstrophy: Array


# -----------------------------------------------------------------------
# Ghia et al. (1982) benchmark data for lid-driven cavity at Re = 100
# u-velocity along vertical centreline (y, u) and
# v-velocity along horizontal centreline (x, v).
# -----------------------------------------------------------------------
GHIA_RE100_Y = jnp.array(
    [
        0.0000,
        0.0547,
        0.0625,
        0.0703,
        0.1016,
        0.1719,
        0.2813,
        0.4531,
        0.5000,
        0.6172,
        0.7344,
        0.8516,
        0.9531,
        0.9609,
        0.9688,
        0.9766,
        1.0000,
    ]
)
GHIA_RE100_U = jnp.array(
    [
        0.00000,
        -0.03717,
        -0.04192,
        -0.04775,
        -0.06434,
        -0.10150,
        -0.15662,
        -0.21090,
        -0.20581,
        -0.13641,
        0.00332,
        0.23151,
        0.68717,
        0.73722,
        0.78871,
        0.84123,
        1.00000,
    ]
)
GHIA_RE100_X = jnp.array(
    [
        0.0000,
        0.0625,
        0.0703,
        0.0781,
        0.0938,
        0.1563,
        0.2266,
        0.2344,
        0.5000,
        0.8047,
        0.8594,
        0.9063,
        0.9453,
        0.9531,
        0.9609,
        0.9688,
        1.0000,
    ]
)
GHIA_RE100_V = jnp.array(
    [
        0.00000,
        0.09233,
        0.10091,
        0.10890,
        0.12317,
        0.16077,
        0.17507,
        0.17527,
        0.05454,
        -0.24533,
        -0.22445,
        -0.16914,
        -0.10313,
        -0.08864,
        -0.07391,
        -0.05906,
        0.00000,
    ]
)


class IncompressibleNS2D(SomaxModel):
    r"""2D incompressible Navier-Stokes (vorticity-streamfunction).

    Solves the vorticity transport equation::

        d(omega)/dt + u * d(omega)/dx + v * d(omega)/dy = nu * laplacian(omega)

    where the velocity is recovered at each step via the Poisson
    inversion :math:`\nabla^2 \psi = -\omega` and
    :math:`u = \partial\psi/\partial y`,
    :math:`v = -\partial\psi/\partial x`.

    Supports two canonical benchmarks:

    - **Lid-driven cavity**: Dirichlet BCs (``poisson_bc="dst"``),
      no-slip walls with moving lid at the top.
    - **Channel flow**: Periodic in x, no-slip walls in y
      (``poisson_bc="dst"``), driven by a body force.

    Args:
        params: Differentiable parameters (viscosity ``nu``).
        grid: 2D Arakawa C-grid.
        diff: Difference operators.
        interp: Interpolation operators.
        advection: Advection operator.
        poisson_bc: Spectral solver BC type for Poisson inversion.
        u_lid: Lid velocity for cavity flow (default 1.0).
        body_force: Constant vorticity source for channel flow.
        method: Advection reconstruction method.
    """

    params: NSParams
    grid: ArakawaCGrid2D = eqx.field(static=True)
    diff: Difference2D = eqx.field(static=True)
    interp: Interpolation2D = eqx.field(static=True)
    advection: FVXAdvection2D = eqx.field(static=True)
    poisson_bc: str = eqx.field(static=True, default="dst")
    u_lid: float = eqx.field(static=True, default=1.0)
    body_force: float = eqx.field(static=True, default=0.0)
    method: str = eqx.field(static=True, default="upwind1")

    def _solve_psi(self, omega: Array) -> Float[Array, "Ny Nx"]:
        """Recover streamfunction from vorticity via Poisson solve."""
        return streamfunction_from_vorticity(
            -omega, self.grid.dx, self.grid.dy, bc=self.poisson_bc
        )

    def _velocity_from_psi(
        self, psi: Float[Array, "Ny Nx"]
    ) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
        """Compute velocity from streamfunction."""
        u = self.diff.diff_y_T_to_V(psi)  # u = dpsi/dy at V-points
        v = -self.diff.diff_x_T_to_U(psi)  # v = -dpsi/dx at U-points
        return u, v

    def vector_field(
        self, t: float, state: PyTree, args: PyTree | None = None
    ) -> NSVorticityState:
        """Compute vorticity tendency."""
        omega = state.omega
        # 1. Poisson inversion: nabla^2 psi = -omega
        psi = self._solve_psi(omega)
        # 2. Velocity from streamfunction
        u_V, v_U = self._velocity_from_psi(psi)
        # Interpolate to T-points for scalar advection
        u_T = self.interp.V_to_T(u_V)
        v_T = self.interp.U_to_T(v_U)
        # Interpolate to face points for transport
        u_on_U = self.interp.T_to_U(u_T)
        v_on_V = self.interp.T_to_V(v_T)
        # 3. Advection of vorticity
        adv = self.advection(omega, u_on_U, v_on_V, method=self.method)
        # 4. Diffusion
        diff = self.params.nu * self.diff.laplacian(omega)
        # 5. Body force (channel flow)
        domega_dt = adv + diff + self.body_force
        return NSVorticityState(omega=domega_dt)

    def apply_boundary_conditions(self, state: PyTree) -> NSVorticityState:
        """Apply vorticity boundary conditions.

        For cavity flow (``poisson_bc="dst"``): Thom's formula at walls.
        For channel flow (``poisson_bc="fft"``): periodic in x, walls in y.
        """
        omega = state.omega
        if self.poisson_bc == "dst":
            # Lid-driven cavity: all walls are no-slip
            # Solve for psi to compute wall vorticity (Thom's formula)
            psi = self._solve_psi(omega)
            dx2 = self.grid.dx**2
            dy2 = self.grid.dy**2
            # Bottom wall (y=0): omega = -2*psi[1]/dy^2
            omega = omega.at[0, :].set(-2.0 * psi[1, :] / dy2)
            # Top wall (y=Ly): omega = -2*psi[-2]/dy^2 - 2*u_lid/dy
            omega = omega.at[-1, :].set(
                -2.0 * psi[-2, :] / dy2 - 2.0 * self.u_lid / self.grid.dy
            )
            # Left wall (x=0): omega = -2*psi[:,1]/dx^2
            omega = omega.at[:, 0].set(-2.0 * psi[:, 1] / dx2)
            # Right wall (x=Lx): omega = -2*psi[:,-2]/dx^2
            omega = omega.at[:, -1].set(-2.0 * psi[:, -2] / dx2)
        else:
            # Periodic in x, walls in y
            omega = enforce_periodic(omega)
            # No-slip walls in y via Thom's formula
            psi = self._solve_psi(omega)
            dy2 = self.grid.dy**2
            omega = omega.at[0, :].set(-2.0 * psi[1, :] / dy2)
            omega = omega.at[-1, :].set(-2.0 * psi[-2, :] / dy2)
        return NSVorticityState(omega=omega)

    def diagnose(self, state: PyTree) -> NSDiagnostics:
        """Compute velocity, streamfunction, KE, and enstrophy."""
        psi = self._solve_psi(state.omega)
        u_V, v_U = self._velocity_from_psi(psi)
        u_T = self.interp.V_to_T(u_V)
        v_T = self.interp.U_to_T(v_U)
        interior_u = u_T[1:-1, 1:-1]
        interior_v = v_T[1:-1, 1:-1]
        interior_w = state.omega[1:-1, 1:-1]
        self.grid.dx * self.grid.dy
        ke = 0.5 * jnp.mean(interior_u**2 + interior_v**2)
        enstrophy = 0.5 * jnp.mean(interior_w**2)
        return NSDiagnostics(
            psi=psi,
            u=u_T,
            v=v_T,
            kinetic_energy=ke,
            enstrophy=enstrophy,
        )

    @staticmethod
    def create(
        nx: int = 64,
        ny: int = 64,
        Lx: float = 1.0,
        Ly: float = 1.0,
        nu: float = 0.01,
        problem: str = "cavity",
        u_lid: float = 1.0,
        body_force: float = 0.0,
        method: str = "upwind1",
    ) -> IncompressibleNS2D:
        """Convenience factory.

        Args:
            nx: Number of interior cells in x.
            ny: Number of interior cells in y.
            Lx: Domain length in x.
            Ly: Domain length in y.
            nu: Kinematic viscosity.
            problem: ``"cavity"`` (Dirichlet) or ``"channel"`` (periodic x).
            u_lid: Lid velocity for cavity flow.
            body_force: Constant vorticity source for channel flow.
            method: Advection reconstruction method.

        Returns:
            An ``IncompressibleNS2D`` model instance.
        """
        grid = ArakawaCGrid2D.from_interior(nx, ny, Lx, Ly)
        params = NSParams(nu=jnp.array(nu))
        diff = Difference2D(grid=grid)
        interp = Interpolation2D(grid=grid)
        advection = FVXAdvection2D(grid=grid)
        # Both cavity and channel use DST (Dirichlet psi=0 at walls).
        # Channel periodicity in x is enforced via vorticity BCs.
        poisson_bc = "dst"
        return IncompressibleNS2D(
            params=params,
            grid=grid,
            diff=diff,
            interp=interp,
            advection=advection,
            poisson_bc=poisson_bc,
            u_lid=u_lid,
            body_force=body_force,
            method=method,
        )
