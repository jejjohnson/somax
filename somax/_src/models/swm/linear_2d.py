"""2D linear shallow water model."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from finitevolx import (
    ArakawaCGrid2D,
    Coriolis2D,
    Difference2D,
    Interpolation2D,
    coriolis_fn,
    enforce_periodic,
)
from jaxtyping import Array, Float, PyTree

from somax._src.core.model import SomaxModel
from somax._src.core.types import Diagnostics, Params, PhysConsts, State


class LinearSW2DState(State):
    """State for the 2D linear shallow water model.

    Args:
        h: Height perturbation (η = h - H₀), shape ``(Ny, Nx)``.
        u: x-velocity on U-points, shape ``(Ny, Nx)``.
        v: y-velocity on V-points, shape ``(Ny, Nx)``.
    """

    h: Float[Array, "Ny Nx"]
    u: Float[Array, "Ny Nx"]
    v: Float[Array, "Ny Nx"]


class LinearSW2DParams(Params):
    """Differentiable parameters for the 2D linear shallow water model.

    Args:
        lateral_viscosity: Harmonic viscosity coefficient nu.
        bottom_drag: Linear bottom drag coefficient kappa.
    """

    lateral_viscosity: Array
    bottom_drag: Array


class LinearSW2DPhysConsts(PhysConsts):
    """Frozen physical constants for the 2D linear shallow water model.

    Args:
        gravity: Gravitational acceleration g.
        f0: Reference Coriolis parameter.
        beta: Meridional gradient of f (β-plane).
        H0: Mean layer depth.
    """

    gravity: float = eqx.field(static=True, default=9.81)
    f0: float = eqx.field(static=True, default=1e-4)
    beta: float = eqx.field(static=True, default=0.0)
    H0: float = eqx.field(static=True, default=100.0)


class LinearSW2DDiagnostics(Diagnostics):
    """Diagnostics for the 2D linear shallow water model.

    Args:
        energy: Domain-integrated energy.
        relative_vorticity: ζ = dv/dx - du/dy at X-points.
    """

    energy: Array
    relative_vorticity: Float[Array, "Ny Nx"]


class LinearShallowWater2D(SomaxModel):
    r"""2D linear shallow water model on an Arakawa C-grid.

    Solves the linearised shallow water equations::

        dh/dt = -H₀ · (du/dx + dv/dy)
        du/dt = -g · dh/dx + f·v + nu*laplacian(u) - kappa*u
        dv/dt = -g · dh/dy - f·u + nu*laplacian(v) - kappa*v

    Supports both f-plane (β=0) and β-plane Coriolis.

    Args:
        params: Differentiable parameters (viscosity, drag).
        consts: Frozen physical constants (g, f₀, β, H₀).
        grid: 2D Arakawa C-grid.
        diff: Difference operators.
        interp: Interpolation operators.
        coriolis: Coriolis operator.
        f_field: Precomputed Coriolis parameter field f(y).
        bc_type: Boundary condition type (``"periodic"`` or ``"wall"``).
    """

    params: LinearSW2DParams
    consts: LinearSW2DPhysConsts = eqx.field(static=True)
    grid: ArakawaCGrid2D = eqx.field(static=True)
    diff: Difference2D = eqx.field(static=True)
    interp: Interpolation2D = eqx.field(static=True)
    coriolis: Coriolis2D = eqx.field(static=True)
    f_field: Float[Array, "Ny Nx"]
    bc_type: str = eqx.field(static=True, default="periodic")

    def vector_field(
        self, t: float, state: PyTree, args: PyTree | None = None
    ) -> LinearSW2DState:
        """Compute tendencies for the linear shallow water equations."""
        h, u, v = state.h, state.u, state.v
        g = self.consts.gravity
        H0 = self.consts.H0
        nu = self.params.lateral_viscosity
        kappa = self.params.bottom_drag

        # Mass equation: dh/dt = -H0 * div(u, v)
        dh_dt = -H0 * self.diff.divergence(u, v)

        # Pressure gradient
        du_dt = -g * self.diff.diff_x_T_to_U(h)
        dv_dt = -g * self.diff.diff_y_T_to_V(h)

        # Coriolis: du += f*v, dv -= f*u
        du_cor, dv_cor = self.coriolis(u, v, self.f_field)
        du_dt = du_dt + du_cor
        dv_dt = dv_dt + dv_cor

        # Diffusion
        du_dt = du_dt + nu * self.diff.laplacian(u)
        dv_dt = dv_dt + nu * self.diff.laplacian(v)

        # Bottom drag
        du_dt = du_dt - kappa * u
        dv_dt = dv_dt - kappa * v

        return LinearSW2DState(h=dh_dt, u=du_dt, v=dv_dt)

    def apply_boundary_conditions(self, state: PyTree) -> LinearSW2DState:
        """Apply boundary conditions (periodic or free-slip wall)."""
        if self.bc_type == "periodic":
            h = enforce_periodic(state.h)
            u = enforce_periodic(state.u)
            v = enforce_periodic(state.v)
        else:
            h, u, v = state.h, state.u, state.v
            # No normal flow: wall faces AND ghost faces
            u = u.at[:, 0].set(0.0).at[:, -2].set(0.0).at[:, -1].set(0.0)
            v = v.at[0, :].set(0.0).at[-2, :].set(0.0).at[-1, :].set(0.0)
            # Free-slip: tangential velocity ghost = adjacent interior
            u = u.at[0, :].set(u[1, :]).at[-1, :].set(u[-2, :])
            v = v.at[:, 0].set(v[:, 1]).at[:, -1].set(v[:, -2])
            # Height: zero-gradient at ghost cells
            h = h.at[0, :].set(h[1, :]).at[-1, :].set(h[-2, :])
            h = h.at[:, 0].set(h[:, 1]).at[:, -1].set(h[:, -2])
        return LinearSW2DState(h=h, u=u, v=v)

    def diagnose(self, state: PyTree) -> LinearSW2DDiagnostics:
        """Compute energy and relative vorticity."""
        g = self.consts.gravity
        H0 = self.consts.H0
        h, u, v = state.h, state.u, state.v
        s = (slice(1, -1), slice(1, -1))
        energy = 0.5 * jnp.sum(g * h[s] ** 2 + H0 * (u[s] ** 2 + v[s] ** 2))
        zeta = self.diff.curl(u, v)
        return LinearSW2DDiagnostics(energy=energy, relative_vorticity=zeta)

    @staticmethod
    def create(
        nx: int = 64,
        ny: int = 64,
        Lx: float = 1e6,
        Ly: float = 1e6,
        g: float = 9.81,
        f0: float = 1e-4,
        beta: float = 0.0,
        H0: float = 100.0,
        lateral_viscosity: float = 0.0,
        bottom_drag: float = 0.0,
        bc: str = "periodic",
    ) -> LinearShallowWater2D:
        """Convenience factory.

        Args:
            nx: Number of interior cells in x.
            ny: Number of interior cells in y.
            Lx: Domain length in x (m).
            Ly: Domain length in y (m).
            g: Gravitational acceleration (m/s²).
            f0: Reference Coriolis parameter (1/s).
            beta: Meridional gradient of f (1/(m·s)).
            H0: Mean layer depth (m).
            lateral_viscosity: Harmonic viscosity (m²/s).
            bottom_drag: Linear bottom drag (1/s).
            bc: Boundary condition type (``"periodic"`` or ``"wall"``).

        Returns:
            A ``LinearShallowWater2D`` model instance.
        """
        grid = ArakawaCGrid2D.from_interior(nx, ny, Lx, Ly)
        params = LinearSW2DParams(
            lateral_viscosity=jnp.array(lateral_viscosity),
            bottom_drag=jnp.array(bottom_drag),
        )
        consts = LinearSW2DPhysConsts(gravity=g, f0=f0, beta=beta, H0=H0)
        diff = Difference2D(grid=grid)
        interp = Interpolation2D(grid=grid)
        coriolis = Coriolis2D(grid=grid)

        # Build f(y) = f0 + beta * (y - y0) on full grid
        y = jnp.arange(grid.Ny) * grid.dy
        y0 = Ly / 2.0
        Y = jnp.broadcast_to(y[:, None], (grid.Ny, grid.Nx))
        f_field = coriolis_fn(Y, f0=f0, beta=beta, y0=y0)

        return LinearShallowWater2D(
            params=params,
            consts=consts,
            grid=grid,
            diff=diff,
            interp=interp,
            coriolis=coriolis,
            f_field=f_field,
            bc_type=bc,
        )
