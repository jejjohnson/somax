"""1D nonlinear shallow water model."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from finitevolx import Advection1D, ArakawaCGrid1D, Difference1D, Interpolation1D
from jaxtyping import Array, PyTree

from somax._src.core.model import SomaxModel
from somax._src.core.types import Diagnostics, Params, PhysConsts, State


class NonlinearSW1DState(State):
    """State for the 1D nonlinear shallow water model.

    Args:
        h: Total layer thickness, shape ``(Nx,)``.
        u: x-velocity on U-points, shape ``(Nx,)``.
        v: y-velocity on T-points (Coriolis coupling), shape ``(Nx,)``.
    """

    h: Array
    u: Array
    v: Array


class NonlinearSW1DParams(Params):
    """Differentiable parameters for the 1D nonlinear shallow water model.

    Args:
        lateral_viscosity: Harmonic viscosity coefficient nu.
        bottom_drag: Linear bottom drag coefficient kappa.
    """

    lateral_viscosity: Array
    bottom_drag: Array


class NonlinearSW1DPhysConsts(PhysConsts):
    """Frozen physical constants for the 1D nonlinear shallow water model.

    Args:
        gravity: Gravitational acceleration g.
        f0: Coriolis parameter (f-plane).
        H0: Mean layer depth (for diagnostics only).
    """

    gravity: float = eqx.field(static=True, default=9.81)
    f0: float = eqx.field(static=True, default=1e-4)
    H0: float = eqx.field(static=True, default=100.0)


class NonlinearSW1DDiagnostics(Diagnostics):
    """Diagnostics for the 1D nonlinear shallow water model.

    Args:
        energy: Domain-integrated energy.
        mass: Domain-integrated mass (h).
    """

    energy: Array
    mass: Array


class NonlinearShallowWater1D(SomaxModel):
    r"""1D nonlinear shallow water model on an Arakawa C-grid.

    Solves the nonlinear shallow water equations::

        dh/dt = -d(h·u)/dx
        du/dt = -u·du/dx - g·dh/dx + f₀·v + nu*laplacian(u) - kappa*u
        dv/dt = -f₀·u                      + nu*laplacian(v) - kappa*v

    Args:
        params: Differentiable parameters (viscosity, drag).
        consts: Frozen physical constants (g, f₀, H₀).
        grid: 1D Arakawa C-grid.
        diff: Difference operators.
        interp: Interpolation operators.
        advection: Advection operator.
        method: Advection reconstruction method.
    """

    params: NonlinearSW1DParams
    consts: NonlinearSW1DPhysConsts = eqx.field(static=True)
    grid: ArakawaCGrid1D = eqx.field(static=True)
    diff: Difference1D = eqx.field(static=True)
    interp: Interpolation1D = eqx.field(static=True)
    advection: Advection1D = eqx.field(static=True)
    method: str = eqx.field(static=True, default="upwind1")

    def vector_field(
        self, t: float, state: PyTree, args: PyTree | None = None
    ) -> NonlinearSW1DState:
        """Compute tendencies for the nonlinear shallow water equations."""
        h, u, v = state.h, state.u, state.v
        g = self.consts.gravity
        f0 = self.consts.f0
        nu = self.params.lateral_viscosity
        kappa = self.params.bottom_drag

        # Mass equation: dh/dt = -d(hu)/dx (via advection operator)
        dh_dt = self.advection(h, u, method=self.method)

        # Momentum: -u * du/dx (advection of u by u)
        du_adv = self.advection(u, u, method=self.method)

        # Pressure gradient: -g * dh/dx
        pressure_grad = -g * self.diff.diff_x_T_to_U(h)

        # Coriolis
        v_on_U = self.interp.T_to_U(v)
        u_on_T = self.interp.U_to_T(u)

        du_dt = du_adv + pressure_grad + f0 * v_on_U
        dv_dt = -f0 * u_on_T

        # Diffusion
        du_dt = du_dt + nu * self.diff.laplacian(u)
        dv_dt = dv_dt + nu * self.diff.laplacian(v)

        # Bottom drag
        du_dt = du_dt - kappa * u
        dv_dt = dv_dt - kappa * v

        return NonlinearSW1DState(h=dh_dt, u=du_dt, v=dv_dt)

    def apply_boundary_conditions(self, state: PyTree) -> NonlinearSW1DState:
        """Apply periodic boundary conditions."""
        h = state.h.at[0].set(state.h[-2]).at[-1].set(state.h[1])
        u = state.u.at[0].set(state.u[-2]).at[-1].set(state.u[1])
        v = state.v.at[0].set(state.v[-2]).at[-1].set(state.v[1])
        return NonlinearSW1DState(h=h, u=u, v=v)

    def diagnose(self, state: PyTree) -> NonlinearSW1DDiagnostics:
        """Compute energy and mass."""
        g = self.consts.gravity
        h, u, v = state.h, state.u, state.v
        interior = slice(1, -1)
        energy = 0.5 * jnp.sum(
            g * h[interior] ** 2 + h[interior] * (u[interior] ** 2 + v[interior] ** 2)
        )
        mass = jnp.sum(h[interior])
        return NonlinearSW1DDiagnostics(energy=energy, mass=mass)

    @staticmethod
    def create(
        nx: int = 200,
        Lx: float = 1e6,
        g: float = 9.81,
        f0: float = 1e-4,
        H0: float = 100.0,
        lateral_viscosity: float = 0.0,
        bottom_drag: float = 0.0,
        method: str = "upwind1",
    ) -> NonlinearShallowWater1D:
        """Convenience factory.

        Args:
            nx: Number of interior cells.
            Lx: Domain length (m).
            g: Gravitational acceleration (m/s²).
            f0: Coriolis parameter (1/s).
            H0: Mean layer depth (m).
            lateral_viscosity: Harmonic viscosity (m²/s).
            bottom_drag: Linear bottom drag (1/s).
            method: Advection reconstruction method.

        Returns:
            A ``NonlinearShallowWater1D`` model instance.
        """
        grid = ArakawaCGrid1D.from_interior(nx, Lx)
        params = NonlinearSW1DParams(
            lateral_viscosity=jnp.array(lateral_viscosity),
            bottom_drag=jnp.array(bottom_drag),
        )
        consts = NonlinearSW1DPhysConsts(gravity=g, f0=f0, H0=H0)
        diff = Difference1D(grid=grid)
        interp = Interpolation1D(grid=grid)
        advection = Advection1D(grid=grid)
        return NonlinearShallowWater1D(
            params=params,
            consts=consts,
            grid=grid,
            diff=diff,
            interp=interp,
            advection=advection,
            method=method,
        )
