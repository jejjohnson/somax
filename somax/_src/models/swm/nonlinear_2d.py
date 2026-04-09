"""2D nonlinear shallow water model (vector-invariant form)."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from finitevolx import (
    Advection2D as FVXAdvection2D,
    ArakawaCGrid2D,
    Coriolis2D,
    Difference2D,
    Diffusion2D as FVXDiffusion2D,
    Interpolation2D,
    Vorticity2D,
    coriolis_fn,
    enforce_periodic,
    kinetic_energy,
)
from jaxtyping import Array, Float, PyTree

from somax._src.core.model import SomaxModel
from somax._src.core.types import Diagnostics, Params, PhysConsts, State


class NonlinearSW2DState(State):
    """State for the 2D nonlinear shallow water model.

    Args:
        h: Total layer thickness, shape ``(Ny, Nx)``.
        u: x-velocity on U-points, shape ``(Ny, Nx)``.
        v: y-velocity on V-points, shape ``(Ny, Nx)``.
    """

    h: Float[Array, "Ny Nx"]
    u: Float[Array, "Ny Nx"]
    v: Float[Array, "Ny Nx"]


class NonlinearSW2DParams(Params):
    """Differentiable parameters for the 2D nonlinear shallow water model.

    Args:
        lateral_viscosity: Harmonic viscosity coefficient nu.
        bottom_drag: Linear bottom drag coefficient kappa.
        wind_amplitude: Wind stress amplitude (m/s^2).
    """

    lateral_viscosity: Array
    bottom_drag: Array
    wind_amplitude: Array


class NonlinearSW2DPhysConsts(PhysConsts):
    """Frozen physical constants for the 2D nonlinear shallow water model.

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


class NonlinearSW2DDiagnostics(Diagnostics):
    """Diagnostics for the 2D nonlinear shallow water model.

    Args:
        energy: Domain-integrated total energy.
        enstrophy: Domain-integrated potential enstrophy.
        potential_vorticity: PV field q = (ζ+f)/h at X-points.
        relative_vorticity: ζ = dv/dx - du/dy at X-points.
        kinetic_energy_field: KE at T-points.
    """

    energy: Array
    enstrophy: Array
    potential_vorticity: Float[Array, "Ny Nx"]
    relative_vorticity: Float[Array, "Ny Nx"]
    kinetic_energy_field: Float[Array, "Ny Nx"]


class NonlinearShallowWater2D(SomaxModel):
    r"""2D nonlinear shallow water model (vector-invariant form).

    Solves the rotating shallow water equations in vector-invariant
    form on an Arakawa C-grid::

        dh/dt = -div(h*u)
        du/dt = +q·h̄v - ∂P/∂x + nu*laplacian(u) - kappa*u
        dv/dt = -q·h̄u - ∂P/∂y + nu*laplacian(v) - kappa*v

    where q = (ζ+f)/h is potential vorticity and P = KE + g·h
    is the Bernoulli potential.

    Args:
        params: Differentiable parameters.
        consts: Frozen physical constants.
        grid: 2D Arakawa C-grid.
        diff: Difference operators.
        interp: Interpolation operators.
        coriolis: Coriolis operator.
        vorticity: Vorticity/PV operator.
        advection: Scalar advection operator (for mass).
        diffusion: Diffusion operator.
        f_field: Precomputed Coriolis field f(y).
        wind_stress_x: Precomputed x-wind stress pattern (normalised).
        wind_stress_y: Precomputed y-wind stress pattern (normalised).
        bc_type: Boundary condition type.
        method: Advection reconstruction method for mass equation.
    """

    params: NonlinearSW2DParams
    consts: NonlinearSW2DPhysConsts = eqx.field(static=True)
    grid: ArakawaCGrid2D = eqx.field(static=True)
    diff: Difference2D = eqx.field(static=True)
    interp: Interpolation2D = eqx.field(static=True)
    coriolis: Coriolis2D = eqx.field(static=True)
    vorticity: Vorticity2D = eqx.field(static=True)
    advection: FVXAdvection2D = eqx.field(static=True)
    diffusion: FVXDiffusion2D = eqx.field(static=True)
    f_field: Float[Array, "Ny Nx"]
    wind_stress_x: Float[Array, "Ny Nx"]
    wind_stress_y: Float[Array, "Ny Nx"]
    bc_type: str = eqx.field(static=True, default="periodic")
    method: str = eqx.field(static=True, default="upwind1")

    def vector_field(
        self, t: float, state: PyTree, args: PyTree | None = None
    ) -> NonlinearSW2DState:
        """Compute tendencies for the nonlinear shallow water equations."""
        h, u, v = state.h, state.u, state.v
        g = self.consts.gravity
        nu = self.params.lateral_viscosity
        kappa = self.params.bottom_drag
        tau0 = self.params.wind_amplitude

        # 1. Mass equation: dh/dt = -div(hu, hv)
        dh_dt = self.advection(h, u, v, method=self.method)

        # 2. Potential vorticity: q = (zeta + f) / h at X-points
        q = self.vorticity.potential_vorticity(u, v, h, self.f_field)

        # 3. Cross-velocity PV fluxes (vector-invariant form):
        #    du/dt = +q_bar * (hv)_bar   (v drives u via Coriolis)
        #    dv/dt = -q_bar * (hu)_bar   (u drives v via Coriolis)
        h_on_U = self.interp.T_to_U(h)
        h_on_V = self.interp.T_to_V(h)
        uh = h_on_U * u  # mass flux at U-points
        vh = h_on_V * v  # mass flux at V-points

        q_on_U = self.interp.X_to_U(q)
        q_on_V = self.interp.X_to_V(q)
        vh_on_U = self.interp.V_to_U(vh)
        uh_on_V = self.interp.U_to_V(uh)

        # 4. Bernoulli potential: P = KE + g*h
        ke = kinetic_energy(u, v)
        P = ke + g * h

        # 5. Momentum tendencies
        du_dt = q_on_U * vh_on_U - self.diff.diff_x_T_to_U(P)
        dv_dt = -q_on_V * uh_on_V - self.diff.diff_y_T_to_V(P)

        # 6. Wind forcing
        du_dt = du_dt + tau0 * self.wind_stress_x
        dv_dt = dv_dt + tau0 * self.wind_stress_y

        # 7. Diffusion
        du_dt = du_dt + self.diffusion(u, nu)
        dv_dt = dv_dt + self.diffusion(v, nu)

        # 8. Bottom drag
        du_dt = du_dt - kappa * u
        dv_dt = dv_dt - kappa * v

        return NonlinearSW2DState(h=dh_dt, u=du_dt, v=dv_dt)

    def apply_boundary_conditions(self, state: PyTree) -> NonlinearSW2DState:
        """Apply boundary conditions to all state fields.

        For periodic: ghost cells wrap around.
        For wall (free-slip, no-normal-flow):
          - u = 0 at x-boundaries (no normal flow at east/west walls)
          - v = 0 at y-boundaries (no normal flow at north/south walls)
          - Tangential velocity: zero-gradient (free-slip)
          - h: zero-gradient at all walls
        """
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
        return NonlinearSW2DState(h=h, u=u, v=v)

    def diagnose(self, state: PyTree) -> NonlinearSW2DDiagnostics:
        """Compute energy, enstrophy, PV, vorticity, and KE fields."""
        g = self.consts.gravity
        h, u, v = state.h, state.u, state.v
        s = (slice(1, -1), slice(1, -1))

        ke = kinetic_energy(u, v)
        zeta = self.vorticity.relative_vorticity(u, v)
        q = self.vorticity.potential_vorticity(u, v, h, self.f_field)

        # Total energy: KE + PE
        energy = jnp.sum(ke[s] + 0.5 * g * h[s] ** 2)
        # Potential enstrophy: 0.5 * q^2 * h
        h_on_X = self.interp.T_to_X(h)
        enstrophy = 0.5 * jnp.sum(q[s] ** 2 * h_on_X[s])

        return NonlinearSW2DDiagnostics(
            energy=energy,
            enstrophy=enstrophy,
            potential_vorticity=q,
            relative_vorticity=zeta,
            kinetic_energy_field=ke,
        )

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
        wind_amplitude: float = 0.0,
        wind_profile: str = "doublegyre",
        bc: str = "periodic",
        method: str = "upwind1",
    ) -> NonlinearShallowWater2D:
        """Convenience factory.

        Args:
            nx: Number of interior cells in x.
            ny: Number of interior cells in y.
            Lx: Domain length in x (m).
            Ly: Domain length in y (m).
            g: Gravitational acceleration (m/s^2).
            f0: Reference Coriolis parameter (1/s).
            beta: Meridional gradient of f (1/(m*s)).
            H0: Mean layer depth (m).
            lateral_viscosity: Harmonic viscosity (m^2/s).
            bottom_drag: Linear bottom drag (1/s).
            wind_amplitude: Wind stress amplitude (m/s^2).
            wind_profile: Wind stress pattern. ``"doublegyre"``
                gives tau_x = -cos(2*pi*y/Ly), ``"single"`` gives
                tau_x = -cos(pi*y/Ly).
            bc: Boundary condition type.
            method: Advection reconstruction method for mass.

        Returns:
            A ``NonlinearShallowWater2D`` model instance.
        """
        grid = ArakawaCGrid2D.from_interior(nx, ny, Lx, Ly)
        params = NonlinearSW2DParams(
            lateral_viscosity=jnp.array(lateral_viscosity),
            bottom_drag=jnp.array(bottom_drag),
            wind_amplitude=jnp.array(wind_amplitude),
        )
        consts = NonlinearSW2DPhysConsts(gravity=g, f0=f0, beta=beta, H0=H0)
        diff = Difference2D(grid=grid)
        interp = Interpolation2D(grid=grid)
        coriolis = Coriolis2D(grid=grid)
        vorticity_op = Vorticity2D(grid=grid)
        advection = FVXAdvection2D(grid=grid)
        diffusion = FVXDiffusion2D(grid=grid)

        y = jnp.arange(grid.Ny) * grid.dy
        y0 = Ly / 2.0
        Y = jnp.broadcast_to(y[:, None], (grid.Ny, grid.Nx))
        f_field = coriolis_fn(Y, f0=f0, beta=beta, y0=y0)

        # Wind stress pattern (normalised)
        if wind_profile == "single":
            wind_stress_x = -jnp.cos(jnp.pi * Y / Ly)
        else:
            # Double gyre: tau_x = -cos(2*pi*y/Ly)
            wind_stress_x = -jnp.cos(2.0 * jnp.pi * Y / Ly)
        wind_stress_y = jnp.zeros_like(wind_stress_x)

        return NonlinearShallowWater2D(
            params=params,
            consts=consts,
            grid=grid,
            diff=diff,
            interp=interp,
            coriolis=coriolis,
            vorticity=vorticity_op,
            advection=advection,
            diffusion=diffusion,
            f_field=f_field,
            wind_stress_x=wind_stress_x,
            wind_stress_y=wind_stress_y,
            bc_type=bc,
            method=method,
        )
