"""Barotropic quasi-geostrophic model."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from finitevolx import (
    ArakawaCGrid2D,
    Difference2D,
    Interpolation2D,
    arakawa_jacobian,
    streamfunction_from_vorticity,
    zero_boundaries,
)
from jaxtyping import Array, Float, PyTree

from somax._src.core.model import SomaxModel
from somax._src.core.types import Diagnostics, Params, PhysConsts, State


class BarotropicQGState(State):
    """State for the barotropic quasi-geostrophic model.

    Args:
        q: Potential vorticity on T-points, shape ``(Ny, Nx)``.
    """

    q: Float[Array, "Ny Nx"]


class BarotropicQGParams(Params):
    """Differentiable parameters for the barotropic QG model.

    Args:
        lateral_viscosity: Harmonic viscosity coefficient nu.
        bottom_drag: Linear bottom drag coefficient kappa.
        wind_amplitude: Wind forcing amplitude tau0.
    """

    lateral_viscosity: Array
    bottom_drag: Array
    wind_amplitude: Array


class BarotropicQGPhysConsts(PhysConsts):
    """Frozen physical constants for the barotropic QG model.

    Args:
        f0: Reference Coriolis parameter.
        beta: Meridional gradient of f.
    """

    f0: float = eqx.field(static=True, default=1e-4)
    beta: float = eqx.field(static=True, default=1.6e-11)


class BarotropicQGDiagnostics(Diagnostics):
    """Diagnostics for the barotropic QG model.

    Args:
        psi: Streamfunction.
        u: x-velocity (geostrophic).
        v: y-velocity (geostrophic).
        kinetic_energy: Domain-integrated kinetic energy.
        enstrophy: Domain-integrated enstrophy.
        relative_vorticity: ζ = ∇²ψ.
    """

    psi: Float[Array, "Ny Nx"]
    u: Float[Array, "Ny Nx"]
    v: Float[Array, "Ny Nx"]
    kinetic_energy: Array
    enstrophy: Array
    relative_vorticity: Float[Array, "Ny Nx"]


class BarotropicQG(SomaxModel):
    r"""Barotropic quasi-geostrophic model on an Arakawa C-grid.

    Solves the barotropic QG PV equation::

        dq/dt = -J(ψ, q) + tau0*F_wind - kappa*laplacian(psi) + nu*laplacian(q)

    where:
        - q is potential vorticity
        - ψ is the streamfunction from PV inversion: ∇²ψ = q - β·y
        - J(ψ, q) is the Arakawa Jacobian (energy+enstrophy conserving)
        - u = -∂ψ/∂y, v = ∂ψ/∂x (geostrophic velocity)

    Args:
        params: Differentiable parameters.
        consts: Frozen physical constants.
        grid: 2D Arakawa C-grid.
        diff: Difference operators.
        interp: Interpolation operators.
        beta_y: Precomputed β·y field.
        wind_forcing: Precomputed wind stress curl pattern (normalised).
        poisson_bc: Spectral solver BC type for PV inversion.
    """

    params: BarotropicQGParams
    consts: BarotropicQGPhysConsts = eqx.field(static=True)
    grid: ArakawaCGrid2D = eqx.field(static=True)
    diff: Difference2D = eqx.field(static=True)
    interp: Interpolation2D = eqx.field(static=True)
    beta_y: Float[Array, "Ny Nx"]
    wind_forcing: Float[Array, "Ny Nx"]
    poisson_bc: str = eqx.field(static=True, default="dst")

    def _invert_pv(self, q: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        r"""Recover streamfunction from PV anomaly via Poisson inversion.

        The state q is the PV anomaly (relative vorticity zeta = nabla^2 psi).
        Solves nabla^2 psi = q with Dirichlet BCs (psi=0 at walls).
        """
        return streamfunction_from_vorticity(
            q, self.grid.dx, self.grid.dy, bc=self.poisson_bc
        )

    def vector_field(
        self, t: float, state: PyTree, args: PyTree | None = None
    ) -> BarotropicQGState:
        """Compute PV anomaly tendency."""
        q = state.q
        nu = self.params.lateral_viscosity
        kappa = self.params.bottom_drag
        tau0 = self.params.wind_amplitude

        # 1. PV inversion: nabla^2 psi = q (anomaly)
        psi = self._invert_pv(q)

        # 2. Advection: -J(psi, q + beta*y)  (advect total PV)
        q_total = q + self.beta_y
        J_interior = arakawa_jacobian(psi, q_total, self.grid.dx, self.grid.dy)
        J_full = jnp.zeros_like(q)
        J_full = J_full.at[1:-1, 1:-1].set(J_interior)
        dq_dt = -J_full

        # 3. Wind forcing
        dq_dt = dq_dt + tau0 * self.wind_forcing

        # 4. Bottom drag: -kappa * nabla^2 psi (acts on relative vorticity)
        zeta = self.diff.laplacian(psi)
        dq_dt = dq_dt - kappa * zeta

        # 5. Lateral diffusion: nu * laplacian(q)
        dq_dt = dq_dt + nu * self.diff.laplacian(q)

        return BarotropicQGState(q=dq_dt)

    def apply_boundary_conditions(self, state: PyTree) -> BarotropicQGState:
        """Apply Dirichlet BCs (q=0 at boundaries for free-slip walls)."""
        q = zero_boundaries(state.q)
        return BarotropicQGState(q=q)

    def diagnose(self, state: PyTree) -> BarotropicQGDiagnostics:
        """Compute streamfunction, velocity, KE, and enstrophy."""
        q = state.q
        psi = self._invert_pv(q)

        # Geostrophic velocity: u = -dpsi/dy, v = dpsi/dx
        u = -self.diff.diff_y_T_to_V(psi)
        v = self.diff.diff_x_T_to_U(psi)

        s = (slice(1, -1), slice(1, -1))
        u_T = self.interp.V_to_T(u)
        v_T = self.interp.U_to_T(v)
        ke = 0.5 * jnp.sum(u_T[s] ** 2 + v_T[s] ** 2)
        zeta = self.diff.laplacian(psi)
        enstrophy = 0.5 * jnp.sum(q[s] ** 2)

        return BarotropicQGDiagnostics(
            psi=psi,
            u=u,
            v=v,
            kinetic_energy=ke,
            enstrophy=enstrophy,
            relative_vorticity=zeta,
        )

    @staticmethod
    def create(
        nx: int = 64,
        ny: int = 64,
        Lx: float = 1e6,
        Ly: float = 1e6,
        f0: float = 1e-4,
        beta: float = 1.6e-11,
        lateral_viscosity: float = 0.0,
        bottom_drag: float = 0.0,
        wind_amplitude: float = 0.0,
        wind_profile: str = "doublegyre",
    ) -> BarotropicQG:
        """Convenience factory.

        Args:
            nx: Number of interior cells in x.
            ny: Number of interior cells in y.
            Lx: Domain length in x (m).
            Ly: Domain length in y (m).
            f0: Reference Coriolis parameter (1/s).
            beta: Meridional gradient of f (1/(m·s)).
            lateral_viscosity: Harmonic viscosity (m²/s).
            bottom_drag: Linear bottom drag (1/s).
            wind_amplitude: Wind forcing amplitude.
            wind_profile: Wind stress curl profile. ``"doublegyre"``
                gives sin(2πy/Ly), ``"single"`` gives sin(πy/Ly).

        Returns:
            A ``BarotropicQG`` model instance.
        """
        grid = ArakawaCGrid2D.from_interior(nx, ny, Lx, Ly)
        params = BarotropicQGParams(
            lateral_viscosity=jnp.array(lateral_viscosity),
            bottom_drag=jnp.array(bottom_drag),
            wind_amplitude=jnp.array(wind_amplitude),
        )
        consts = BarotropicQGPhysConsts(f0=f0, beta=beta)
        diff = Difference2D(grid=grid)
        interp = Interpolation2D(grid=grid)

        # Precompute β·y field
        y = jnp.arange(grid.Ny) * grid.dy
        y0 = Ly / 2.0
        Y = jnp.broadcast_to(y[:, None], (grid.Ny, grid.Nx))
        beta_y = beta * (Y - y0)

        # Wind forcing profile (normalised curl of wind stress)
        if wind_profile == "single":
            wind_forcing = jnp.sin(jnp.pi * Y / Ly)
        else:
            # Double gyre: curl(tau) ~ -sin(2*pi*y/Ly)
            wind_forcing = -jnp.sin(2.0 * jnp.pi * Y / Ly)

        return BarotropicQG(
            params=params,
            consts=consts,
            grid=grid,
            diff=diff,
            interp=interp,
            beta_y=beta_y,
            wind_forcing=wind_forcing,
            poisson_bc="dst",
        )
