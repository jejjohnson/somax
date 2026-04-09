"""Multilayer 2D nonlinear shallow water model (vector-invariant form)."""

from __future__ import annotations

from functools import partial

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
    multilayer,
)
from jaxtyping import Array, Float, PyTree

from somax._src.core.model import SomaxModel
from somax._src.core.transforms import ModalTransform, StratificationProfile
from somax._src.core.types import Diagnostics, Params, PhysConsts, State


class MultilayerSW2DState(State):
    """State for the multilayer 2D nonlinear shallow water model.

    Args:
        h: Total layer thickness per layer, shape ``(nl, Ny, Nx)``.
        u: x-velocity on U-points per layer, shape ``(nl, Ny, Nx)``.
        v: y-velocity on V-points per layer, shape ``(nl, Ny, Nx)``.
    """

    h: Float[Array, "nl Ny Nx"]
    u: Float[Array, "nl Ny Nx"]
    v: Float[Array, "nl Ny Nx"]


class MultilayerSW2DParams(Params):
    """Differentiable parameters for the multilayer 2D shallow water model.

    Args:
        lateral_viscosity: Harmonic viscosity coefficient nu (m^2/s).
        bottom_drag: Linear bottom drag coefficient kappa (1/s).
        wind_amplitude: Wind stress amplitude (m/s^2).
    """

    lateral_viscosity: Array
    bottom_drag: Array
    wind_amplitude: Array


class MultilayerSW2DPhysConsts(PhysConsts):
    """Frozen physical constants for the multilayer 2D shallow water model.

    Args:
        gravity: Gravitational acceleration g.
        f0: Reference Coriolis parameter.
        beta: Meridional gradient of f (beta-plane).
        n_layers: Number of vertical layers.
    """

    gravity: float = eqx.field(static=True, default=9.81)
    f0: float = eqx.field(static=True, default=1e-4)
    beta: float = eqx.field(static=True, default=0.0)
    n_layers: int = eqx.field(static=True, default=2)


class MultilayerSW2DDiagnostics(Diagnostics):
    """Diagnostics for the multilayer 2D shallow water model.

    Args:
        energy: Domain-integrated total energy per layer.
        total_energy: Domain-integrated total energy (scalar).
        enstrophy: Domain-integrated potential enstrophy per layer.
        total_enstrophy: Total potential enstrophy (scalar).
        potential_vorticity: PV field q = (zeta+f)/h at X-points per layer.
        relative_vorticity: Relative vorticity zeta per layer.
        kinetic_energy_field: KE at T-points per layer.
    """

    energy: Array
    total_energy: Array
    enstrophy: Array
    total_enstrophy: Array
    potential_vorticity: Float[Array, "nl Ny Nx"]
    relative_vorticity: Float[Array, "nl Ny Nx"]
    kinetic_energy_field: Float[Array, "nl Ny Nx"]


class MultilayerShallowWater2D(SomaxModel):
    r"""Multilayer 2D nonlinear shallow water model (vector-invariant form).

    Solves the rotating shallow water equations per layer k::

        dh_k/dt = -div(h_k * u_k)
        du_k/dt = +q_k * (h_k v_k)_bar - dP_k/dx + forcing
        dv_k/dt = -q_k * (h_k u_k)_bar - dP_k/dy + forcing

    where q_k = (zeta_k + f) / h_k is potential vorticity and
    P_k = KE_k + p_k is the Bernoulli potential with hydrostatic
    pressure coupling between layers:
    p_k = sum_{j=0}^{k} g'_j * h_j (cumulative).

    Wind forcing is applied to the top layer only; bottom drag
    to the bottom layer only.

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
        strat: Stratification profile (layer depths and reduced gravities).
        modal: Precomputed modal transform.
        f_field: Precomputed Coriolis field f(y) at T-points.
        f_field_ml: Coriolis field broadcast to ``(nl, Ny, Nx)``.
        wind_stress_x: Precomputed x-wind stress pattern (normalised).
        wind_stress_y: Precomputed y-wind stress pattern (normalised).
        bc_type: Boundary condition type.
        method: Advection reconstruction method for mass equation.
    """

    params: MultilayerSW2DParams
    consts: MultilayerSW2DPhysConsts = eqx.field(static=True)
    grid: ArakawaCGrid2D = eqx.field(static=True)
    diff: Difference2D = eqx.field(static=True)
    interp: Interpolation2D = eqx.field(static=True)
    coriolis: Coriolis2D = eqx.field(static=True)
    vorticity: Vorticity2D = eqx.field(static=True)
    advection: FVXAdvection2D = eqx.field(static=True)
    diffusion: FVXDiffusion2D = eqx.field(static=True)
    strat: StratificationProfile
    modal: ModalTransform
    f_field: Float[Array, "Ny Nx"]
    f_field_ml: Float[Array, "nl Ny Nx"]
    wind_stress_x: Float[Array, "Ny Nx"]
    wind_stress_y: Float[Array, "Ny Nx"]
    bc_type: str = eqx.field(static=True, default="periodic")
    method: str = eqx.field(static=True, default="upwind1")

    def vector_field(
        self, t: float, state: PyTree, args: PyTree | None = None
    ) -> MultilayerSW2DState:
        """Compute tendencies for the multilayer shallow water equations."""
        h, u, v = state.h, state.u, state.v
        nu = self.params.lateral_viscosity
        kappa = self.params.bottom_drag
        tau0 = self.params.wind_amplitude

        # --- 1. Mass equation: dh_k/dt = -div(h_k * u_k, h_k * v_k) ---
        dh_dt = multilayer(partial(self.advection, method=self.method))(h, u, v)

        # --- 2. Potential vorticity: q_k = (zeta_k + f) / h_k at X-points ---
        # Broadcast f_field to (nl, Ny, Nx) so multilayer vmaps correctly
        q = multilayer(self.vorticity.potential_vorticity)(u, v, h, self.f_field_ml)

        # --- 3. Cross-velocity PV fluxes (vector-invariant form) ---
        h_on_U = multilayer(self.interp.T_to_U)(h)
        h_on_V = multilayer(self.interp.T_to_V)(h)
        uh = h_on_U * u  # mass flux at U-points
        vh = h_on_V * v  # mass flux at V-points

        q_on_U = multilayer(self.interp.X_to_U)(q)
        q_on_V = multilayer(self.interp.X_to_V)(q)
        vh_on_U = multilayer(self.interp.V_to_U)(vh)
        uh_on_V = multilayer(self.interp.U_to_V)(uh)

        # --- 4. Bernoulli potential: P_k = KE_k + p_k ---
        ke = multilayer(kinetic_energy)(u, v)
        # Hydrostatic pressure coupling: p_k = sum_{j=0}^{k} g'_j * h_j
        g_prime = self.strat.g_prime
        weighted_h = g_prime[:, None, None] * h
        p = jnp.cumsum(weighted_h, axis=0)
        P = ke + p  # Bernoulli potential per layer

        # --- 5. Momentum tendencies ---
        du_dt = q_on_U * vh_on_U - multilayer(self.diff.diff_x_T_to_U)(P)
        dv_dt = -q_on_V * uh_on_V - multilayer(self.diff.diff_y_T_to_V)(P)

        # --- 6. Wind forcing (top layer only) ---
        du_dt = du_dt.at[0].add(tau0 * self.wind_stress_x / self.strat.H[0])
        dv_dt = dv_dt.at[0].add(tau0 * self.wind_stress_y / self.strat.H[0])

        # --- 7. Diffusion (all layers) ---
        du_dt = du_dt + multilayer(lambda f: self.diffusion(f, nu))(u)
        dv_dt = dv_dt + multilayer(lambda f: self.diffusion(f, nu))(v)

        # --- 8. Bottom drag (bottom layer only) ---
        du_dt = du_dt.at[-1].add(-kappa * u[-1])
        dv_dt = dv_dt.at[-1].add(-kappa * v[-1])

        return MultilayerSW2DState(h=dh_dt, u=du_dt, v=dv_dt)

    def apply_boundary_conditions(self, state: PyTree) -> MultilayerSW2DState:
        """Apply boundary conditions to all state fields per layer.

        For periodic: ghost cells wrap around.
        For wall (free-slip, no-normal-flow):
          - u = 0 at x-boundaries per layer
          - v = 0 at y-boundaries per layer
          - Tangential velocity: zero-gradient (free-slip)
          - h: zero-gradient at all walls
        """
        if self.bc_type == "periodic":
            h = multilayer(enforce_periodic)(state.h)
            u = multilayer(enforce_periodic)(state.u)
            v = multilayer(enforce_periodic)(state.v)
        else:
            h, u, v = state.h, state.u, state.v
            # Apply wall BCs per layer using vmap
            u = _apply_wall_bc_u(u)
            v = _apply_wall_bc_v(v)
            h = _apply_wall_bc_h(h)
        return MultilayerSW2DState(h=h, u=u, v=v)

    def diagnose(self, state: PyTree) -> MultilayerSW2DDiagnostics:
        """Compute energy, enstrophy, PV, vorticity, and KE per layer."""
        h, u, v = state.h, state.u, state.v
        g_prime = self.strat.g_prime
        s = (slice(None), slice(1, -1), slice(1, -1))

        ke = multilayer(kinetic_energy)(u, v)
        zeta = multilayer(self.vorticity.relative_vorticity)(u, v)
        q = multilayer(self.vorticity.potential_vorticity)(u, v, h, self.f_field_ml)
        h_on_X = multilayer(self.interp.T_to_X)(h)

        cell_area = self.grid.dx * self.grid.dy

        # Energy per layer: KE + PE
        ke_sum = jnp.sum(ke[s], axis=(-2, -1)) * cell_area
        pe_sum = 0.5 * g_prime * jnp.sum(h[s] ** 2, axis=(-2, -1)) * cell_area
        energy_per_layer = ke_sum + pe_sum

        # Potential enstrophy per layer: 0.5 * q^2 * h
        enstrophy_per_layer = (
            0.5 * jnp.sum(q[s] ** 2 * h_on_X[s], axis=(-2, -1)) * cell_area
        )

        return MultilayerSW2DDiagnostics(
            energy=energy_per_layer,
            total_energy=jnp.sum(energy_per_layer),
            enstrophy=enstrophy_per_layer,
            total_enstrophy=jnp.sum(enstrophy_per_layer),
            potential_vorticity=q,
            relative_vorticity=zeta,
            kinetic_energy_field=ke,
        )

    @staticmethod
    def create(
        nx: int = 64,
        ny: int = 64,
        Lx: float = 4e6,
        Ly: float = 4e6,
        g: float = 9.81,
        f0: float = 9.375e-5,
        beta: float = 1.754e-11,
        n_layers: int = 3,
        H: tuple[float, ...] = (400.0, 1100.0, 2600.0),
        g_prime: tuple[float, ...] = (9.81, 0.025, 0.0125),
        stratification: StratificationProfile | None = None,
        lateral_viscosity: float = 0.0,
        bottom_drag: float = 0.0,
        wind_amplitude: float = 0.0,
        wind_profile: str = "doublegyre",
        bc: str = "periodic",
        method: str = "upwind1",
    ) -> MultilayerShallowWater2D:
        """Convenience factory for the multilayer shallow water model.

        Args:
            nx: Number of interior cells in x.
            ny: Number of interior cells in y.
            Lx: Domain length in x (m).
            Ly: Domain length in y (m).
            g: Gravitational acceleration (m/s^2) (only used
                if ``stratification`` is None to set g_prime[0]).
            f0: Reference Coriolis parameter (1/s).
            beta: Meridional gradient of f (1/(m*s)).
            n_layers: Number of layers (ignored if ``stratification`` given).
            H: Layer thicknesses (m), top to bottom
                (ignored if ``stratification`` given).
            g_prime: Reduced gravities (m/s^2) at each interface;
                g_prime[0] = g (full gravity)
                (ignored if ``stratification`` given).
            stratification: Pre-built ``StratificationProfile``, or None
                to build from ``H`` and ``g_prime``.
            lateral_viscosity: Harmonic viscosity (m^2/s).
            bottom_drag: Linear bottom drag (1/s).
            wind_amplitude: Wind stress amplitude (m/s^2).
            wind_profile: Wind stress pattern. ``"doublegyre"``
                gives tau_x = -cos(2*pi*y/Ly), ``"single"`` gives
                tau_x = -cos(pi*y/Ly).
            bc: Boundary condition type.
            method: Advection reconstruction method for mass.

        Returns:
            A ``MultilayerShallowWater2D`` model instance.

        Raises:
            ValueError: If ``n_layers``, ``H``, and ``g_prime`` have
                inconsistent lengths (when ``stratification`` is None).
        """
        grid = ArakawaCGrid2D.from_interior(nx, ny, Lx, Ly)

        # Stratification
        if stratification is not None:
            strat = stratification
        else:
            if len(H) != n_layers or len(g_prime) != n_layers:
                msg = (
                    f"n_layers ({n_layers}), len(H) ({len(H)}), and "
                    f"len(g_prime) ({len(g_prime)}) must all be equal"
                )
                raise ValueError(msg)
            strat = StratificationProfile.from_layers(H=list(H), g_prime=list(g_prime))

        nl = strat.nl

        # Modal transform (for diagnostics / optional barotropic filter)
        modal = ModalTransform.from_stratification(strat, f0)

        params = MultilayerSW2DParams(
            lateral_viscosity=jnp.array(lateral_viscosity),
            bottom_drag=jnp.array(bottom_drag),
            wind_amplitude=jnp.array(wind_amplitude),
        )
        consts = MultilayerSW2DPhysConsts(gravity=g, f0=f0, beta=beta, n_layers=nl)
        diff = Difference2D(grid=grid)
        interp = Interpolation2D(grid=grid)
        coriolis = Coriolis2D(grid=grid)
        vorticity_op = Vorticity2D(grid=grid)
        advection = FVXAdvection2D(grid=grid)
        diffusion = FVXDiffusion2D(grid=grid)

        # Precompute Coriolis field
        y = jnp.arange(grid.Ny) * grid.dy
        y0 = Ly / 2.0
        Y = jnp.broadcast_to(y[:, None], (grid.Ny, grid.Nx))
        f_field = coriolis_fn(Y, f0=f0, beta=beta, y0=y0)
        f_field_ml = jnp.broadcast_to(f_field[None], (nl, grid.Ny, grid.Nx)).copy()

        # Wind stress pattern (normalised)
        if wind_profile == "single":
            wind_stress_x = -jnp.cos(jnp.pi * Y / Ly)
        else:
            wind_stress_x = -jnp.cos(2.0 * jnp.pi * Y / Ly)
        wind_stress_y = jnp.zeros_like(wind_stress_x)

        return MultilayerShallowWater2D(
            params=params,
            consts=consts,
            grid=grid,
            diff=diff,
            interp=interp,
            coriolis=coriolis,
            vorticity=vorticity_op,
            advection=advection,
            diffusion=diffusion,
            strat=strat,
            modal=modal,
            f_field=f_field,
            f_field_ml=f_field_ml,
            wind_stress_x=wind_stress_x,
            wind_stress_y=wind_stress_y,
            bc_type=bc,
            method=method,
        )


# --- Wall BC helpers (applied per layer via vmap) ---


def _wall_bc_u_single(u: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
    """Wall BCs for u: no normal flow at x-boundaries, free-slip tangential."""
    # No normal flow at x-boundaries
    u = u.at[:, 0].set(0.0).at[:, -2].set(0.0).at[:, -1].set(0.0)
    # Free-slip tangential
    u = u.at[0, :].set(u[1, :]).at[-1, :].set(u[-2, :])
    return u


def _wall_bc_v_single(v: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
    """Wall BCs for v: no normal flow at y-boundaries, free-slip tangential."""
    # No normal flow at y-boundaries
    v = v.at[0, :].set(0.0).at[-2, :].set(0.0).at[-1, :].set(0.0)
    # Free-slip tangential
    v = v.at[:, 0].set(v[:, 1]).at[:, -1].set(v[:, -2])
    return v


def _wall_bc_h_single(h: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
    """Wall BCs for h: zero-gradient at all walls."""
    h = h.at[0, :].set(h[1, :]).at[-1, :].set(h[-2, :])
    h = h.at[:, 0].set(h[:, 1]).at[:, -1].set(h[:, -2])
    return h


_apply_wall_bc_u = multilayer(_wall_bc_u_single)
_apply_wall_bc_v = multilayer(_wall_bc_v_single)
_apply_wall_bc_h = multilayer(_wall_bc_h_single)
