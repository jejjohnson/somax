"""Baroclinic (multilayer) quasi-geostrophic model."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from finitevolx import (
    ArakawaCGrid2D,
    Difference2D,
    Interpolation2D,
    arakawa_jacobian,
    multilayer,
    pv_inversion,
    zero_boundaries,
)
from jaxtyping import Array, Float, PyTree

from somax._src.core.model import SomaxModel
from somax._src.core.transforms import ModalTransform, StratificationProfile
from somax._src.core.types import Diagnostics, Params, PhysConsts, State


class BaroclinicQGState(State):
    """State for the multilayer quasi-geostrophic model.

    Args:
        q: Layer PV anomaly on T-points, shape ``(nl, Ny, Nx)``.
            The total PV in layer k is ``q[k] + beta * (y - y0)``.
            PV inversion solves
            ``(nabla^2 - f0^2 * A) psi = q`` in layer space,
            which decouples to per-mode Helmholtz problems in modal space.
    """

    q: Float[Array, "nl Ny Nx"]


class BaroclinicQGParams(Params):
    """Differentiable parameters for the multilayer QG model.

    Args:
        lateral_viscosity: Harmonic viscosity coefficient nu (m^2/s).
        bottom_drag: Linear bottom drag coefficient kappa (1/s).
        wind_amplitude: Wind forcing amplitude tau0.
    """

    lateral_viscosity: Array
    bottom_drag: Array
    wind_amplitude: Array


class BaroclinicQGPhysConsts(PhysConsts):
    """Frozen physical constants for the multilayer QG model.

    Args:
        f0: Reference Coriolis parameter (1/s).
        beta: Meridional gradient of f (1/(m*s)).
        n_layers: Number of layers.
    """

    f0: float = eqx.field(static=True, default=1e-4)
    beta: float = eqx.field(static=True, default=1.6e-11)
    n_layers: int = eqx.field(static=True, default=2)


class BaroclinicQGDiagnostics(Diagnostics):
    """Diagnostics for the multilayer QG model.

    Args:
        psi: Streamfunction per layer, shape ``(nl, Ny, Nx)``.
        u: x-velocity (geostrophic) per layer.
        v: y-velocity (geostrophic) per layer.
        kinetic_energy: Domain-integrated KE per layer, shape ``(nl,)``.
        total_kinetic_energy: Domain-integrated total KE (scalar).
        enstrophy: Domain-integrated enstrophy per layer, shape ``(nl,)``.
        total_enstrophy: Domain-integrated total enstrophy (scalar).
        relative_vorticity: Relative vorticity per layer.
        rossby_radii: Deformation radii per mode, shape ``(nl,)``.
    """

    psi: Float[Array, "nl Ny Nx"]
    u: Float[Array, "nl Ny Nx"]
    v: Float[Array, "nl Ny Nx"]
    kinetic_energy: Array
    total_kinetic_energy: Array
    enstrophy: Array
    total_enstrophy: Array
    relative_vorticity: Float[Array, "nl Ny Nx"]
    rossby_radii: Array


class BaroclinicQG(SomaxModel):
    r"""Multilayer quasi-geostrophic model on an Arakawa C-grid.

    Solves the multilayer QG PV equation per layer k::

        dq_k/dt = -J(psi_k, q_k + beta*y)
                  + tau0 * F_wind / H[0]  (top layer only)
                  - kappa * zeta_{nl-1}   (bottom layer only)
                  + nu * laplacian(q_k)

    PV inversion uses vertical mode decomposition::

        q_modal = Cl2m @ q_layer
        (nabla^2 - f0^2 * lambda_m) psi_modal_m = q_modal_m
        psi_layer = Cm2l @ psi_modal

    following the MQGeometry approach (louity/MQGeometry).

    Args:
        params: Differentiable parameters.
        consts: Frozen physical constants.
        grid: 2D Arakawa C-grid.
        diff: Difference operators.
        interp: Interpolation operators.
        modal: Precomputed modal transform.
        strat: Stratification profile.
        beta_y: Precomputed beta*(y - y0) field.
        wind_forcing: Normalised wind stress curl pattern.
        helmholtz_lambdas: f0^2 * eigenvalues per mode, shape ``(nl,)``.
        poisson_bc: Spectral solver BC type for PV inversion.
    """

    params: BaroclinicQGParams
    consts: BaroclinicQGPhysConsts = eqx.field(static=True)
    grid: ArakawaCGrid2D = eqx.field(static=True)
    diff: Difference2D = eqx.field(static=True)
    interp: Interpolation2D = eqx.field(static=True)
    modal: ModalTransform
    strat: StratificationProfile
    beta_y: Float[Array, "Ny Nx"]
    wind_forcing: Float[Array, "Ny Nx"]
    helmholtz_lambdas: Array
    poisson_bc: str = eqx.field(static=True, default="dst")

    def _invert_pv(self, q: Float[Array, "nl Ny Nx"]) -> Float[Array, "nl Ny Nx"]:
        r"""Recover streamfunction from layer PV anomaly.

        1. Project q to modal space.
        2. Solve per-mode Helmholtz: (nabla^2 - lambda_m) psi_m = q_m.
        3. Project back to layer space.
        """
        # Layer -> modal
        q_modal = self.modal.to_modal(q)

        # Per-mode Helmholtz solve
        psi_modal = pv_inversion(
            q_modal,
            self.grid.dx,
            self.grid.dy,
            lambda_=self.helmholtz_lambdas,
            bc=self.poisson_bc,
        )

        # Modal -> layer
        psi = self.modal.to_layer(psi_modal)

        # Enforce boundary conditions on psi
        psi = multilayer(zero_boundaries)(psi)
        return psi

    def vector_field(
        self, t: float, state: PyTree, args: PyTree | None = None
    ) -> BaroclinicQGState:
        """Compute PV anomaly tendency for all layers."""
        q = state.q
        nu = self.params.lateral_viscosity
        kappa = self.params.bottom_drag
        tau0 = self.params.wind_amplitude

        # 1. PV inversion
        psi = self._invert_pv(q)

        # 2. Advection: -J(psi_k, q_k + beta*y) per layer
        q_total = q + self.beta_y[None, :, :]
        J_interior = arakawa_jacobian(psi, q_total, self.grid.dx, self.grid.dy)
        J_full = jnp.zeros_like(q)
        J_full = J_full.at[:, 1:-1, 1:-1].set(J_interior)
        dq_dt = -J_full

        # 3. Wind forcing: top layer only, scaled by 1/H[0]
        dq_dt = dq_dt.at[0].add(tau0 * self.wind_forcing / self.strat.H[0])

        # 4. Bottom drag: -kappa * zeta on bottom layer only
        zeta_bottom = self.diff.laplacian(psi[-1])
        dq_dt = dq_dt.at[-1].add(-kappa * zeta_bottom)

        # 5. Lateral diffusion: nu * laplacian(q) for all layers
        dq_dt = dq_dt + nu * multilayer(self.diff.laplacian)(q)

        return BaroclinicQGState(q=dq_dt)

    def apply_boundary_conditions(self, state: PyTree) -> BaroclinicQGState:
        """Apply Dirichlet BCs (q=0 at boundaries) for all layers."""
        q = multilayer(zero_boundaries)(state.q)
        return BaroclinicQGState(q=q)

    def diagnose(self, state: PyTree) -> BaroclinicQGDiagnostics:
        """Compute streamfunction, velocity, KE, and enstrophy per layer."""
        q = state.q
        psi = self._invert_pv(q)

        # Geostrophic velocity per layer
        u = -multilayer(self.diff.diff_y_T_to_V)(psi)
        v = multilayer(self.diff.diff_x_T_to_U)(psi)

        # KE and enstrophy per layer (interior only)
        s = (slice(None), slice(1, -1), slice(1, -1))
        u_T = multilayer(self.interp.V_to_T)(u)
        v_T = multilayer(self.interp.U_to_T)(v)
        cell_area = self.grid.dx * self.grid.dy
        ke_per_layer = (
            0.5 * jnp.sum(u_T[s] ** 2 + v_T[s] ** 2, axis=(-2, -1)) * cell_area
        )
        enstrophy_per_layer = 0.5 * jnp.sum(q[s] ** 2, axis=(-2, -1)) * cell_area

        zeta = multilayer(self.diff.laplacian)(psi)

        return BaroclinicQGDiagnostics(
            psi=psi,
            u=u,
            v=v,
            kinetic_energy=ke_per_layer,
            total_kinetic_energy=jnp.sum(ke_per_layer),
            enstrophy=enstrophy_per_layer,
            total_enstrophy=jnp.sum(enstrophy_per_layer),
            relative_vorticity=zeta,
            rossby_radii=self.modal.rossby_radii,
        )

    @staticmethod
    def create(
        nx: int = 64,
        ny: int = 64,
        Lx: float = 4e6,
        Ly: float = 4e6,
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
        poisson_bc: str = "dst",
    ) -> BaroclinicQG:
        """Convenience factory for the multilayer QG model.

        Args:
            nx: Number of interior cells in x.
            ny: Number of interior cells in y.
            Lx: Domain length in x (m).
            Ly: Domain length in y (m).
            f0: Reference Coriolis parameter (1/s).
            beta: Meridional gradient of f (1/(m*s)).
            n_layers: Number of layers (ignored if ``stratification`` given).
            H: Layer thicknesses (m), top to bottom
                (ignored if ``stratification`` given).
            g_prime: Reduced gravities (m/s^2) at each interface
                (ignored if ``stratification`` given).
            stratification: Pre-built ``StratificationProfile``, or None
                to build from ``H`` and ``g_prime``.
            lateral_viscosity: Harmonic viscosity (m^2/s).
            bottom_drag: Linear bottom drag (1/s).
            wind_amplitude: Wind forcing amplitude.
            wind_profile: Wind stress curl profile. ``"doublegyre"`` gives
                ``-sin(2*pi*y/Ly)``, ``"single"`` gives ``sin(pi*y/Ly)``.
            poisson_bc: Spectral solver BC type for PV inversion.

        Returns:
            A ``BaroclinicQG`` model instance.

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

        # Modal transform and Helmholtz parameters
        modal = ModalTransform.from_stratification(strat, f0)
        helmholtz_lambdas = f0**2 * modal.eigenvalues

        params = BaroclinicQGParams(
            lateral_viscosity=jnp.array(lateral_viscosity),
            bottom_drag=jnp.array(bottom_drag),
            wind_amplitude=jnp.array(wind_amplitude),
        )
        consts = BaroclinicQGPhysConsts(f0=f0, beta=beta, n_layers=nl)
        diff = Difference2D(grid=grid)
        interp = Interpolation2D(grid=grid)

        # Precompute beta*y field
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

        return BaroclinicQG(
            params=params,
            consts=consts,
            grid=grid,
            diff=diff,
            interp=interp,
            modal=modal,
            strat=strat,
            beta_y=beta_y,
            wind_forcing=wind_forcing,
            helmholtz_lambdas=helmholtz_lambdas,
            poisson_bc=poisson_bc,
        )
