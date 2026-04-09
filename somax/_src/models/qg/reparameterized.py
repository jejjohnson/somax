"""Reparameterized quasi-geostrophic model (QG = SWM + projection).

Based on Thiry, Li, Memin & Roullet (2024), "A Unified Formulation of
Quasi-Geostrophic and Shallow Water Equations via Projection," JAMES 16(10).

The reparameterized QG model uses the same state variables (u, v, h) as the
multilayer shallow water model. The only difference is a geostrophic
projection P applied after each time step, keeping the state on the
geostrophic manifold.
"""

from __future__ import annotations

import equinox as eqx
from finitevolx import (
    ArakawaCGrid2D,
    Difference2D,
    Interpolation2D,
    Vorticity2D,
    multilayer,
    pv_inversion,
    zero_boundaries,
)
from jaxtyping import Array, Float, PyTree

from somax._src.core.model import SomaxModel
from somax._src.core.transforms import ModalTransform, StratificationProfile
from somax._src.core.types import Diagnostics
from somax._src.models.swm.multilayer import (
    MultilayerShallowWater2D,
    MultilayerSW2DParams,
    MultilayerSW2DPhysConsts,
    MultilayerSW2DState,
)


class ReparamQGDiagnostics(Diagnostics):
    """Diagnostics for the reparameterized QG model.

    Args:
        energy: Domain-integrated total energy per layer.
        total_energy: Domain-integrated total energy (scalar).
        enstrophy: Domain-integrated potential enstrophy per layer.
        total_enstrophy: Total potential enstrophy (scalar).
        potential_vorticity: PV field per layer.
        relative_vorticity: Relative vorticity per layer.
        kinetic_energy_field: KE at T-points per layer.
        psi: Streamfunction (pressure) per layer.
        u_ageostrophic: Ageostrophic x-velocity per layer.
        v_ageostrophic: Ageostrophic y-velocity per layer.
    """

    energy: Array
    total_energy: Array
    enstrophy: Array
    total_enstrophy: Array
    potential_vorticity: Float[Array, "nl Ny Nx"]
    relative_vorticity: Float[Array, "nl Ny Nx"]
    kinetic_energy_field: Float[Array, "nl Ny Nx"]
    psi: Float[Array, "nl Ny Nx"]
    u_ageostrophic: Float[Array, "nl Ny Nx"]
    v_ageostrophic: Float[Array, "nl Ny Nx"]


class ReparameterizedQG(SomaxModel):
    r"""Reparameterized QG model: multilayer SWM + geostrophic projection.

    Wraps a ``MultilayerShallowWater2D`` and adds a geostrophic
    projection P = G . (Q.G)^{-1} . Q applied via
    ``apply_boundary_conditions``, keeping the state on the
    geostrophic manifold at each time step.

    The three operators are:

    - **Q** (PV extraction): q = curl(u,v) - f0 * eta / H
    - **(Q.G)^{-1}** (Helmholtz solve): modal decomposition + DST
    - **G** (geostrophic reconstruction): p -> (u_g, v_g, h_g)

    The projection is idempotent (P.P = P), so applying it before each
    RHS evaluation is equivalent to projecting after each time step.

    Args:
        swm: The underlying multilayer shallow water model.
        helmholtz_lambdas: f0^2 * eigenvalues per mode.
        poisson_bc: Spectral solver BC type for Helmholtz.
    """

    swm: MultilayerShallowWater2D
    helmholtz_lambdas: Array
    poisson_bc: str = eqx.field(static=True, default="dst")

    # --- Delegate properties for convenience ---
    @property
    def params(self) -> MultilayerSW2DParams:
        return self.swm.params

    @property
    def consts(self) -> MultilayerSW2DPhysConsts:
        return self.swm.consts

    @property
    def grid(self) -> ArakawaCGrid2D:
        return self.swm.grid

    @property
    def diff(self) -> Difference2D:
        return self.swm.diff

    @property
    def interp(self) -> Interpolation2D:
        return self.swm.interp

    @property
    def vorticity(self) -> Vorticity2D:
        return self.swm.vorticity

    @property
    def strat(self) -> StratificationProfile:
        return self.swm.strat

    @property
    def modal(self) -> ModalTransform:
        return self.swm.modal

    def _solve_helmholtz(self, q: Float[Array, "nl Ny Nx"]) -> Float[Array, "nl Ny Nx"]:
        r"""Solve (Q.G)^{-1}: PV -> pressure via modal Helmholtz.

        Solves (nabla^2 - f0^2 * lambda_m) p_m = q_m per mode.
        """
        q_modal = self.modal.to_modal(q)
        p_modal = pv_inversion(
            q_modal,
            self.grid.dx,
            self.grid.dy,
            lambda_=self.helmholtz_lambdas,
            bc=self.poisson_bc,
        )
        p = self.modal.to_layer(p_modal)
        p = multilayer(zero_boundaries)(p)
        return p

    def project(self, state: PyTree) -> MultilayerSW2DState:
        """Project state onto the geostrophic manifold.

        Implements P = G . (Q.G)^{-1} . Q:
        1. Q: extract QG potential vorticity from (u, v, h)
        2. (Q.G)^{-1}: solve Helmholtz for pressure p
        3. G: reconstruct geostrophic (u_g, v_g, h_g) from p

        Args:
            state: Current state with (h, u, v).

        Returns:
            Geostrophically balanced state.
        """
        h, u, v = state.h, state.u, state.v
        f0 = self.consts.f0
        H = self.strat.H  # (nl,)

        # --- Q: extract PV anomaly ---
        # q = curl(u, v) - f0 * eta / H  where eta = h - H
        zeta = multilayer(self.vorticity.relative_vorticity)(u, v)
        eta = h - H[:, None, None]
        q = zeta - f0 * eta / H[:, None, None]

        # --- (Q.G)^{-1}: PV -> streamfunction via modal Helmholtz ---
        psi = self._solve_helmholtz(q)

        # --- G: streamfunction -> geostrophic state ---
        # u_g = -dpsi/dy,  v_g = dpsi/dx  (same as BaroclinicQG)
        u_g = -multilayer(self.diff.diff_y_T_to_V)(psi)
        v_g = multilayer(self.diff.diff_x_T_to_U)(psi)
        # h_g = H + f0 * H * (A @ psi)
        # A @ psi = Cm2l @ (diag(eigenvalues) @ (Cl2m @ psi))
        psi_modal = self.modal.to_modal(psi)
        A_psi = self.modal.to_layer(self.modal.eigenvalues[:, None, None] * psi_modal)
        h_g = H[:, None, None] * (1.0 + f0 * A_psi)

        return MultilayerSW2DState(h=h_g, u=u_g, v=v_g)

    def vector_field(
        self, t: float, state: PyTree, args: PyTree | None = None
    ) -> MultilayerSW2DState:
        """Compute tendencies using the SWM vector field."""
        return self.swm.vector_field(t, state, args)

    def apply_boundary_conditions(self, state: PyTree) -> MultilayerSW2DState:
        """Apply SWM BCs then project onto the geostrophic manifold."""
        state = self.swm.apply_boundary_conditions(state)
        return self.project(state)

    def diagnose(self, state: PyTree) -> ReparamQGDiagnostics:
        """Compute QG diagnostics including ageostrophic velocity."""
        swm_diag = self.swm.diagnose(state)

        h, u, v = state.h, state.u, state.v
        f0 = self.consts.f0
        H = self.strat.H

        zeta = multilayer(self.vorticity.relative_vorticity)(u, v)
        eta = h - H[:, None, None]
        q = zeta - f0 * eta / H[:, None, None]
        psi = self._solve_helmholtz(q)

        # Geostrophic velocity (psi is streamfunction, not pressure)
        u_g = -multilayer(self.diff.diff_y_T_to_V)(psi)
        v_g = multilayer(self.diff.diff_x_T_to_U)(psi)

        # Ageostrophic velocity = total - geostrophic
        u_ageo = u - u_g
        v_ageo = v - v_g

        return ReparamQGDiagnostics(
            energy=swm_diag.energy,
            total_energy=swm_diag.total_energy,
            enstrophy=swm_diag.enstrophy,
            total_enstrophy=swm_diag.total_enstrophy,
            potential_vorticity=swm_diag.potential_vorticity,
            relative_vorticity=swm_diag.relative_vorticity,
            kinetic_energy_field=swm_diag.kinetic_energy_field,
            psi=psi,
            u_ageostrophic=u_ageo,
            v_ageostrophic=v_ageo,
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
        poisson_bc: str = "dst",
    ) -> ReparameterizedQG:
        """Convenience factory for the reparameterized QG model.

        Accepts all arguments of ``MultilayerShallowWater2D.create()``
        plus ``poisson_bc`` for the Helmholtz solver.

        Args:
            nx: Number of interior cells in x.
            ny: Number of interior cells in y.
            Lx: Domain length in x (m).
            Ly: Domain length in y (m).
            g: Gravitational acceleration (m/s^2).
            f0: Reference Coriolis parameter (1/s).
            beta: Meridional gradient of f (1/(m*s)).
            n_layers: Number of layers.
            H: Layer thicknesses (m).
            g_prime: Reduced gravities (m/s^2).
            stratification: Pre-built ``StratificationProfile``.
            lateral_viscosity: Harmonic viscosity (m^2/s).
            bottom_drag: Linear bottom drag (1/s).
            wind_amplitude: Wind stress amplitude (m/s^2).
            wind_profile: Wind stress pattern.
            bc: Boundary condition type.
            method: Advection reconstruction method.
            poisson_bc: Spectral solver BC type for Helmholtz.

        Returns:
            A ``ReparameterizedQG`` model instance.
        """
        swm = MultilayerShallowWater2D.create(
            nx=nx,
            ny=ny,
            Lx=Lx,
            Ly=Ly,
            g=g,
            f0=f0,
            beta=beta,
            n_layers=n_layers,
            H=H,
            g_prime=g_prime,
            stratification=stratification,
            lateral_viscosity=lateral_viscosity,
            bottom_drag=bottom_drag,
            wind_amplitude=wind_amplitude,
            wind_profile=wind_profile,
            bc=bc,
            method=method,
        )

        helmholtz_lambdas = f0**2 * swm.modal.eigenvalues

        return ReparameterizedQG(
            swm=swm,
            helmholtz_lambdas=helmholtz_lambdas,
            poisson_bc=poisson_bc,
        )
