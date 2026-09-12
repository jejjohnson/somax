"""Reparameterized quasi-geostrophic model (QG = SWM + projection).

Based on Thiry, Li, Memin & Roullet (2024), "A Unified Formulation of
Quasi-Geostrophic and Shallow Water Equations via Projection," JAMES 16(10).

The reparameterized QG model uses the same state variables (u, v, h) as the
multilayer shallow water model. The only difference is a geostrophic
projection P applied after each time step, keeping the state on the
geostrophic manifold.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from finitevolx import (
    CartesianGrid2D,
    Difference2D,
    Interpolation2D,
    Mask2D,
    Vorticity2D,
    multilayer,
    pv_inversion,
    zero_boundaries,
)
from jaxtyping import Array, Float, PyTree

from somax._src.core.model import SomaxModel
from somax._src.core.scales import Scales
from somax._src.core.transforms import ModalTransform, StratificationProfile
from somax._src.core.types import Diagnostics
from somax._src.models._nondim import (
    burger_to_g_prime,
    reject_derived_kwargs,
    require_non_negative,
    require_positive,
)
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
        mass: Domain-integrated mass/volume per layer (flux-form invariant).
        casimir_q3: PV Casimir per layer.
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
    mass: Array
    casimir_q3: Array
    potential_vorticity: Float[Array, "nl Ny Nx"]
    relative_vorticity: Float[Array, "nl Ny Nx"]
    kinetic_energy_field: Float[Array, "nl Ny Nx"]
    psi: Float[Array, "nl Ny Nx"]
    u_ageostrophic: Float[Array, "nl Ny Nx"]
    v_ageostrophic: Float[Array, "nl Ny Nx"]

    def invariants(self) -> dict[str, Array]:
        """Mass (tight), total energy, potential enstrophy, PV Casimir.

        Inherited from the underlying SWM diagnostics — the reparameterized
        QG integrates the SWM vector field, so the same flux-form mass
        conservation and implicit-dissipation caveats apply.
        """
        return {
            "mass": jnp.sum(self.mass),
            "total_energy": self.total_energy,
            "potential_enstrophy": self.total_enstrophy,
            "casimir_q3": jnp.sum(self.casimir_q3),
        }


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
    def grid(self) -> CartesianGrid2D:
        return self.swm.grid

    @property
    def mask(self) -> Mask2D | None:
        return self.swm.mask

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
        # grad_perp returns (u@U, v@V) = (-dpsi/dy, dpsi/dx) directly
        u_g, v_g = multilayer(self.diff.grad_perp)(psi)
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

        # Geostrophic velocity via grad_perp: (u@U, v@V)
        u_g, v_g = multilayer(self.diff.grad_perp)(psi)

        # Ageostrophic velocity = total - geostrophic
        u_ageo = u - u_g
        v_ageo = v - v_g

        return ReparamQGDiagnostics(
            energy=swm_diag.energy,
            total_energy=swm_diag.total_energy,
            enstrophy=swm_diag.enstrophy,
            total_enstrophy=swm_diag.total_enstrophy,
            mass=swm_diag.mass,
            casimir_q3=swm_diag.casimir_q3,
            potential_vorticity=swm_diag.potential_vorticity,
            relative_vorticity=swm_diag.relative_vorticity,
            kinetic_energy_field=swm_diag.kinetic_energy_field,
            psi=psi,
            u_ageostrophic=u_ageo,
            v_ageostrophic=v_ageo,
        )

    @staticmethod
    def from_nondimensional(
        *,
        nx: int = 64,
        ny: int = 64,
        rossby: float,
        beta_hat: float,
        burger: Sequence[float],
        thickness_ratio: Sequence[float],
        delta_M: float,
        delta_S: float = 0.0,
        wind_hat: float | None = None,
        aspect: float = 1.0,
        check_resolution: bool = True,
        **create_kw: Any,
    ) -> tuple[ReparameterizedQG, Scales]:
        r"""Build the model from dimensionless numbers instead of SI coefficients.

        Non-dimensional form
        --------------------
        Scale set: **advective** (:meth:`somax.Scales.advective`), with
        ``L = U = 1`` so that ``T = L/U = 1`` and ``f0 = 1/Ro`` — the
        same set :meth:`somax.BarotropicQG.from_nondimensional` uses,
        plus stratification.

        =================  ===============================  ======================
        Input              Definition                       ``create()`` kwarg
        =================  ===============================  ======================
        ``rossby``         :math:`Ro = U/(f_0 L)`           ``f0 = 1/Ro``
        ``beta_hat``       :math:`\beta L^2/U`               ``beta``
        ``burger``         :math:`g'_k H_k/(f_0 L)^2`       ``g_prime``
        ``delta_M``        :math:`(\nu/\beta)^{1/3}/L`       ``lateral_viscosity``
        ``delta_S``        :math:`\kappa/(\beta L)`           ``bottom_drag``
        =================  ===============================  ======================

        ``burger`` is the **per-interface** Burger number, which inverts
        to the stratification one-to-one. It is not the per-mode value:
        the deformation radius of vertical mode ``m`` is a combination
        of the interface numbers, and the built model reports the
        resulting radii as ``model.modal.rossby_radii`` — already in
        units of ``L``, so directly comparable with ``dx``.

        Args:
            nx: Interior cells in x.
            ny: Interior cells in y.
            rossby: Rossby number. Sets ``f0 = 1/Ro``.
            beta_hat: Dimensionless planetary vorticity gradient.
            burger: Per-interface Burger numbers, top to bottom.
            thickness_ratio: Layer thicknesses relative to the depth
                scale. Same length as ``burger``.
            delta_M: Munk-layer width as a fraction of ``L``.
            delta_S: Stommel-layer width as a fraction of ``L``.
            wind_hat: Dimensionless wind amplitude
                :math:`\\tau_0 L/(U^2 H_1)`. One power of ``L``, not two:
                this model's right-hand side is
                :class:`MultilayerShallowWater2D`'s *momentum* equation,
                whose tendency scale is :math:`U^2/L`. The :math:`L^2`
                group belongs to :class:`BaroclinicQG`, which advances a
                PV tendency instead. Defaults to ``beta_hat``, the
                Sverdrup-balanced value.
            aspect: ``Ly / Lx``; the domain is ``Lx = 1``.
            check_resolution: Run the Munk, Stommel and deformation
                radius guards.
            **create_kw: Forwarded to :meth:`create`.

        Returns:
            ``(model, scales)`` with ``scales.kind == "advective"``.

        Raises:
            ValueError: If an input is out of range.
            AssertionFailedError: If ``check_resolution`` is set and the
                grid cannot resolve a boundary layer or the first
                internal deformation radius.
        """
        context = "ReparameterizedQG.from_nondimensional"
        reject_derived_kwargs(
            context,
            (
                "Lx",
                "Ly",
                "f0",
                "beta",
                "n_layers",
                "H",
                "g_prime",
                "stratification",
                "lateral_viscosity",
                "bottom_drag",
                "wind_amplitude",
            ),
            **create_kw,
        )
        require_positive(context, rossby=rossby, beta_hat=beta_hat, aspect=aspect)
        require_non_negative(context, delta_M=delta_M, delta_S=delta_S)
        if wind_hat is not None:
            # Only when supplied: the default is beta_hat, already
            # checked. Left unvalidated, a non-finite value was
            # copied straight into wind_amplitude and produced
            # non-finite tendencies on the first step.
            require_non_negative(context, wind_hat=wind_hat)
        f0 = 1.0 / rossby
        g_prime = burger_to_g_prime(context, burger, thickness_ratio, f0=f0, length=1.0)
        thickness = tuple(float(h) for h in thickness_ratio)

        model = ReparameterizedQG.create(
            nx=nx,
            ny=ny,
            Lx=1.0,
            Ly=aspect,
            f0=f0,
            beta=beta_hat,
            n_layers=len(thickness),
            H=thickness,
            g_prime=g_prime,
            lateral_viscosity=delta_M**3 * beta_hat,
            bottom_drag=delta_S * beta_hat,
            # The RHS applies the wind as tau0 * F / H[0], so the
            # dimensionless group tau_hat = tau0 L / (U**2 H_1) maps to
            # a kwarg carrying the top-layer thickness. (At unit scales
            # the two spellings of the group coincide; the power of L
            # matters when converting a dimensional configuration.)
            wind_amplitude=(beta_hat if wind_hat is None else wind_hat) * thickness[0],
            **create_kw,
        )
        # g is the *first interface's* reduced gravity, not standard
        # gravity: these scales describe the nondimensional model,
        # where the surface-mode gravity is g_prime[0]. Leaving it
        # at 9.81 would make scales.burger disagree with burger[0]
        # and put the thickness-anomaly scale f0 U L / g out by the
        # ratio between them.
        scales = Scales.advective(L=1.0, U=1.0, f0=f0, H=thickness[0], g=g_prime[0])

        if check_resolution:
            from somax._src.core.resolution import (
                check_deformation_radius,
                check_munk_width,
                check_stommel_width,
            )

            if delta_M > 0.0:
                check_munk_width(None, model)
            if delta_S > 0.0:
                check_stommel_width(None, model)
            check_deformation_radius(None, model)

        return model, scales

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
        bc: str = "wall",
        method: str = "upwind1",
        poisson_bc: str = "dst",
        mask: Mask2D | None = None,
    ) -> ReparameterizedQG:
        """Convenience factory for the reparameterized QG model.

        Accepts all arguments of ``MultilayerShallowWater2D.create()``
        plus ``poisson_bc`` for the Helmholtz solver.

        Note:
            The geostrophic projection uses a Dirichlet (DST) Helmholtz
            solver and ``zero_boundaries``, so only wall/Dirichlet BCs
            are consistent. Periodic BCs are not supported.

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
            bc: Boundary condition type (must be ``"wall"``).
            method: Advection reconstruction method.
            poisson_bc: Spectral solver BC type for Helmholtz.
            mask: Optional Arakawa C-grid mask (``None`` = all-ocean).

        Returns:
            A ``ReparameterizedQG`` model instance.

        Raises:
            ValueError: If ``bc`` is not ``"wall"``.
        """
        if bc != "wall":
            msg = (
                f"ReparameterizedQG requires wall BCs (got bc={bc!r}). "
                "The geostrophic projection uses Dirichlet Helmholtz "
                "inversion which is incompatible with periodic BCs."
            )
            raise ValueError(msg)

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
            mask=mask,
        )

        helmholtz_lambdas = f0**2 * swm.modal.eigenvalues

        return ReparameterizedQG(
            swm=swm,
            helmholtz_lambdas=helmholtz_lambdas,
            poisson_bc=poisson_bc,
        )
