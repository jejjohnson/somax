"""Baroclinic (multilayer) quasi-geostrophic model."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

import equinox as eqx
import jax.numpy as jnp
from finitevolx import (
    CartesianGrid2D,
    Difference2D,
    Interpolation2D,
    Mask2D,
    arakawa_jacobian,
    multilayer,
    pv_inversion,
    zero_boundaries,
)
from jaxtyping import Array, Float, PyTree

from somax._src.core.model import SomaxModel
from somax._src.core.scales import Scales
from somax._src.core.transforms import ModalTransform, StratificationProfile
from somax._src.core.types import (
    Diagnostics,
    Params,
    PhysConsts,
    State,
    as_parameter,
)
from somax._src.models._nondim import (
    burger_to_g_prime,
    reject_derived_kwargs,
    require_non_negative,
    require_positive,
)


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

    # QG potential vorticity is a T-point field, not a corner one.
    mask_locations: ClassVar[dict[str, str]] = {"q": "h"}


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

    def invariants(self) -> dict[str, Array]:
        """Total kinetic energy and total enstrophy across layers.

        Quadratic QG invariants summed over layers; conserved up to
        time-truncation error in the inviscid/unforced limit, otherwise a
        budget signal.
        """
        return {
            "total_kinetic_energy": self.total_kinetic_energy,
            "total_enstrophy": self.total_enstrophy,
        }


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
        mask: Optional Arakawa C-grid mask (``None`` = all-ocean).
        modal: Precomputed modal transform.
        strat: Stratification profile.
        beta_y: Precomputed beta*(y - y0) field.
        wind_forcing: Normalised wind stress curl pattern.
        helmholtz_lambdas: f0^2 * eigenvalues per mode, shape ``(nl,)``.
        poisson_bc: Spectral solver BC type for PV inversion.
    """

    params: BaroclinicQGParams
    consts: BaroclinicQGPhysConsts = eqx.field(static=True)
    grid: CartesianGrid2D = eqx.field(static=True)
    diff: Difference2D
    interp: Interpolation2D
    mask: Mask2D | None
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
    ) -> tuple[BaroclinicQG, Scales]:
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
                :math:`\\tau_0 L^2/(U^2 H_1)`. Defaults to
                ``beta_hat``, the Sverdrup-balanced value.
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
        context = "BaroclinicQG.from_nondimensional"
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

        model = BaroclinicQG.create(
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
            # dimensionless group tau_hat = tau0 L**2 / (U**2 H_1)
            # maps to a kwarg carrying the top-layer thickness.
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
        mask: Mask2D | None = None,
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
            mask: Optional Arakawa C-grid mask (``None`` = all-ocean).

        Returns:
            A ``BaroclinicQG`` model instance.

        Raises:
            ValueError: If ``n_layers``, ``H``, and ``g_prime`` have
                inconsistent lengths (when ``stratification`` is None).
        """
        grid = CartesianGrid2D.from_interior(nx, ny, Lx, Ly)

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
            lateral_viscosity=as_parameter(lateral_viscosity),
            bottom_drag=as_parameter(bottom_drag),
            wind_amplitude=as_parameter(wind_amplitude),
        )
        consts = BaroclinicQGPhysConsts(f0=f0, beta=beta, n_layers=nl)
        diff = Difference2D(grid=grid, mask=mask)
        interp = Interpolation2D(grid=grid, mask=mask)

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
            mask=mask,
            modal=modal,
            strat=strat,
            beta_y=beta_y,
            wind_forcing=wind_forcing,
            helmholtz_lambdas=helmholtz_lambdas,
            poisson_bc=poisson_bc,
        )
