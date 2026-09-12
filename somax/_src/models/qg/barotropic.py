"""Barotropic quasi-geostrophic model."""

from __future__ import annotations

from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
from finitevolx import (
    CartesianGrid2D,
    Difference2D,
    Interpolation2D,
    Mask2D,
    arakawa_jacobian,
    streamfunction_from_vorticity,
    zero_boundaries,
)
from jaxtyping import Array, Float, PyTree

from somax._src.core.model import SomaxModel
from somax._src.core.scales import Scales
from somax._src.core.types import (
    Diagnostics,
    Params,
    PhysConsts,
    State,
    as_parameter,
)
from somax._src.models._nondim import (
    require_non_negative,
    require_positive,
)


class BarotropicQGState(State):
    """State for the barotropic quasi-geostrophic model.

    Args:
        q: PV anomaly (relative vorticity, zeta = nabla^2 psi)
            on T-points, shape ``(Ny, Nx)``. The total PV is
            ``q + beta * y``; the inversion solves ``nabla^2 psi = q``.
    """

    q: Float[Array, "Ny Nx"]

    # QG potential vorticity is a T-point field, not a corner one.
    mask_locations: ClassVar[dict[str, str]] = {"q": "h"}


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

    def invariants(self) -> dict[str, Array]:
        """Energy and enstrophy — the barotropic QG quadratic invariants.

        The Arakawa Jacobian used for advection conserves both energy and
        enstrophy in the inviscid, unforced limit (up to time-truncation
        error), so for that configuration these drive close to zero; with
        viscosity/forcing they quantify the budget instead.
        """
        return {
            "kinetic_energy": self.kinetic_energy,
            "enstrophy": self.enstrophy,
        }


class BarotropicQG(SomaxModel):
    r"""Barotropic quasi-geostrophic model on an Arakawa C-grid.

    Solves the barotropic QG PV equation::

        dq/dt = -J(ψ, q) + tau0*F_wind - kappa*laplacian(psi) + nu*laplacian(q)

    where:
        - q is the PV anomaly (relative vorticity, nabla^2 psi)
        - total PV is q + beta*y
        - psi is the streamfunction from inversion: nabla^2 psi = q
        - J(psi, q + beta*y) is the Arakawa Jacobian (energy+enstrophy conserving)
        - u = -dpsi/dy, v = dpsi/dx (geostrophic velocity)

    Args:
        params: Differentiable parameters.
        consts: Frozen physical constants.
        grid: 2D Arakawa C-grid.
        diff: Difference operators.
        interp: Interpolation operators.
        mask: Optional Arakawa C-grid mask (``None`` = all-ocean).
        beta_y: Precomputed β·y field.
        wind_forcing: Precomputed wind stress curl pattern (normalised).
        poisson_bc: Spectral solver BC type for PV inversion.
    """

    params: BarotropicQGParams
    consts: BarotropicQGPhysConsts = eqx.field(static=True)
    grid: CartesianGrid2D = eqx.field(static=True)
    diff: Difference2D
    interp: Interpolation2D
    mask: Mask2D | None
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
        cell_area = self.grid.dx * self.grid.dy
        ke = 0.5 * jnp.sum(u_T[s] ** 2 + v_T[s] ** 2) * cell_area
        zeta = self.diff.laplacian(psi)
        enstrophy = 0.5 * jnp.sum(q[s] ** 2) * cell_area

        return BarotropicQGDiagnostics(
            psi=psi,
            u=u,
            v=v,
            kinetic_energy=ke,
            enstrophy=enstrophy,
            relative_vorticity=zeta,
        )

    @staticmethod
    def from_nondimensional(
        *,
        nx: int = 64,
        ny: int = 64,
        rossby: float,
        beta_hat: float,
        delta_M: float,
        delta_S: float = 0.0,
        delta_I: float | None = None,
        aspect: float = 1.0,
        wind_profile: str = "doublegyre",
        mask: Mask2D | None = None,
        check_resolution: bool = True,
    ) -> tuple[BarotropicQG, Scales]:
        r"""Build the model from dimensionless numbers instead of SI coefficients.

        Non-dimensional form
        --------------------
        Scale set: **advective** (:meth:`somax.Scales.advective`), with
        ``L = U = H = 1`` so that ``T = L/U = 1`` and ``f0 = 1/Ro``.
        In these units the barotropic QG equation reads

        .. math::

            \partial_t q + J(\psi,\, q + \hat\beta y)
                = \delta_M^3 \hat\beta\, \nabla^2 q
                - \delta_S \hat\beta\, \nabla^2 \psi
                + \hat\tau\, \mathrm{curl}\,\tau .

        The dimensionless numbers and what each sets:

        ==============  ==========================  =========================
        Input           Definition                  ``create()`` kwarg
        ==============  ==========================  =========================
        ``rossby``      :math:`Ro = U/(f_0 L)`      ``f0 = 1/Ro``
        ``beta_hat``    :math:`\hat\beta = \beta L^2/U`  ``beta``
        ``delta_M``     :math:`(\nu/\beta)^{1/3}/L`   ``lateral_viscosity``
        ``delta_S``     :math:`\kappa/(\beta L)`      ``bottom_drag``
        ``delta_I``     :math:`(U/\beta)^{1/2}/L`     ``wind_amplitude``
        ==============  ==========================  =========================

        Choosing ``delta_M`` and ``delta_S`` rather than ``nu`` and
        ``kappa`` is the point of this factory: the western-boundary-layer
        widths are what has to be resolved, and picking them directly
        replaces tuning two coefficients until a run is both stable and
        resolved.

        Args:
            nx: Interior cells in x.
            ny: Interior cells in y.
            rossby: Rossby number ``Ro = U / (f0 L)``. Sets ``f0 = 1/Ro``.
            beta_hat: Planetary vorticity gradient ``beta L^2 / U``.
            delta_M: Munk-layer width as a fraction of ``L``.
            delta_S: Stommel-layer width as a fraction of ``L``. Zero
                (the default) means no bottom drag.
            delta_I: Inertial-layer width as a fraction of ``L``. This
                fixes the wind amplitude through the Sverdrup balance
                ``tau0 = delta_I**2 * beta_hat**2``. Left at ``None``,
                the wind is set to ``beta_hat``, the amplitude whose
                Sverdrup interior velocity is exactly the velocity scale
                ``U`` — equivalent to ``delta_I = beta_hat**-0.5``.
            aspect: ``Ly / Lx``. The domain is ``Lx = 1`` by construction.
            wind_profile: Passed through to :meth:`create`.
            mask: Passed through to :meth:`create`.
            check_resolution: Run the Munk and Stommel resolution guards
                and raise if the boundary layers are unresolved. Set
                ``False`` only to build a deliberately coarse model (a
                unit test, say).

        Returns:
            ``(model, scales)``. The ``Scales`` records the unit scale
            set the model *runs in* — ``L = U = H = 1``.

            It is therefore not enough on its own to convert a
            dimensional state: with ``L = U = 1`` the vorticity scale
            is 1 and the time scale is 1, so
            :meth:`somax.StateAffine.from_scales` built from it leaves
            a dimensional ``q`` unchanged. Converting needs the
            *physical* scale set as well — the
            ``Scales.advective(L=..., U=...)`` describing the run being
            reproduced — with this one as the target:

            >>> physical = Scales.advective(L=1.0e6, U=0.05, f0=1.0e-4)
            >>> to_units = StateAffine.from_scales(  # doctest: +SKIP
            ...     BarotropicQGState, physical
            ... )

            Keep both; this return value is the second of the pair.

        Raises:
            ValueError: If a dimensionless input is out of range.
            AssertionFailedError: If ``check_resolution`` is set and a
                boundary layer is unresolved on this grid.

        Example:
            >>> model, scales = BarotropicQG.from_nondimensional(
            ...     nx=128, ny=128, rossby=0.02, beta_hat=50.0,
            ...     delta_M=0.03, delta_S=0.01,
            ... )
            >>> scales.kind
            'advective'
        """
        # Every check is stated as "must be finite and in range" rather
        # than as a one-sided comparison: every comparison with NaN is
        # false, and +inf passes a bare ``> 0``, so the one-sided form
        # lets both through. A NaN delta_M then makes a NaN viscosity,
        # and the resolution guard below falls through too because its
        # ratio compares false against both thresholds — handing back a
        # model that will quietly corrupt a simulation.
        context = "from_nondimensional"
        require_positive(context, rossby=rossby, beta_hat=beta_hat, aspect=aspect)
        require_non_negative(context, delta_M=delta_M, delta_S=delta_S)
        if delta_I is not None:
            require_positive(context, delta_I=delta_I)

        # Unit scales: L = U = H = 1, so f0 = 1/Ro and every coefficient
        # below is already the nondimensional group itself.
        wind_amplitude = beta_hat if delta_I is None else delta_I**2 * beta_hat**2
        model = BarotropicQG.create(
            nx=nx,
            ny=ny,
            Lx=1.0,
            Ly=aspect,
            f0=1.0 / rossby,
            beta=beta_hat,
            lateral_viscosity=delta_M**3 * beta_hat,
            bottom_drag=delta_S * beta_hat,
            wind_amplitude=wind_amplitude,
            wind_profile=wind_profile,
            mask=mask,
        )
        scales = Scales.advective(L=1.0, U=1.0, f0=1.0 / rossby, H=1.0)

        if check_resolution:
            from somax._src.cli._assertions import (
                check_munk_width,
                check_stommel_width,
            )

            check_munk_width(None, model)
            if delta_S > 0.0:
                check_stommel_width(None, model)

        return model, scales

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
        mask: Mask2D | None = None,
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
            mask: Optional Arakawa C-grid mask (``None`` = all-ocean).

        Returns:
            A ``BarotropicQG`` model instance.
        """
        grid = CartesianGrid2D.from_interior(nx, ny, Lx, Ly)
        params = BarotropicQGParams(
            lateral_viscosity=as_parameter(lateral_viscosity),
            bottom_drag=as_parameter(bottom_drag),
            wind_amplitude=as_parameter(wind_amplitude),
        )
        consts = BarotropicQGPhysConsts(f0=f0, beta=beta)
        diff = Difference2D(grid=grid, mask=mask)
        interp = Interpolation2D(grid=grid, mask=mask)

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
            mask=mask,
            beta_y=beta_y,
            wind_forcing=wind_forcing,
            poisson_bc="dst",
        )
