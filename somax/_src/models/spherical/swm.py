"""Single-layer shallow water on a sphere (vector-invariant form)."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from finitevolx import (
    Interpolation2D,
    Mask2D,
    SphericalAdvection2D,
    SphericalDifference2D,
    SphericalDiffusion2D,
    SphericalGrid2D,
    SphericalVorticity2D,
    kinetic_energy,
    spherical_area_weights,
)
from finitevolx._src.utils.constants import OMEGA, R_EARTH
from jaxtyping import Array, Float, PyTree

from somax._src.core.model import SomaxModel
from somax._src.core.scales import Scales
from somax._src.core.types import Diagnostics, Params, PhysConsts, State
from somax._src.models._nondim import (
    require_non_negative,
    require_positive,
    resolve_burger_spherical,
)


class SphericalSWMState(State):
    """State for the spherical shallow water model.

    Args:
        h: Layer thickness at T-points, shape ``(Ny, Nx)``.
        u: Zonal velocity at U-points, shape ``(Ny, Nx)``.
        v: Meridional velocity at V-points, shape ``(Ny, Nx)``.
    """

    h: Float[Array, "Ny Nx"]
    u: Float[Array, "Ny Nx"]
    v: Float[Array, "Ny Nx"]


class SphericalSWMParams(Params):
    """Differentiable parameters for the spherical shallow water model.

    Args:
        lateral_viscosity: Harmonic viscosity coefficient nu (m^2/s).
        bottom_drag: Linear bottom drag coefficient kappa (1/s).
        wind_amplitude: Wind stress amplitude (m/s^2).
    """

    lateral_viscosity: Array
    bottom_drag: Array
    wind_amplitude: Array


class SphericalSWMPhysConsts(PhysConsts):
    """Frozen physical constants for the spherical shallow water model.

    Note the absence of ``f0`` and ``beta``. On a sphere the Coriolis
    parameter is a *field*, ``f(phi) = 2 Omega sin(phi)``, and its
    meridional gradient follows from the geometry rather than being a
    free constant — which is the whole point of leaving the beta plane.

    Args:
        gravity: Gravitational acceleration g (m/s^2).
        omega: Planet angular velocity (rad/s).
        radius: Planet radius (m).
        H0: Mean layer depth (m).
    """

    gravity: float = eqx.field(static=True, default=9.81)
    omega: float = eqx.field(static=True, default=OMEGA)
    radius: float = eqx.field(static=True, default=R_EARTH)
    H0: float = eqx.field(static=True, default=1000.0)


class SphericalSWMDiagnostics(Diagnostics):
    """Diagnostics for the spherical shallow water model.

    Integrals use the spherical cell area ``R^2 cos(lat) dlon dlat``
    rather than a uniform ``dx dy``, so a conservation statement means
    the same thing at every latitude.

    Args:
        energy: Area-integrated total energy.
        enstrophy: Area-integrated potential enstrophy.
        mass: Area-integrated mass.
        casimir_q3: PV Casimir, the ``q^3`` moment.
        potential_vorticity: PV at X-points.
        relative_vorticity: Relative vorticity at X-points.
        kinetic_energy_field: Kinetic energy at T-points.
    """

    energy: Array
    enstrophy: Array
    mass: Array
    casimir_q3: Array
    potential_vorticity: Float[Array, "Ny Nx"]
    relative_vorticity: Float[Array, "Ny Nx"]
    kinetic_energy_field: Float[Array, "Ny Nx"]

    def invariants(self) -> dict[str, Array]:
        """Mass (tight), total energy, potential enstrophy, PV Casimir."""
        return {
            "mass": self.mass,
            "total_energy": self.energy,
            "potential_enstrophy": self.enstrophy,
            "casimir_q3": self.casimir_q3,
        }


class SphericalSWM(SomaxModel):
    r"""Shallow water on a sphere, vector-invariant form.

    Solves the rotating shallow-water equations on a spherical Arakawa
    C-grid::

        dh/dt = -div_sphere(h u)
        du/dt = +q (h v)_bar - (1/(R cos(phi))) dP/dlon + nu lap(u) - kappa u
        dv/dt = -q (h u)_bar - (1/R) dP/dlat            + nu lap(v) - kappa v

    with ``q = (zeta + f)/h`` the potential vorticity and
    ``P = KE + g h`` the Bernoulli potential. Every horizontal
    derivative carries the spherical metric: the ``1/(R cos(phi))``
    factor in longitude and ``1/R`` in latitude, supplied by the
    finitevolx spherical operators.

    Coriolis is the full ``f(phi) = 2 Omega sin(phi)``, not a beta-plane
    expansion about a reference latitude. The planetary vorticity
    gradient ``beta = 2 Omega cos(phi)/R`` is then implicit in the field
    and varies correctly from equator to pole.

    Known limitation
    ----------------
    Mass is conserved only to discretisation accuracy, not to machine
    precision as in the Cartesian :class:`NonlinearShallowWater2D`. The
    spherical flux divergence does not telescope exactly against
    ``spherical_area_weights`` — the cell area it implicitly divides by
    differs from the one that function returns — so a balanced
    solid-body rotation loses of order ``1e-3`` of its mass over six
    hours. The drift is independent of the time step and only weakly
    dependent on resolution, which places it in the spatial operator.
    Tracked upstream as jejjohnson/finitevolX#247; until it is fixed,
    treat the mass diagnostic here as a drift signal rather than a
    conserved quantity.

    Args:
        params: Differentiable parameters.
        consts: Frozen physical constants.
        grid: Spherical Arakawa C-grid.
        diff: Spherical difference operators.
        interp: Interpolation operators (staggering only, metric-free).
        vorticity: Spherical vorticity / PV operator.
        advection: Spherical scalar advection, for the mass equation.
        diffusion: Spherical harmonic diffusion.
        mask: Optional land/ocean mask (``None`` = all-ocean).
        f_field: Precomputed Coriolis field at X-points.
        wind_stress_x: Normalised zonal wind-stress pattern.
        wind_stress_y: Normalised meridional wind-stress pattern.
        method: Advection reconstruction method for the mass equation.
    """

    params: SphericalSWMParams
    consts: SphericalSWMPhysConsts = eqx.field(static=True)
    # Not static: unlike a Cartesian grid, which is all floats, a
    # spherical grid carries cos(lat) and coordinate arrays. Marking
    # it static would hand equinox JAX arrays as hashable statics.
    grid: SphericalGrid2D
    diff: SphericalDifference2D
    interp: Interpolation2D
    vorticity: SphericalVorticity2D
    advection: SphericalAdvection2D
    diffusion: SphericalDiffusion2D
    mask: Mask2D | None
    f_field: Float[Array, "Ny Nx"]
    wind_stress_x: Float[Array, "Ny Nx"]
    wind_stress_y: Float[Array, "Ny Nx"]
    method: str = eqx.field(static=True, default="upwind1")

    def vector_field(
        self, t: float, state: PyTree, args: PyTree | None = None
    ) -> SphericalSWMState:
        """Compute tendencies for the spherical shallow water equations."""
        h, u, v = state.h, state.u, state.v
        g = self.consts.gravity
        nu = self.params.lateral_viscosity
        kappa = self.params.bottom_drag
        tau0 = self.params.wind_amplitude

        # 1. Mass: dh/dt = -div_sphere(h u)
        dh_dt = self.advection(h, u, v, method=self.method)

        # 2. Potential vorticity q = (zeta + f)/h at X-points.
        q = self.vorticity.potential_vorticity(u, v, h, self.f_field)

        # 3. Cross-velocity PV fluxes (vector-invariant Coriolis term).
        h_on_U = self.interp.T_to_U(h)
        h_on_V = self.interp.T_to_V(h)
        uh = h_on_U * u
        vh = h_on_V * v

        q_on_U = self.interp.X_to_U(q)
        q_on_V = self.interp.X_to_V(q)
        vh_on_U = self.interp.V_to_U(vh)
        uh_on_V = self.interp.U_to_V(uh)

        # 4. Bernoulli potential P = KE + g h.
        bernoulli = kinetic_energy(u, v) + g * h

        # 5. Momentum. The pressure gradients carry the spherical metric.
        du_dt = q_on_U * vh_on_U - self.diff.diff_lon_T_to_U(bernoulli)
        dv_dt = -q_on_V * uh_on_V - self.diff.diff_lat_T_to_V(bernoulli)

        # 6. Wind forcing.
        du_dt = du_dt + tau0 * self.wind_stress_x
        dv_dt = dv_dt + tau0 * self.wind_stress_y

        # 7. Lateral diffusion.
        du_dt = du_dt + self.diffusion(u, nu)
        dv_dt = dv_dt + self.diffusion(v, nu)

        # 8. Linear bottom drag.
        du_dt = du_dt - kappa * u
        dv_dt = dv_dt - kappa * v

        return SphericalSWMState(h=dh_dt, u=du_dt, v=dv_dt)

    def apply_boundary_conditions(self, state: PyTree) -> SphericalSWMState:
        """Periodic in longitude, solid walls in latitude.

        Longitude wraps because a lat-lon grid spanning the full globe
        is genuinely periodic there. Latitude does not: the poles are
        coordinate singularities, not a boundary the flow crosses, so
        the meridional velocity is set to zero at the northern and
        southern edges and thickness is given a zero-gradient condition.
        """
        h, u, v = state.h, state.u, state.v

        # Zonal wrap: west ghost takes the last interior column, and
        # vice versa.
        h = h.at[:, 0].set(h[:, -2]).at[:, -1].set(h[:, 1])
        u = u.at[:, 0].set(u[:, -2]).at[:, -1].set(u[:, 1])
        v = v.at[:, 0].set(v[:, -2]).at[:, -1].set(v[:, 1])

        # Meridional walls: no flow through the polar edges.
        v = v.at[0, :].set(0.0).at[-2, :].set(0.0).at[-1, :].set(0.0)
        # Free-slip zonal velocity, zero-gradient thickness.
        u = u.at[0, :].set(u[1, :]).at[-1, :].set(u[-2, :])
        h = h.at[0, :].set(h[1, :]).at[-1, :].set(h[-2, :])

        return SphericalSWMState(h=h, u=u, v=v)

    def diagnose(self, state: PyTree) -> SphericalSWMDiagnostics:
        """Area-weighted invariants plus PV, vorticity and KE fields."""
        g = self.consts.gravity
        h, u, v = state.h, state.u, state.v
        interior = (slice(1, -1), slice(1, -1))

        ke = kinetic_energy(u, v)
        zeta = self.vorticity.relative_vorticity(u, v)
        q = self.vorticity.potential_vorticity(u, v, h, self.f_field)

        # Spherical cell area, not a uniform dx*dy.
        area = spherical_area_weights(self.grid)[interior]
        h_on_X = self.interp.T_to_X(h)

        energy = jnp.sum(area * (ke[interior] + 0.5 * g * h[interior] ** 2))
        enstrophy = 0.5 * jnp.sum(area * q[interior] ** 2 * h_on_X[interior])
        mass = jnp.sum(area * h[interior])
        casimir_q3 = jnp.sum(area * q[interior] ** 3 * h_on_X[interior])

        return SphericalSWMDiagnostics(
            energy=energy,
            enstrophy=enstrophy,
            mass=mass,
            casimir_q3=casimir_q3,
            potential_vorticity=q,
            relative_vorticity=zeta,
            kinetic_energy_field=ke,
        )

    @staticmethod
    def from_nondimensional(
        *,
        nx: int = 128,
        ny: int = 64,
        rossby: float,
        burger: float | None = None,
        froude: float | None = None,
        lamb: float | None = None,
        ekman: float = 0.0,
        ekman_lateral: float = 0.0,
        wind_hat: float = 0.0,
        lat_range: tuple[float, float] = (-80.0, 80.0),
        lon_range: tuple[float, float] = (0.0, 360.0),
        check_resolution: bool = True,
        **create_kw: Any,
    ) -> tuple[SphericalSWM, Scales]:
        r"""Build the model from dimensionless numbers instead of SI coefficients.

        Non-dimensional form
        --------------------
        Scale set: **planetary** (:meth:`somax.Scales.planetary`), with
        ``a = Omega = H = 1``. The planet radius is the length scale and
        the rotation period the time scale, so ``T = 1/Omega = 1`` and
        ``f0 = 2 Omega = 2``; the velocity scale follows from the Rossby
        number as ``U = 2 Omega a Ro = 2 Ro``.

        The factor of two is not cosmetic. On a sphere the Coriolis
        parameter is ``f(phi) = 2 Omega sin(phi)``, so the quantity that
        plays the role of a Cartesian ``f0`` is ``2 Omega``, and the
        conventional spherical Rossby number is ``U/(2 Omega a)``.
        Writing it that way keeps :attr:`Scales.rossby` the same
        ``U/(f0 L)`` it is for every other family, at the cost of the
        twos that appear in the table below.

        ==================  ==================================  ====================
        Input               Definition                          ``create()`` kwarg
        ==================  ==================================  ====================
        ``rossby``          :math:`Ro = U/(2\Omega a)`           sets ``U``
        ``burger``          :math:`Bu = gH/(2\Omega a)^2`        ``g = 4Bu``
        ``froude``          :math:`Fr = U/\sqrt{gH}`             ``g`` (via ``Bu``)
        ``lamb``            :math:`\varepsilon = 1/Bu`           ``g = 4/\varepsilon``
        ``ekman``           :math:`\kappa/(2\Omega)`             ``bottom_drag``
        ``ekman_lateral``   :math:`\nu/(2\Omega a^2)`            ``lateral_viscosity``
        ``wind_hat``        :math:`\tau_0/(2\Omega U)`           ``wind_amplitude``
        ==================  ==================================  ====================

        The latitude range stays in degrees: it is a geometric choice,
        not a scale, and a nondimensional sphere is still a sphere.

        Args:
            nx: Interior cells in longitude.
            ny: Interior cells in latitude.
            rossby: Rossby number; sets the velocity scale.
            burger: Burger number. Give exactly one of this,
                ``froude`` or ``lamb``.
            froude: Froude number. Equivalent through
                ``Bu = (Ro/Fr)**2``.
            lamb: Lamb parameter ``eps = 4 Omega**2 a**2/(gH)``, the
                inverse Burger number and the usual spelling in the
                spherical literature.
            ekman: Linear bottom-drag Ekman number.
            ekman_lateral: Lateral-viscosity Ekman number.
            wind_hat: Dimensionless wind acceleration.
            lat_range: ``(lat_min, lat_max)`` in degrees.
            lon_range: ``(lon_min, lon_max)`` in degrees.
            check_resolution: Run the equatorial-deformation-radius
                guard and raise if the grid cannot span it. Set
                ``False`` only to build a deliberately coarse model.
            **create_kw: Forwarded to :meth:`create` (``wind_profile``,
                ``method``, ``mask``).

        Returns:
            ``(model, scales)`` with ``scales.kind == "planetary"``.

        Raises:
            ValueError: If an input is out of range, or if not exactly
                one of ``burger``, ``froude`` and ``lamb`` is given.
            AssertionFailedError: If ``check_resolution`` is set and the
                equatorial deformation radius is unresolved.

        Example:
            >>> model, scales = SphericalSWM.from_nondimensional(
            ...     nx=128, ny=64, rossby=0.05, lamb=10.0,
            ... )
            >>> scales.kind
            'planetary'
        """
        context = "SphericalSWM.from_nondimensional"
        require_positive(context, rossby=rossby)
        require_non_negative(
            context,
            ekman=ekman,
            ekman_lateral=ekman_lateral,
            wind_hat=wind_hat,
        )
        bu = resolve_burger_spherical(context, burger, froude, lamb, rossby)

        # Unit scales: a = Omega = H0 = 1, so f0 = 2 and U = 2 Ro. Rate
        # coefficients carry one factor of f0; the wind is an
        # acceleration and so carries f0 * U.
        f0 = 2.0
        velocity = f0 * rossby
        model = SphericalSWM.create(
            nx=nx,
            ny=ny,
            lon_range=lon_range,
            lat_range=lat_range,
            radius=1.0,
            omega=1.0,
            g=f0**2 * bu,
            H0=1.0,
            lateral_viscosity=f0 * ekman_lateral,
            bottom_drag=f0 * ekman,
            wind_amplitude=wind_hat * f0 * velocity,
            **create_kw,
        )
        scales = Scales.planetary(a=1.0, Omega=1.0, H=1.0, rossby=rossby, g=f0**2 * bu)

        if check_resolution:
            from somax._src.cli._assertions import (
                check_equatorial_deformation_radius,
            )

            check_equatorial_deformation_radius(None, model)

        return model, scales

    @staticmethod
    def create(
        nx: int = 128,
        ny: int = 64,
        lon_range: tuple[float, float] = (0.0, 360.0),
        lat_range: tuple[float, float] = (-80.0, 80.0),
        radius: float = R_EARTH,
        omega: float = OMEGA,
        g: float = 9.81,
        H0: float = 1000.0,
        lateral_viscosity: float = 0.0,
        bottom_drag: float = 0.0,
        wind_amplitude: float = 0.0,
        wind_profile: str = "zonal",
        method: str = "upwind1",
        mask: Mask2D | None = None,
    ) -> SphericalSWM:
        """Convenience factory.

        Args:
            nx: Interior cells in longitude.
            ny: Interior cells in latitude.
            lon_range: ``(lon_min, lon_max)`` in degrees.
            lat_range: ``(lat_min, lat_max)`` in degrees.
            radius: Planet radius (m).
            omega: Planet angular velocity (rad/s).
            g: Gravitational acceleration (m/s^2).
            H0: Mean layer depth (m).
            lateral_viscosity: Harmonic viscosity (m^2/s).
            bottom_drag: Linear bottom drag (1/s).
            wind_amplitude: Wind stress amplitude (m/s^2).
            wind_profile: ``"zonal"`` gives a ``-cos(2 lat)`` zonal
                stress (trades and westerlies); ``"none"`` gives no
                pattern.
            method: Advection reconstruction for the mass equation.
            mask: Optional land/ocean mask.

        Returns:
            A ``SphericalSWM`` instance.
        """
        grid = SphericalGrid2D.from_interior(nx, ny, lon_range, lat_range, R=radius)

        params = SphericalSWMParams(
            lateral_viscosity=jnp.asarray(lateral_viscosity),
            bottom_drag=jnp.asarray(bottom_drag),
            wind_amplitude=jnp.asarray(wind_amplitude),
        )
        consts = SphericalSWMPhysConsts(gravity=g, omega=omega, radius=radius, H0=H0)

        # Coriolis on X-points, where the PV lives: f = 2 Omega sin(lat).
        f_field = 2.0 * omega * jnp.sin(_lat_at_corner(grid))

        lat_T = grid.lat_T
        if wind_profile == "none":
            wind_stress_x = jnp.zeros_like(lat_T)
        else:
            # Trades / westerlies: eastward at mid-latitudes, westward
            # at the equator and poles.
            wind_stress_x = -jnp.cos(2.0 * lat_T)
        wind_stress_y = jnp.zeros_like(lat_T)

        return SphericalSWM(
            params=params,
            consts=consts,
            grid=grid,
            diff=SphericalDifference2D(grid=grid, mask=mask),
            interp=Interpolation2D(grid=grid, mask=mask),
            vorticity=SphericalVorticity2D(grid=grid, mask=mask),
            advection=SphericalAdvection2D(grid=grid, mask=mask),
            diffusion=SphericalDiffusion2D(grid=grid, mask=mask),
            mask=mask,
            f_field=f_field,
            wind_stress_x=wind_stress_x,
            wind_stress_y=wind_stress_y,
            method=method,
        )


def _lat_at_corner(grid: SphericalGrid2D) -> Float[Array, "Ny Nx"]:
    """Latitude at X-points (NE corners), half a cell north of T."""
    return grid.lat_T + 0.5 * grid.dlat
