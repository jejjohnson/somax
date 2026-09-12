"""Barotropic quasi-geostrophic flow on a sphere."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from finitevolx import (
    Interpolation2D,
    Mask2D,
    SphericalAdvection2D,
    SphericalDifference2D,
    SphericalDiffusion2D,
    SphericalGrid2D,
    SphericalLaplacian2D,
    solve_cg,
    spherical_area_weights,
)
from finitevolx._src.utils.constants import OMEGA, R_EARTH
from jaxtyping import Array, Float, PyTree

from somax._src.core.model import SomaxModel
from somax._src.core.types import Diagnostics, Params, PhysConsts, State
from somax._src.models.spherical._geometry import (
    _require_global_longitude,
    _require_open_poles,
)


class SphericalQGState(State):
    """State for the spherical barotropic QG model.

    Args:
        q: Relative vorticity at T-points, shape ``(Ny, Nx)``. The
            streamfunction is recovered on the fly by inverting the
            spherical Laplacian, so ``psi`` is not carried.
    """

    q: Float[Array, "Ny Nx"]


class SphericalQGParams(Params):
    """Differentiable parameters for the spherical QG model.

    Args:
        lateral_viscosity: Harmonic viscosity coefficient nu (m^2/s).
        bottom_drag: Linear bottom drag coefficient kappa (1/s).
        wind_amplitude: Wind stress curl amplitude (1/s^2).
    """

    lateral_viscosity: Array
    bottom_drag: Array
    wind_amplitude: Array


class SphericalQGPhysConsts(PhysConsts):
    """Frozen physical constants for the spherical QG model.

    Args:
        omega: Planet angular velocity (rad/s).
        radius: Planet radius (m).
    """

    omega: float = eqx.field(static=True, default=OMEGA)
    radius: float = eqx.field(static=True, default=R_EARTH)


class SphericalQGDiagnostics(Diagnostics):
    """Diagnostics for the spherical QG model.

    Args:
        kinetic_energy: Area-integrated kinetic energy.
        enstrophy: Area-integrated enstrophy, the ``q^2`` moment.
        circulation: Area-integrated relative vorticity.
        streamfunction: Recovered streamfunction at T-points.
        relative_vorticity: The state's vorticity field.
    """

    kinetic_energy: Array
    enstrophy: Array
    circulation: Array
    streamfunction: Float[Array, "Ny Nx"]
    relative_vorticity: Float[Array, "Ny Nx"]

    def invariants(self) -> dict[str, Array]:
        """Kinetic energy, enstrophy and circulation."""
        return {
            "kinetic_energy": self.kinetic_energy,
            "enstrophy": self.enstrophy,
            "circulation": self.circulation,
        }


class SphericalQG(SomaxModel):
    r"""Barotropic quasi-geostrophic flow on a sphere.

    Advects absolute vorticity ``q + f`` by the non-divergent flow
    recovered from the streamfunction::

        dq/dt = -adv_sphere(q + f, u, v) + nu lap(q) - kappa q + tau curl

    with ``(u, v) = (-1/R dpsi/dlat, 1/(R cos(phi)) dpsi/dlon)`` and
    ``psi`` from ``lap_sphere(psi) = q``.

    The planetary vorticity gradient is not a constant here. Advecting
    the *absolute* vorticity ``q + f(phi)`` with ``f = 2 Omega sin(phi)``
    reproduces ``beta = 2 Omega cos(phi)/R`` implicitly, so it varies
    from its maximum at the equator to zero at the poles rather than
    being frozen at a reference latitude.

    Inversion is iterative. The spherical Laplacian is not diagonal in
    any transform a lat-lon grid affords — the ``cos(phi)`` metric
    couples latitudes — so the DST route the Cartesian model uses does
    not apply, and the elliptic problem is solved by conjugate
    gradients against the ``SphericalLaplacian2D`` operator.

    Args:
        params: Differentiable parameters.
        consts: Frozen physical constants.
        grid: Spherical Arakawa C-grid.
        diff: Spherical difference operators.
        interp: Interpolation operators.
        laplacian: Spherical Laplacian, used both in the RHS and as the
            operator the inversion solves against.
        advection: Spherical scalar advection.
        diffusion: Spherical harmonic diffusion.
        mask: Optional land/ocean mask (``None`` = all-ocean).
        f_field: Precomputed Coriolis field at T-points.
        wind_forcing: Normalised wind-stress-curl pattern.
        method: Advection reconstruction method.
        cg_tol: Convergence tolerance for the PV inversion, used for
            both the relative and the absolute criterion. The default
            is chosen for float32: a tighter absolute tolerance never
            trips, and CG runs to its step cap.
        cg_max_steps: Iteration cap for the PV inversion.
    """

    params: SphericalQGParams
    consts: SphericalQGPhysConsts = eqx.field(static=True)
    # Not static: unlike a Cartesian grid, which is all floats, a
    # spherical grid carries cos(lat) and coordinate arrays. Marking
    # it static would hand equinox JAX arrays as hashable statics.
    grid: SphericalGrid2D
    diff: SphericalDifference2D
    interp: Interpolation2D
    laplacian: SphericalLaplacian2D
    advection: SphericalAdvection2D
    diffusion: SphericalDiffusion2D
    mask: Mask2D | None
    f_field: Float[Array, "Ny Nx"]
    wind_forcing: Float[Array, "Ny Nx"]
    method: str = eqx.field(static=True, default="upwind1")
    cg_tol: float = eqx.field(static=True, default=1e-6)
    cg_max_steps: int = eqx.field(static=True, default=500)

    def _expand(self, interior: Float[Array, "Ny_i Nx_i"]) -> Float[Array, "Ny Nx"]:
        """Place interior unknowns into a full field with its ghost ring.

        Longitude wraps; the polar ghost rows stay zero, which is the
        Dirichlet condition that makes the elliptic problem well posed.
        """
        full = jnp.zeros((self.grid.Ny, self.grid.Nx)).at[1:-1, 1:-1].set(interior)
        return full.at[:, 0].set(full[:, -2]).at[:, -1].set(full[:, 1])

    def invert_pv(self, q: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """Recover the streamfunction from vorticity: ``lap(psi) = q``.

        Three things have to be arranged before conjugate gradients
        will touch this problem, and each was found by it refusing to:

        **The unknowns are the interior only,** with ``psi = 0`` on the
        polar ghost rows and longitude wrapping. A Laplacian over the
        whole array has zero rows on the ghost ring and a constant null
        space; the Dirichlet condition removes both, and it is the
        physically right one — no flow through the polar walls, with the
        gauge pinned.

        **The operator is area-weighted.** The spherical Laplacian is
        self-adjoint in the area-weighted inner product, not the
        Euclidean one CG assumes: measured directly, the bare operator
        is 15% asymmetric, while ``area * lap`` is symmetric to machine
        precision and negative definite. CG on the bare operator does
        not converge to anything.

        **The right-hand side is normalised.** With an Earth-radius
        grid the weighted residual sits around ``1e-13``, and CG's
        internal inner products then underflow float32. Solving the
        unit-norm system and rescaling afterwards is exact, since the
        operator is linear.

        Args:
            q: Relative vorticity at T-points.

        Returns:
            Streamfunction at T-points, zero on the polar boundary.
        """
        interior = (slice(1, -1), slice(1, -1))
        weights = spherical_area_weights(self.grid)[interior]

        def operator(
            psi_interior: Float[Array, "Ny_i Nx_i"],
        ) -> Float[Array, "Ny_i Nx_i"]:
            return weights * self.laplacian(self._expand(psi_interior))[interior]

        rhs = weights * q[interior]
        scale = jnp.linalg.norm(rhs)
        # A zero right-hand side has psi = 0; dividing by its norm would
        # be 0/0, and CG would be handed NaNs.
        safe_scale = jnp.where(scale == 0.0, 1.0, scale)
        solution, _ = solve_cg(
            operator,
            rhs / safe_scale,
            rtol=self.cg_tol,
            atol=self.cg_tol,
            max_steps=self.cg_max_steps,
        )
        return self._expand(solution * scale)

    def velocities(
        self, psi: Float[Array, "Ny Nx"]
    ) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
        """Non-divergent velocity from the streamfunction.

        ``u = -(1/R) dpsi/dlat`` at U-points and
        ``v = (1/(R cos(phi))) dpsi/dlon`` at V-points; both metric
        factors are supplied by the spherical difference operators.

        Args:
            psi: Streamfunction at T-points.

        Returns:
            ``(u, v)`` on their staggered points.
        """
        u = -self.interp.V_to_U(self.diff.diff_lat_T_to_V(psi))
        v = self.interp.U_to_V(self.diff.diff_lon_T_to_U(psi))
        return u, v

    def vector_field(
        self, t: float, state: PyTree, args: PyTree | None = None
    ) -> SphericalQGState:
        """Compute the vorticity tendency."""
        q = state.q
        nu = self.params.lateral_viscosity
        kappa = self.params.bottom_drag
        tau0 = self.params.wind_amplitude

        psi = self.invert_pv(q)
        u, v = self.velocities(psi)

        # Advect absolute vorticity: this is where beta comes from.
        absolute = q + self.f_field
        dq_dt = self.advection(absolute, u, v, method=self.method)

        dq_dt = dq_dt + tau0 * self.wind_forcing
        dq_dt = dq_dt + self.diffusion(q, nu)
        dq_dt = dq_dt - kappa * q

        # Same reason as the shallow-water model: the polar rows carry
        # a Dirichlet condition that the integrator does not reapply to
        # the returned state, so the tendency there has to be zero for
        # the condition to survive a step.
        dq_dt = dq_dt.at[0, :].set(0.0).at[-1, :].set(0.0)
        return SphericalQGState(q=dq_dt)

    def apply_boundary_conditions(self, state: PyTree) -> SphericalQGState:
        """Periodic in longitude, zero vorticity on the polar edges."""
        q = state.q
        q = q.at[:, 0].set(q[:, -2]).at[:, -1].set(q[:, 1])
        q = q.at[0, :].set(0.0).at[-1, :].set(0.0)
        return SphericalQGState(q=q)

    def diagnose(self, state: PyTree) -> SphericalQGDiagnostics:
        """Area-weighted energy, enstrophy and circulation."""
        q = state.q
        interior = (slice(1, -1), slice(1, -1))
        area = spherical_area_weights(self.grid)[interior]

        psi = self.invert_pv(q)
        u, v = self.velocities(psi)

        u_on_T = self.interp.U_to_T(u)
        v_on_T = self.interp.V_to_T(v)
        ke = 0.5 * jnp.sum(area * (u_on_T[interior] ** 2 + v_on_T[interior] ** 2))
        enstrophy = 0.5 * jnp.sum(area * q[interior] ** 2)
        circulation = jnp.sum(area * q[interior])

        return SphericalQGDiagnostics(
            kinetic_energy=ke,
            enstrophy=enstrophy,
            circulation=circulation,
            streamfunction=psi,
            relative_vorticity=q,
        )

    @staticmethod
    def create(
        nx: int = 128,
        ny: int = 64,
        lon_range: tuple[float, float] = (0.0, 360.0),
        lat_range: tuple[float, float] = (-80.0, 80.0),
        radius: float = R_EARTH,
        omega: float = OMEGA,
        lateral_viscosity: float = 0.0,
        bottom_drag: float = 0.0,
        wind_amplitude: float = 0.0,
        wind_profile: str = "zonal",
        method: str = "upwind1",
        mask: Mask2D | None = None,
        cg_tol: float = 1e-6,
        cg_max_steps: int = 500,
        wind_forcing: Float[Array, "Ny Nx"] | None = None,
    ) -> SphericalQG:
        """Convenience factory.

        Args:
            nx: Interior cells in longitude.
            ny: Interior cells in latitude.
            lon_range: ``(lon_min, lon_max)`` in degrees.
            lat_range: ``(lat_min, lat_max)`` in degrees.
            radius: Planet radius (m).
            omega: Planet angular velocity (rad/s).
            lateral_viscosity: Harmonic viscosity (m^2/s).
            bottom_drag: Linear bottom drag (1/s).
            wind_amplitude: Wind-stress-curl amplitude (1/s^2).
            wind_profile: ``"zonal"`` gives a ``-sin(2 lat)`` curl;
                ``"none"`` gives no pattern.
            method: Advection reconstruction method.
            mask: Optional land/ocean mask.
            cg_tol: Convergence tolerance for the PV inversion, used for
            both the relative and the absolute criterion. The default
            is chosen for float32: a tighter absolute tolerance never
            trips, and CG runs to its step cap.
            cg_max_steps: Iteration cap for the PV inversion.
            wind_forcing: Precomputed wind-stress-curl pattern on the
                model grid, overriding ``wind_profile``. For a scenario
                that supplies its own forcing field.

        Returns:
            A ``SphericalQG`` instance.
        """
        context = "SphericalQG.create"
        _require_global_longitude(context, lon_range)
        _require_open_poles(context, lat_range)
        grid = SphericalGrid2D.from_interior(nx, ny, lon_range, lat_range, R=radius)

        params = SphericalQGParams(
            lateral_viscosity=jnp.asarray(lateral_viscosity),
            bottom_drag=jnp.asarray(bottom_drag),
            wind_amplitude=jnp.asarray(wind_amplitude),
        )
        consts = SphericalQGPhysConsts(omega=omega, radius=radius)

        lat_T = grid.lat_T
        f_field = 2.0 * omega * jnp.sin(lat_T)
        if wind_forcing is not None:
            wind_forcing = jnp.asarray(wind_forcing)
        elif wind_profile == "none":
            wind_forcing = jnp.zeros_like(lat_T)
        else:
            wind_forcing = -jnp.sin(2.0 * lat_T)

        return SphericalQG(
            params=params,
            consts=consts,
            grid=grid,
            diff=SphericalDifference2D(grid=grid, mask=mask),
            interp=Interpolation2D(grid=grid, mask=mask),
            laplacian=SphericalLaplacian2D(grid=grid, mask=mask),
            advection=SphericalAdvection2D(grid=grid, mask=mask),
            diffusion=SphericalDiffusion2D(grid=grid, mask=mask),
            mask=mask,
            f_field=f_field,
            wind_forcing=wind_forcing,
            method=method,
            cg_tol=cg_tol,
            cg_max_steps=cg_max_steps,
        )
