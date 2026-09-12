"""2D linear convection: du/dt + cx du/dx + cy du/dy = 0."""

from __future__ import annotations

import math
from typing import Any, ClassVar

import equinox as eqx
import jax.numpy as jnp
from finitevolx import (
    CartesianGrid2D,
    Difference2D,
    Interpolation2D,
    Mask2D,
    enforce_periodic,
)
from jaxtyping import Array, PyTree

from somax._src.core.model import SomaxModel
from somax._src.core.scales import Scales
from somax._src.core.types import Diagnostics, Params, State, as_parameter
from somax._src.models._nondim import require_positive


class LinearConvection2DParams(Params):
    """Differentiable parameters for 2D linear convection.

    Args:
        cx: Wave speed in x-direction.
        cy: Wave speed in y-direction.
    """

    cx: Array
    cy: Array


class LinearConvection2DState(State):
    """State for 2D linear convection.

    Args:
        u: Scalar field on T-points, shape ``(Ny, Nx)`` including ghost cells.
    """

    u: Array

    # ``u`` here is a T-point scalar, not a C-grid velocity.
    mask_locations: ClassVar[dict[str, str]] = {"u": "h"}


class LinearConvection2DDiagnostics(Diagnostics):
    """Diagnostics for 2D linear convection.

    Args:
        energy: Integrated energy.
    """

    energy: Array


class LinearConvection2D(SomaxModel):
    """2D linear convection on an Arakawa C-grid.

    Solves ``du/dt + cx * du/dx + cy * du/dy = 0``.

    Args:
        params: Differentiable parameters (wave speeds ``cx``, ``cy``).
        grid: 2D Arakawa C-grid.
        diff: Difference operators.
        interp: Interpolation operators.
        mask: Optional Arakawa C-grid mask (``None`` = all-ocean).
    """

    params: LinearConvection2DParams
    grid: CartesianGrid2D = eqx.field(static=True)
    diff: Difference2D
    interp: Interpolation2D
    mask: Mask2D | None

    def vector_field(
        self, t: float, state: PyTree, args: PyTree | None = None
    ) -> LinearConvection2DState:
        """Compute tendency: du/dt = -cx * du/dx - cy * du/dy."""
        # Flux in x: c_x * u at U-points, then divergence back to T
        flux_x = self.params.cx * self.interp.T_to_U(state.u)
        # Flux in y: c_y * u at V-points, then divergence back to T
        flux_y = self.params.cy * self.interp.T_to_V(state.u)
        du_dt = -(self.diff.diff_x_U_to_T(flux_x) + self.diff.diff_y_V_to_T(flux_y))
        return LinearConvection2DState(u=du_dt)

    def apply_boundary_conditions(self, state: PyTree) -> LinearConvection2DState:
        """Apply periodic boundary conditions."""
        return LinearConvection2DState(u=enforce_periodic(state.u))

    def diagnose(self, state: PyTree) -> LinearConvection2DDiagnostics:
        """Compute energy diagnostic."""
        interior = state.u[1:-1, 1:-1]
        energy = 0.5 * jnp.sum(interior**2) * self.grid.dx * self.grid.dy
        return LinearConvection2DDiagnostics(energy=energy)

    @staticmethod
    def from_nondimensional(
        *,
        nx: int = 64,
        ny: int = 64,
        aspect: float = 1.0,
        direction: tuple[float, float] = (1.0, 1.0),
        **create_kw: Any,
    ) -> tuple[LinearConvection2D, Scales]:
        r"""Build the model at unit scales instead of SI coefficients.

        Non-dimensional form
        --------------------
        Advective scale set (:meth:`somax.Scales.advective`) with
        ``L = 1`` and the wave speed as the velocity scale, so ``T = 1``
        and the equation reads ``d_t u + c.grad u = 0`` with ``|c| = 1``.

        Scaling out the *speed* leaves no free dimensionless number,
        but it does not remove the *direction*: ``(1, 0)``, ``(1, 1)``
        and ``(-1, 2)`` are different problems on the same grid, and
        only their common magnitude is a unit choice. ``direction``
        carries that, normalised so the speed stays 1. The Courant
        number is a time-step choice rather than a property of the
        problem, so use ``scales.dt_from_cfl``.


        Args:
            nx: Interior cells in x.
            ny: Interior cells in y.
            aspect: ``Ly / Lx``; the domain is ``Lx = 1``.
            direction: Propagation direction ``(cx, cy)``, normalised
                to unit speed. Components may be zero or negative;
                only the zero vector is rejected.
            **create_kw: Forwarded to :meth:`create` (``mask``).

        Returns:
            ``(model, scales)``. ``scales.dt_from_cfl(C, (nx, ny),
            extent=(1.0, aspect))`` gives a step in the same time unit;
            pass both cell counts and the aspect ratio, since the bound
            depends on the *smaller* spacing.
        """
        context = "LinearConvection2D.from_nondimensional"
        require_positive(context, aspect=aspect)
        cx, cy = _unit_direction(context, direction)
        model = LinearConvection2D.create(
            nx=nx,
            ny=ny,
            Lx=1.0,
            Ly=aspect,
            cx=cx,
            cy=cy,
            **create_kw,
        )
        return model, Scales.advective(L=1.0, U=1.0)

    @staticmethod
    def create(
        nx: int = 64,
        ny: int = 64,
        Lx: float = 2.0,
        Ly: float = 2.0,
        cx: float = 1.0,
        cy: float = 1.0,
        mask: Mask2D | None = None,
    ) -> LinearConvection2D:
        """Convenience factory.

        Args:
            nx: Number of interior cells in x.
            ny: Number of interior cells in y.
            Lx: Domain length in x.
            Ly: Domain length in y.
            cx: Wave speed in x.
            cy: Wave speed in y.
            mask: Optional Arakawa C-grid mask (``None`` = all-ocean).

        Returns:
            A ``LinearConvection2D`` model instance.
        """
        grid = CartesianGrid2D.from_interior(nx, ny, Lx, Ly)
        params = LinearConvection2DParams(cx=as_parameter(cx), cy=as_parameter(cy))
        diff = Difference2D(grid=grid, mask=mask)
        interp = Interpolation2D(grid=grid, mask=mask)
        return LinearConvection2D(
            params=params, grid=grid, diff=diff, interp=interp, mask=mask
        )


def _unit_direction(
    context: str, direction: tuple[float, float]
) -> tuple[float, float]:
    """Normalise a propagation direction to unit speed.

    Args:
        context: Caller name, for error messages.
        direction: ``(cx, cy)``; components may be zero or negative.

    Returns:
        ``(cx, cy)`` scaled so ``cx**2 + cy**2 == 1``.

    Raises:
        ValueError: If the pair is not two finite numbers, or is the
            zero vector, which has no direction to normalise.
    """
    if len(direction) != 2:
        raise ValueError(
            f"{context}: direction must be a (cx, cy) pair; got {direction!r}."
        )
    cx, cy = (float(c) for c in direction)
    if not (math.isfinite(cx) and math.isfinite(cy)):
        raise ValueError(
            f"{context}: direction components must be finite; got {direction!r}."
        )
    speed = math.hypot(cx, cy)
    if speed == 0.0:
        raise ValueError(
            f"{context}: direction must not be the zero vector — there is no "
            f"direction to normalise, and the equation would be trivial."
        )
    return cx / speed, cy / speed
