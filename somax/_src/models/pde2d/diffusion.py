"""2D diffusion equation: du/dt = nu * laplacian(u)."""

from __future__ import annotations

from typing import Any, ClassVar

import equinox as eqx
import jax.numpy as jnp
from finitevolx import CartesianGrid2D, Difference2D, Mask2D, enforce_periodic
from jaxtyping import Array, PyTree

from somax._src.core.model import SomaxModel
from somax._src.core.scales import Scales
from somax._src.core.types import Diagnostics, Params, State, as_parameter
from somax._src.models._nondim import require_positive


class Diffusion2DParams(Params):
    """Differentiable parameters for 2D diffusion.

    Args:
        nu: Kinematic viscosity (diffusion coefficient).
    """

    nu: Array


class Diffusion2DState(State):
    """State for 2D diffusion.

    Args:
        u: Scalar field on T-points, shape ``(Ny, Nx)``.
    """

    u: Array

    # ``u`` here is a T-point scalar, not a C-grid velocity: it is the
    # transported quantity, and the velocity is a separate coefficient.
    # Its amplitude comes from the initial condition, so there is no
    # scale for it in ``Scales`` and the nondimensionalisation leaves
    # it alone rather than dividing it by ``U``.
    scale_kinds: ClassVar[dict[str, str]] = {"u": "tracer"}
    mask_locations: ClassVar[dict[str, str]] = {"u": "h"}


class Diffusion2DDiagnostics(Diagnostics):
    """Diagnostics for 2D diffusion.

    Args:
        energy: Integrated energy.
    """

    energy: Array


class Diffusion2D(SomaxModel):
    r"""2D diffusion equation on an Arakawa C-grid.

    Solves ``du/dt = nu * (d²u/dx² + d²u/dy²)``.

    Args:
        params: Differentiable parameters (viscosity ``nu``).
        grid: 2D Arakawa C-grid.
        diff: Difference operators.
        mask: Optional Arakawa C-grid mask (``None`` = all-ocean).
    """

    params: Diffusion2DParams
    grid: CartesianGrid2D = eqx.field(static=True)
    diff: Difference2D
    mask: Mask2D | None

    def vector_field(
        self, t: float, state: PyTree, args: PyTree | None = None
    ) -> Diffusion2DState:
        r"""Compute tendency: du/dt = nu * laplacian(u)."""
        du_dt = self.params.nu * self.diff.laplacian(state.u)
        return Diffusion2DState(u=du_dt)

    def apply_boundary_conditions(self, state: PyTree) -> Diffusion2DState:
        """Apply periodic boundary conditions."""
        return Diffusion2DState(u=enforce_periodic(state.u))

    def diagnose(self, state: PyTree) -> Diffusion2DDiagnostics:
        """Compute energy diagnostic."""
        interior = state.u[1:-1, 1:-1]
        energy = 0.5 * jnp.sum(interior**2) * self.grid.dx * self.grid.dy
        return Diffusion2DDiagnostics(energy=energy)

    @staticmethod
    def from_nondimensional(
        *,
        nx: int = 64,
        ny: int = 64,
        aspect: float = 1.0,
        **create_kw: Any,
    ) -> tuple[Diffusion2D, Scales]:
        r"""Build the model at unit scales instead of SI coefficients.

        Non-dimensional form
        --------------------
        Diffusive scale set (:meth:`somax.Scales.diffusive`) with
        ``L = kappa = 1``, so ``T = L**2/kappa = 1`` and the equation
        reads ``d_t u = laplacian(u)``. Pure diffusion has no velocity
        scale, so this scaling leaves no free dimensionless number:
        every diffusion problem is the same problem once rescaled, and
        only the grid and the initial condition remain to be chosen.


        Args:
            nx: Interior cells in x.
            ny: Interior cells in y.
            aspect: ``Ly / Lx``; the domain is ``Lx = 1``.
            **create_kw: Forwarded to :meth:`create` (``mask``).

        Returns:
            ``(model, scales)``. ``scales.dt_from_cfl(C, (nx, ny),
            extent=(1.0, aspect))`` gives a step in the same time unit;
            pass both cell counts and the aspect ratio, since the bound
            depends on the *smaller* spacing.
        """
        context = "Diffusion2D.from_nondimensional"
        require_positive(context, aspect=aspect)
        model = Diffusion2D.create(
            nx=nx,
            ny=ny,
            Lx=1.0,
            Ly=aspect,
            nu=1.0,
            **create_kw,
        )
        return model, Scales.diffusive(L=1.0, kappa=1.0)

    @staticmethod
    def create(
        nx: int = 64,
        ny: int = 64,
        Lx: float = 2.0,
        Ly: float = 2.0,
        nu: float = 0.01,
        mask: Mask2D | None = None,
    ) -> Diffusion2D:
        """Convenience factory.

        Args:
            nx: Number of interior cells in x.
            ny: Number of interior cells in y.
            Lx: Domain length in x.
            Ly: Domain length in y.
            nu: Kinematic viscosity.
            mask: Optional Arakawa C-grid mask (``None`` = all-ocean).

        Returns:
            A ``Diffusion2D`` model instance.
        """
        grid = CartesianGrid2D.from_interior(nx, ny, Lx, Ly)
        params = Diffusion2DParams(nu=as_parameter(nu))
        diff = Difference2D(grid=grid, mask=mask)
        return Diffusion2D(params=params, grid=grid, diff=diff, mask=mask)
