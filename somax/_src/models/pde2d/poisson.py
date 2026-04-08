"""2D elliptic solvers: Laplace, Poisson, and Helmholtz equations."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from finitevolx import ArakawaCGrid2D, streamfunction_from_vorticity
from jaxtyping import Array, Float


class PoissonSolver2D(eqx.Module):
    r"""Solve the 2D Poisson equation: :math:`\nabla^2 \phi = f`.

    Wraps finitevolX spectral solvers (DST for Dirichlet, DCT for
    Neumann, FFT for periodic). The solver operates on interior cells
    and returns a full-grid array including ghost cells.

    Args:
        grid: 2D Arakawa C-grid.
        bc_type: Boundary condition type (``"dirichlet"``, ``"neumann"``,
            or ``"periodic"``).
    """

    grid: ArakawaCGrid2D = eqx.field(static=True)
    bc_type: str = eqx.field(static=True, default="dirichlet")

    def solve(self, rhs: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        r"""Solve :math:`\nabla^2 \phi = f`.

        Args:
            rhs: Right-hand side on the full grid (ghost cells ignored).

        Returns:
            Solution :math:`\phi` on the full grid.
        """
        bc_map = {"dirichlet": "dst", "neumann": "dct", "periodic": "fft"}
        bc = bc_map[self.bc_type]
        return streamfunction_from_vorticity(rhs, self.grid.dx, self.grid.dy, bc=bc)

    @staticmethod
    def create(
        nx: int = 64,
        ny: int = 64,
        Lx: float = 1.0,
        Ly: float = 1.0,
        bc: str = "dirichlet",
    ) -> PoissonSolver2D:
        """Convenience factory.

        Args:
            nx: Number of interior cells in x.
            ny: Number of interior cells in y.
            Lx: Domain length in x.
            Ly: Domain length in y.
            bc: Boundary condition type.

        Returns:
            A ``PoissonSolver2D`` instance.
        """
        grid = ArakawaCGrid2D.from_interior(nx, ny, Lx, Ly)
        return PoissonSolver2D(grid=grid, bc_type=bc)


class LaplaceSolver2D(eqx.Module):
    r"""Solve the 2D Laplace equation: :math:`\nabla^2 \phi = 0`.

    A thin wrapper around :class:`PoissonSolver2D` with zero RHS.
    The solution is determined entirely by boundary conditions.

    Args:
        grid: 2D Arakawa C-grid.
        bc_type: Boundary condition type.
    """

    grid: ArakawaCGrid2D = eqx.field(static=True)
    bc_type: str = eqx.field(static=True, default="dirichlet")

    def solve(
        self, boundary_values: Float[Array, "Ny Nx"] | None = None
    ) -> Float[Array, "Ny Nx"]:
        r"""Solve :math:`\nabla^2 \phi = 0`.

        Args:
            boundary_values: Optional array with desired boundary values
                in the ghost cells (interior is ignored).

        Returns:
            Solution :math:`\phi` on the full grid.
        """
        rhs = jnp.zeros((self.grid.Ny, self.grid.Nx))
        bc_map = {"dirichlet": "dst", "neumann": "dct", "periodic": "fft"}
        bc = bc_map[self.bc_type]
        return streamfunction_from_vorticity(rhs, self.grid.dx, self.grid.dy, bc=bc)

    @staticmethod
    def create(
        nx: int = 64,
        ny: int = 64,
        Lx: float = 1.0,
        Ly: float = 1.0,
        bc: str = "dirichlet",
    ) -> LaplaceSolver2D:
        """Convenience factory."""
        grid = ArakawaCGrid2D.from_interior(nx, ny, Lx, Ly)
        return LaplaceSolver2D(grid=grid, bc_type=bc)


class HelmholtzSolver2D(eqx.Module):
    r"""Solve the 2D Helmholtz equation: :math:`(\nabla^2 - \lambda) \phi = f`.

    Wraps finitevolX spectral solvers with a non-zero Helmholtz
    parameter. This arises in quasi-geostrophic PV inversion
    :math:`(\nabla^2 - F)\psi = q`, Yukawa screening, and
    reaction-diffusion steady states.

    Args:
        grid: 2D Arakawa C-grid.
        bc_type: Boundary condition type.
        lambda_: Helmholtz parameter (screening coefficient).
    """

    grid: ArakawaCGrid2D = eqx.field(static=True)
    bc_type: str = eqx.field(static=True, default="dirichlet")
    lambda_: float = eqx.field(static=True, default=1.0)

    def solve(self, rhs: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        r"""Solve :math:`(\nabla^2 - \lambda) \phi = f`.

        Args:
            rhs: Right-hand side on the full grid.

        Returns:
            Solution :math:`\phi` on the full grid.
        """
        bc_map = {"dirichlet": "dst", "neumann": "dct", "periodic": "fft"}
        bc = bc_map[self.bc_type]
        return streamfunction_from_vorticity(
            rhs, self.grid.dx, self.grid.dy, bc=bc, lambda_=self.lambda_
        )

    @staticmethod
    def create(
        nx: int = 64,
        ny: int = 64,
        Lx: float = 1.0,
        Ly: float = 1.0,
        lambda_: float = 1.0,
        bc: str = "dirichlet",
    ) -> HelmholtzSolver2D:
        """Convenience factory.

        Args:
            nx: Number of interior cells in x.
            ny: Number of interior cells in y.
            Lx: Domain length in x.
            Ly: Domain length in y.
            lambda_: Helmholtz parameter.
            bc: Boundary condition type.

        Returns:
            A ``HelmholtzSolver2D`` instance.
        """
        grid = ArakawaCGrid2D.from_interior(nx, ny, Lx, Ly)
        return HelmholtzSolver2D(grid=grid, bc_type=bc, lambda_=lambda_)
