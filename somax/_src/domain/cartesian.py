import einx
import jax
from functools import partial
import equinox as eqx
import jax.numpy as jnp
from typing import Self, Optional, Tuple, Iterable
from jaxtyping import Array
from somax._src.domain.base import Domain
from somax._src.domain.utils import (
    make_coords,
    make_grid_coords,
    bounds_and_points_to_step,
    bounds_to_length,
    bounds_and_step_to_points,
)
from somax._src.operators.stagger import domain_limits_transform


class CartesianDomain1D(Domain):
    xmin: float = eqx.field(static=True)
    xmax: float = eqx.field(static=True)
    dx: float = eqx.field(static=True)
    Nx: int = eqx.field(static=True)
    Lx: float = eqx.field(static=True)

    def __repr__(self) -> str:
        return (
            f"X-Domain"
            + f"\n(xmin, xmax): ({self.xmin}, {self.xmax})"
            + f"\nDx: {self.dx}"
            + f"\nNx: {self.Nx}"
            + f"\nLx: {self.Lx}"
        )

    @classmethod
    def init_from_bounds_and_step_size(
        cls, xmin: float, xmax: float, dx: float
    ) -> Self:
        """initialize 1d domain from bounds and number of points
        Eqs:
            dx = (x_max - x_min) / (Nx - 1)
            Lx = (x1 - x0)

        Args:
            xmin (float): x min value
            xmax (float): maximum value
            Nx (int): the number of points for domain
        Returns:
            domain (Domain): the full domain
        """

        Nx = bounds_and_step_to_points(xmin=xmin, xmax=xmax, dx=dx)
        Lx = bounds_to_length(xmin, xmax)
        return cls(xmin=xmin, xmax=xmax, Nx=Nx, dx=dx, Lx=Lx)

    @classmethod
    def init_from_bounds_and_num_pts(
        cls, xmin: float, xmax: float, Nx: int
    ) -> Self:
        """initialize 1d domain from bounds and step_size
        Eqs:
            Nx = 1 + ceil((x_max - x_min) / dx)
            Lx = (x1 - x0)

        Args:
            xmin (float): x min value
            xmax (float): maximum value
            Nx (int): the number of points for domain
        Returns:
            domain (Domain): the full domain
        """
        dx = bounds_and_points_to_step(xmin=xmin, xmax=xmax, Nx=Nx)
        Lx = bounds_to_length(xmin=xmin, xmax=xmax)
        return cls(xmin=xmin, xmax=xmax, Nx=Nx, dx=dx, Lx=Lx)

    @classmethod
    def init_from_coords(cls, x_coords: Array) -> Self:
        """Initialize 1D domain from coordinates.

        This method initializes a 1D domain using the given x-coordinates. It calculates the necessary parameters
        such as the minimum and maximum values, the number of points, and the step size.

        Args:
            x_coords (Array): The x-coordinates for the domain.

        Returns:
            domain (Domain): The initialized domain.

        """
        dx = jnp.diff(x_coords).mean()
        Nx = len(x_coords)
        xmin, xmax = x_coords.min(), x_coords.max()
        Lx = xmax - xmin
        return cls(
            xmin=float(xmin),
            xmax=float(xmax),
            Nx=int(Nx),
            dx=float(dx),
            Lx=float(Lx),
        )

    @property
    def bounds(self) -> Tuple[int, int]:
        return (self.xmin, self.xmax)

    @property
    def coords(self) -> Array:
        return jnp.asarray(make_coords(self.xmin, self.xmax, self.Nx))

    @property
    def grid(self) -> Array:
        return jnp.asarray(einx.rearrange("Dx -> Dx 1", self.coords))

    @property
    def cell_volume(self) -> float:
        return self.dx

    @property
    def ndim(self) -> int:
        return 1

    def isel(self, sel: slice):
        """
        Selects a subset of coordinates from the Cartesian domain.

        Args:
            sel (slice): The slice object representing the subset of coordinates to select.

        Returns:
            Self: A new instance of the CartesianDomain1D class with the selected subset of coordinates.

        Raises:
            AssertionError: If the input `sel` is not a slice object.

        """
        assert isinstance(sel, slice)
        sel = list([sel])
        sliced_coords = self.coords[sel[0]]
        xmin, xmax = sliced_coords[0], sliced_coords[-1]
        Lx = xmax - xmin
        Nx = len(sliced_coords)
        return CartesianDomain1D(
            xmin=xmin, xmax=xmax, dx=self.dx, Lx=Lx, Nx=Nx
        )

    def pad(
        self,
        pad_width: Tuple[int, int],
    ):

        return pad_domain(self, pad_width=pad_width)

    def stagger(
        self,
        direction: Optional[str] = None,
        stagger: Optional[bool] = None,
    ):
        xmin, xmax, Nx, Lx = domain_limits_transform(
            xmin=self.xmin,
            xmax=self.xmax,
            dx=self.dx,
            Lx=self.Lx,
            Nx=self.Nx,
            direction=direction,
            stagger=stagger,
        )
        return CartesianDomain1D(
            xmin=xmin, xmax=xmax, dx=self.dx, Lx=Lx, Nx=Nx
        )
    

class CartesianDomain2D(Domain):
    x_domain: CartesianDomain1D
    y_domain: CartesianDomain1D

    def __repr__(self) -> str:
        return (
            f"X-Y Domain:"
            + f"\nBounds (xmin, ymin, xmax, ymax): {self.bounds}"
            + f"\nShape (Nx, Ny): {self.shape}"
            + f"\nSize (Lx, Ly): {self.size}:"
            + f"\nRes. (dx, dy): {self.resolution}"
            + f"\nVol (dx * dy): {self.cell_volume}"
        )

    @classmethod
    def init_from_bounds_and_step_size(
        cls, xmin: Tuple[int, int], xmax: Tuple[int, int], dx: Tuple[int, int]
    ) -> Self:
        """initialize 1d domain from bounds and number of points
        Eqs:
            dx = (x_max - x_min) / (Nx - 1)
            Lx = (x1 - x0)

        Args:
            xmin (float): x min value
            xmax (float): maximum value
            Nx (int): the number of points for domain
        Returns:
            domain (Domain): the full domain
        """
        assert len(xmin) == len(xmax) == len(dx) == 2
        x_domain = CartesianDomain1D.init_from_bounds_and_step_size(
            xmin=xmin[0], xmax=xmax[0], dx=dx[0]
        )
        y_domain = CartesianDomain1D.init_from_bounds_and_step_size(
            xmin=xmin[1], xmax=xmax[1], dx=dx[1]
        )
        return cls(
            x_domain=x_domain,
            y_domain=y_domain,
        )

    @classmethod
    def init_from_bounds_and_num_pts(
        cls, xmin: Tuple[int, int], xmax: Tuple[int, int], Nx: Tuple[int, int]
    ) -> Self:
        """initialize 1d domain from bounds and step_size
        Eqs:
            Nx = 1 + ceil((x_max - x_min) / dx)
            Lx = (x1 - x0)

        Args:
            xmin (float): x min value
            xmax (float): maximum value
            Nx (int): the number of points for domain
        Returns:
            domain (Domain): the full domain
        """
        assert len(xmin) == len(xmax) == len(Nx) == 2
        x_domain = CartesianDomain1D.init_from_bounds_and_num_pts(
            xmin=xmin[0], xmax=xmax[0], Nx=Nx[0]
        )
        y_domain = CartesianDomain1D.init_from_bounds_and_num_pts(
            xmin=xmin[1], xmax=xmax[1], Nx=Nx[1]
        )
        return cls(
            x_domain=x_domain,
            y_domain=y_domain,
        )

    @classmethod
    def init_from_coords(cls, coords: Tuple[Array, Array]) -> Self:
        """
        Initialize a Cartesian object from coordinates.

        Args:
            coords (Tuple[Array, Array]): A tuple containing the x and y coordinates.

        Returns:
            Self: A new instance of the Cartesian object.

        Raises:
            AssertionError: If the length of the coordinates is zero.

        """
        assert len(coords)
        x_domain = CartesianDomain1D.init_from_coords(x_coords=coords[0])
        y_domain = CartesianDomain1D.init_from_coords(x_coords=coords[1])
        return cls(
            x_domain=x_domain,
            y_domain=y_domain,
        )

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return (
            float(self.x_domain.xmin),
            float(self.y_domain.xmin),
            float(self.x_domain.xmax),
            float(self.y_domain.xmax),
        )

    @property
    def shape(self) -> Tuple[int, int]:
        return (int(self.x_domain.Nx), int(self.y_domain.Nx))

    @property
    def size(self) -> Tuple[float, float]:
        return (float(self.x_domain.Lx), float(self.y_domain.Lx))

    @property
    def resolution(self) -> Tuple[float, float]:
        return (float(self.x_domain.dx), float(self.y_domain.dx))

    @property
    def coords_x(self) -> Array:
        return self.x_domain.coords

    @property
    def coords_y(self) -> Array:
        return self.y_domain.coords

    @property
    def grid(self) -> Array:
        x_coords = self.x_domain.coords
        y_coords = self.y_domain.coords
        return make_grid_coords([x_coords, y_coords])

    @property
    def cell_volume(self) -> float:
        return self.x_domain.dx * self.y_domain.dx

    @property
    def ndim(self) -> int:
        return 2

    def isel_x(self, sel: slice) -> Self:
        # check coordinates are correct
        x_domain = self.x_domain.isel(sel=sel)
        return CartesianDomain2D(x_domain=x_domain, y_domain=self.y_domain)

    def isel_y(self, sel: slice) -> Self:
        # check coordinates are correct
        y_domain = self.y_domain.isel(sel=sel)
        return CartesianDomain2D(x_domain=self.x_domain, y_domain=y_domain)

    def pad_x(
        self,
        pad_width: Tuple[int, int],
    ) -> Self:

        x_domain = self.x_domain.pad(pad_width=pad_width)
        return CartesianDomain2D(x_domain=x_domain, y_domain=self.y_domain)

    def pad_y(
        self,
        pad_width: Tuple[int, int],
    ) -> Self:
        y_domain = self.y_domain.pad(pad_width=pad_width)
        return CartesianDomain2D(x_domain=self.x_domain, y_domain=y_domain)

    def stagger_x(
        self,
        direction: Optional[str] = None,
        stagger: Optional[bool] = None,
    ) -> Self:
        return stagger_x_field(self, direction, stagger)

    def stagger_y(
        self,
        direction: Optional[Iterable[str]] = None,
        stagger: Optional[Iterable[bool]] = None,
    ) -> Self:
        y_domain = self.y_domain.stagger(direction=direction, stagger=stagger)
        return CartesianDomain2D(x_domain=self.x_domain, y_domain=y_domain)


def pad_domain(domain: Domain, pad_width: Tuple[int,int]) -> Domain:
        """
        Pads the CartesianDomain1D object with the specified pad width.

        Args:
            pad_width (Tuple[int, int]): The amount of padding to add on each side of the domain.

        Returns:
            CartesianDomain1D: A new CartesianDomain1D object with the specified padding.

        Raises:
            AssertionError: If pad_width is not a tuple or if its length is not equal to 2.
        """

        assert isinstance(pad_width, tuple)
        assert len(pad_width) == 2

        xmin = domain.xmin - pad_width[0] * domain.dx
        xmax = domain.xmax + pad_width[1] * domain.dx
        Lx = xmax - xmin
        Nx = sum(pad_width) + domain.Nx

        return CartesianDomain1D(
            xmin=xmin, xmax=xmax, dx=domain.dx, Lx=Lx, Nx=Nx
        )

@partial(jax.jit, static_argnames=("direction", "stagger",))
def stagger_x_field(
    domain: Domain,
    direction: Optional[str] = None,
    stagger: Optional[bool] = None,
) -> CartesianDomain2D:
    x_domain = domain.x_domain.stagger(direction=direction, stagger=stagger)
    return CartesianDomain2D(x_domain=x_domain, y_domain=domain.y_domain)