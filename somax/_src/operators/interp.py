from typing import Iterable

from interpax import Interpolator2D
import jax.numpy as jnp
from jax.scipy.interpolate import RegularGridInterpolator
from jax.scipy.ndimage import map_coordinates
from jaxtyping import (
    Array,
    Float,
    ArrayLike
)
from plum import dispatch
from somax._src.domain.cartesian import CartesianDomain2D


class CartesianGrid:
    """
    Linear Multivariate Cartesian Grid interpolation in arbitrary dimensions. Based
    on ``map_coordinates``.

    Notes:
        Translated directly from https://github.com/JohannesBuchner/regulargrid/ to jax.
    """

    values: Array
    """
    Values to interpolate.
    """
    limits: Iterable[Iterable[float]]
    """
    Limits along each dimension of ``values``.
    """

    def __init__(
        self,
        limits: Iterable[Iterable[float]],
        values: Array,
        mode: str = "constant",
        cval: float = jnp.nan,
    ):
        """
        Initializer.

        Args:
            limits: collection of pairs specifying limits of input variables along
                each dimension of ``values``
            values: values to interpolate. These must be defined on a regular grid.
            mode: how to handle out of bounds arguments; see docs for ``map_coordinates``
            cval: constant fill value; see docs for ``map_coordinates``
        """
        super().__init__()
        self.values = values
        self.limits = limits
        self.mode = mode
        self.cval = cval

    def __call__(self, *coords) -> Array:
        """
        Perform interpolation.

        Args:
            coords: point at which to interpolate. These will be broadcasted if
                they are not the same shape.

        Returns:
            Interpolated values, with extrapolation handled according to ``mode``.
        """
        # transform coords into pixel values
        coords = jnp.broadcast_arrays(*coords)
        # coords = jnp.asarray(coords)
        coords = [
            (c - lo) * (n - 1) / (hi - lo)
            for (lo, hi), c, n in zip(self.limits, coords, self.values.shape)
        ]
        return map_coordinates(
            self.values, coords, mode=self.mode, cval=self.cval, order=1
        )


def domain_interpolation_2D(
    u: Float[ArrayLike, "... Nx Ny"],
    source_domain: CartesianDomain2D,
    target_domain: CartesianDomain2D,
    method: str = "linear",
    extrap: bool = False,
) -> Array:
    """Interpolates the values from one domain to a target domain.

    Args:
        u (Array): The input array to be interpolated. Size = [Nx, Ny]
        source_domain (Domain): The domain of the input array.
        target_domain (Domain): The target domain.
        method (str, optional): The interpolation method to use. Defaults to "linear".
        extrap (bool, optional): Whether to perform extrapolation. Defaults to False.

    Returns:
        Array: The input array for the target domain.
    """

    # assert number of dimensions are the same
    assert source_domain.ndim == target_domain.ndim == 2
    # assert source domain is the same as array
    assert source_domain.shape == u.shape

    # initialize interpolator
    interpolator = Interpolator2D(
        x=source_domain.coords_x,
        y=source_domain.coords_y,
        f=u,
        method=method,
        extrap=extrap,
    )

    # get coordinates of target grid
    XY = target_domain.grid

    # interpolate
    u_on_target = interpolator(xq=XY[..., 0].ravel(), yq=XY[..., 1].ravel())

    # reshape
    u_on_target = u_on_target.reshape(target_domain.shape)

    return u_on_target


def regulargrid_interpolator_2D(
    u: Float[ArrayLike, "... Nx Ny"],
    source_domain: CartesianDomain2D,
    target_domain: CartesianDomain2D,
    method: str = "linear",
    fill_value=jnp.nan,
):
    """
    Interpolates a 2D array onto a regular grid using the specified method.

    Args:
        u (numpy.ndarray): The input array to be interpolated.
        source_domain (Domain): The domain of the source grid.
        target_domain (Domain): The domain of the target grid.
        method (str, optional): The interpolation method to use. Defaults to "linear".
        fill_value (float, optional): The value to use for points outside the source grid. Defaults to NaN.

    Returns:
        numpy.ndarray: The interpolated array on the target grid.
    """

    # assert number of dimensions are the same
    assert source_domain.ndim == target_domain.ndim == 2
    # assert source domain is the same as array
    assert source_domain.shape == u.shape

    # initialize interpolator
    interpolator = RegularGridInterpolator(
        points=(source_domain.coords_x, source_domain.coords_y),
        values=u,
        method=method,
        fill_value=fill_value,
    )

    # get coords on target grid
    XY = target_domain.grid

    # interpolate
    u_on_target = interpolator((XY[..., 0].ravel(), XY[..., 1].ravel()))

    # reshape
    u_on_target = u_on_target.reshape(target_domain.shape)
    return u_on_target


def cartesian_interpolator_2D(
    u: Float[ArrayLike, "... Nx Ny"],
    source_domain: CartesianDomain2D,
    target_domain: CartesianDomain2D,
    mode: str = "constant",
    cval: float = 0.0,
) -> Array:
    """This function will interpolate the values
    from one domain to a target domain assuming a
    Cartesian grid, i.e., a constant dx,dy,....
    This method is very fast

    Args:
        u (Array): the input array
            Size = [Nx, Ny]
        source_domain (Domain): the domain of the input array
        target_domain (Domain): the target domain

    Returns:
        u_ (Array): the input array for the target domain
    """

    assert len(source_domain.shape) == len(target_domain.shape) == 2
    assert source_domain.shape == u.shape

    # get limits for domain
    xlims = (source_domain.x_domain.xmin, source_domain.x_domain.xmax)
    ylims = (source_domain.y_domain.xmax, source_domain.y_domain.xmax)

    # initialize interpolator
    interpolator = CartesianGrid(
        limits=(xlims, ylims), values=u, mode=mode, cval=cval
    )

    # get coordinates of target grid
    XY = target_domain.grid

    # interpolate
    u_on_target = interpolator(XY[..., 0].ravel(), XY[..., 1].ravel())

    # reshape
    u_on_target = u_on_target.reshape(target_domain.shape)

    return u_on_target
