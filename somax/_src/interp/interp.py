from typing import (
    Callable,
    Optional,
)
from jaxtyping import (
    Array,
    Float,
)

from interpax import Interpolator2D
import jax.numpy as jnp
from jaxinterp2d import CartesianGrid
from jax.scipy.interpolate import RegularGridInterpolator

from somax.domain import Domain

def domain_interpolation_2D(
    u: Float[Array, "Nx Ny"],
    source_domain: Domain,
    target_domain: Domain,
    method: str = "linear",
    extrap: bool = False,
) -> Array:
    """This function will interpolate the values
    from one domain to a target domain

    Args:
        u (Array): the input array
            Size = [Nx, Ny]
        source_domain (Domain): the domain of the input array
        target_domain (Domain): the target domain

    Returns:
        u_ (Array): the input array for the target domain
    """

    assert len(source_domain.Nx) == len(target_domain.Nx) == 2
    assert source_domain.Nx == u.shape

    # initialize interpolator
    interpolator = Interpolator2D(
        x=source_domain.coords_axis[0],
        y=source_domain.coords_axis[1],
        f=u,
        method=method,
        extrap=extrap,
    )

    # get coordinates of target grid
    X, Y = target_domain.grid_axis

    # interpolate
    u_on_target = interpolator(xq=X.ravel(), yq=Y.ravel())

    # reshape
    u_on_target = u_on_target.reshape(target_domain.Nx)

    return u_on_target


def cartesian_interpolator_2D(
    u: Float[Array, "Nx Ny"],
    source_domain: Domain,
    target_domain: Domain,
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

    assert len(source_domain.Nx) == len(target_domain.Nx) == 2
    assert source_domain.Nx == u.shape

    # get limits for domain
    xlims = (source_domain.xmin[0], source_domain.xmax[0])
    ylims = (source_domain.xmin[1], source_domain.xmax[1])

    # initialize interpolator
    interpolator = CartesianGrid(limits=(xlims, ylims), values=u, mode=mode, cval=cval)

    # get coordinates of target grid
    X, Y = target_domain.grid_axis

    # interpolate
    u_on_target = interpolator(X.ravel(), Y.ravel())

    # reshape
    u_on_target = u_on_target.reshape(target_domain.Nx)

    return u_on_target
