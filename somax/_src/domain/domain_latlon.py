from functools import reduce
from operator import mul
import typing as tp

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array
import numpy as np
from somax._src.domain.cartesian import CartesianDomain2D, CartesianDomain1D
from typing import Tuple
from somax._src.constants import R_EARTH
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
from metpy.calc import lat_lon_grid_deltas
from somax._src.domain.utils import bounds_to_length


def init_cartesian2D_from_lonlat(
    lon: ArrayLike, lat: ArrayLike, radius: float = R_EARTH):

    # estimate grid deltas
    # dx, dy = lat_lon_deltas(lon=lon, lat=lat, radius=radius)
    
    dx, dy = lat_lon_grid_deltas(longitude=lon, latitude=lat)
    
    dx = dx.mean().magnitude
    dy = dy.mean().magnitude
    
    # get shape
    if lon.ndim > 1 & lat.ndim > 1:
        Nx, Nx = lon.shape
    else:
        Nx = lon.shape[0]
        Ny = lat.shape[0]
        
    # get bounds
    xmin, xmax = lon.min(), lon.max()
    ymin, ymax = lat.min(), lat.max()
        
    
    # calculate size
    Lx = bounds_to_length(xmin=xmin, xmax=xmax)
    Ly = bounds_to_length(xmin=ymin, xmax=ymax)
    
    # initialize coordinates
    x_domain = CartesianDomain1D(xmin=xmin, xmax=xmax, dx=dx, Nx=Nx, Lx=Lx)
    y_domain = CartesianDomain1D(xmin=ymin, xmax=ymax, dx=dy, Nx=Ny, Lx=Ly)

    # Initialize using the parent class method
    return CartesianDomain2D(x_domain=x_domain, y_domain=y_domain)


def lat_lon_deltas(
    lon: Array, lat: Array, radius: float = R_EARTH
) -> Tuple[Array, Array]:
    """Calculates the dx,dy for lon/lat coordinates. Uses
    the spherical Earth projected onto a plane approx.

    Eqn:
        d = R √ [Δϕ² + cos(ϕₘ)Δλ]

        Δϕ - change in latitude
        Δλ - change in longitude
        ϕₘ - mean latitude
        R - radius of the Earth

    Args:
        lon (Array): the longitude coordinates [degrees]
        lat (Array): the latitude coordinates [degrees]

    Returns:
        dx (Array): the change in x [m]
        dy (Array): the change in y [m]

    Resources:
        https://en.wikipedia.org/wiki/Geographical_distance#Spherical_Earth_projected_to_a_plane

    """

    assert lon.ndim == lat.ndim
    assert lon.ndim > 0 and lon.ndim < 3

    if lon.ndim < 2:
        lon, lat = jnp.meshgrid(lon, lat, indexing="ij")

    lon = jnp.deg2rad(lon)
    lat = jnp.deg2rad(lat)

    lat_mean = jnp.mean(lat)

    dlon_dx, dlon_dy = jnp.gradient(lon)
    dlat_dx, dlat_dy = jnp.gradient(lat)

    dx = radius * jnp.hypot(dlat_dx, dlon_dx * jnp.cos(lat_mean))
    dy = radius * jnp.hypot(dlat_dy, dlon_dy * jnp.cos(lat_mean))

    return dx, dy