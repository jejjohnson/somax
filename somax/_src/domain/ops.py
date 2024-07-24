from typing import Tuple
from somax._src.constants import R_EARTH
import jax.numpy as jnp
from jaxtyping import Array


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