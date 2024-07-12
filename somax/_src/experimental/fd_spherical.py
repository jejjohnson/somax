from typing import (
    Tuple,
    Union,
)

import finitediffx as fdx
import jax.numpy as jnp
from jaxtyping import (
    Array,
    Float,
)

from somax._src.constants import (
    DEG2M,
)


def difference_spherical_lon(
    u: Float[Array, "... Dlon Dlat"],
    lat_coords: Float[Array, "Dlat"],
    axis: int = -2,
    step_size: float | Float[Array, "Dlon"] = 1.0,
    accuracy: int = 1,
    derivative: int = 1,
    method: str = "central",
) -> Float[Array, "... Dlon Dlat"]:

    # convert coordinates
    cos_theta = jnp.cos(jnp.deg2rad(lat_coords))

    # calculate derivative
    du = fdx.difference(
        array=u,
        axis=axis,
        accuracy=accuracy,
        step_size=step_size,
        derivative=derivative,
        method=method,
    )

    # scale
    du = du / cos_theta / DEG2M

    return du


def difference_spherical_lat(
    u: Float[Array, "... Dlon Dlat"],
    axis: int = -1,
    step_size: float | Float[Array, "Dlat"] = 1.0,
    accuracy: int = 1,
    derivative: int = 1,
    method: str = "central",
) -> Float[Array, "... Dlon Dlat"]:

    # calculate derivative
    du = fdx.difference(
        array=u,
        axis=axis,
        accuracy=accuracy,
        step_size=step_size,
        derivative=derivative,
        method=method,
    )

    # scale
    du /= DEG2M

    return du


def gradient_spherical(
    u: Float[Array, "... Dlon Dlat"],
    lat_coords: Float[Array, "Dlat"],
    step_size_lon: float | Float[Array, "Dlon"] = 1.0,
    step_size_lat: float | Float[Array, "Dlat"] = 1.0,
    accuracy: int = 1,
    method: str = "central",
) -> Tuple[Array, Array]:

    d_dlon = difference_spherical_lon(
        u=u,
        lat_coords=lat_coords,
        axis=-2,
        step_size=step_size_lon,
        accuracy=accuracy,
        derivative=1,
        method=method,
    )

    d_dlat = difference_spherical_lat(
        u=u,
        axis=-1,
        step_size=step_size_lat,
        accuracy=accuracy,
        derivative=1,
        method=method,
    )
    return d_dlon, d_dlat


def perpendicular_gradient_spherical(
    u: Float[Array, "Dlon Dlat"],
    lat_coords: Float[Array, "Dlat"],
    step_size_lon: float | Float[Array, "Dlon"] = 1.0,
    step_size_lat: float | Float[Array, "Dlat"] = 1.0,
    accuracy: int = 1,
    method: str = "central",
) -> tuple[Float[Array, "Dlon Dlat-1"], Float[Array, "Dlon-1 Ny"]]:
    """Calculates the geostrophic gradient for a staggered grid

    Equation:
        u velocity = -∂yΨ
        v velocity = ∂xΨ

    Args:
        u (Array): the input variable
            Size = [Nx,Ny]
        dx (float | Array): the stepsize for the x-direction
        dy (float | Array): the stepsize for the y-direction

    Returns:
        du_dy (Array): the geostrophic velocity in the y-direction
            Size = [Nx,Ny-1]
        du_dx (Array): the geostrophic velocity in the x-direction
            Size = [Nx-1,Ny]

    Note:
        for the geostrophic velocity, we need to multiply the
        derivative in the x-direction by negative 1.
    """

    d_dlat = difference_spherical_lat(
        u=u,
        axis=-1,
        step_size=step_size_lat,
        derivative=1,
        method=method,
        accuracy=accuracy,
    )
    d_dlon = difference_spherical_lon(
        u=u,
        lat_coords=lat_coords,
        axis=-2,
        step_size=step_size_lon,
        derivative=1,
        method=method,
        accuracy=accuracy,
    )
    return -d_dlat, d_dlon


def divergence_spherical(
    u: Float[Array, "... Dlon Dlat"],
    v: Float[Array, "... Dlon Dlat"],
    lat_coords: Float[Array, "Dlat"],
    step_size_lon: float | Float[Array, "Dlon"] = 1.0,
    step_size_lat: float | Float[Array, "Dlat"] = 1.0,
    accuracy: int = 1,
    method: str = "central",
    cos_scaling: bool = False,
) -> Array:
    """Calculates the divergence for a staggered grid

    Equation:
        ∇⋅u̅  = ∂x(u) + ∂y(v)

    Args:
        u (Array): the input array for the u direction
            Size = [Nx+1, Ny]
        v (Array): the input array for the v direction
            Size = [Nx, Ny-1]
        dx (float | Array): the stepsize for the x-direction
        dy (float | Array): the stepsize for the y-direction

    Returns:
        div (Array): the divergence
            Size = [Nx-1,Ny-1]

    """
    # ∂xu
    du_dlon = difference_spherical_lon(
        u=u,
        lat_coords=lat_coords,
        axis=-2,
        step_size=step_size_lon,
        accuracy=accuracy,
        derivative=1,
        method=method,
    )
    # ∂yv
    dv_lat = difference_spherical_lat(
        u=v if not cos_scaling else jnp.cos(jnp.deg2rad(lat_coords)),
        axis=-1,
        step_size=step_size_lat,
        accuracy=accuracy,
        derivative=1,
        method=method,
    )

    return du_dlon + dv_lat


def curl_spherical(
    u: Float[Array, "... Dlon Dlat"],
    v: Float[Array, "... Dlon Dlat"],
    lat_coords: Float[Array, "Dlat"],
    step_size_lon: float | Float[Array, "Dlon"] = 1.0,
    step_size_lat: float | Float[Array, "Dlat"] = 1.0,
    accuracy: int = 1,
    method: str = "central",
) -> Float[Array, "Nx-1 Ny-1"]:
    """
    Calculates the relative vorticity by using finite difference in the y and x direction for the u and v velocities respectively.

    Eqn:
        ζ = ∂v/∂x - ∂u/∂y

    Args:
        u (Array): The input array for the u direction. Size = [Nx, Ny-1].
        v (Array): The input array for the v direction. Size = [Nx-1, Ny].
        lat_coords (Array): The latitude coordinates. Size = [Ny].
        step_size_lon (float | Array, optional): The step size for the longitude direction. Defaults to 1.0.
        step_size_lat (float | Array, optional): The step size for the latitude direction. Defaults to 1.0.
        accuracy (int, optional): The accuracy of the finite difference approximation. Defaults to 1.
        method (str, optional): The method used for the finite difference approximation. Defaults to "central".

    Returns:
        Array: The relative vorticity. Size = [Nx-1, Ny-1].
    """
    # ∂xv
    dv_dlon = difference_spherical_lon(
        u=v,
        lat_coords=lat_coords,
        axis=-2,
        step_size=step_size_lon,
        accuracy=accuracy,
        derivative=1,
        method=method,
    )
    # ∂yu
    du_dlat = difference_spherical_lat(
        u=u,
        axis=-1,
        step_size=step_size_lat,
        accuracy=accuracy,
        derivative=1,
        method=method,
    )

    return dv_dlon - du_dlat


def laplacian_spherical(
    u: Float[Array, "... Dlon Dlat"],
    lat_coords: Float[Array, "Dlat"],
    step_size_lon: float | Float[Array, "Dlon"] = 1.0,
    step_size_lat: float | Float[Array, "Dlat"] = 1.0,
    accuracy: int = 1,
    method: str = "central",
) -> Array:
    """Calculates the divergence for a staggered grid

    Equation:
        ∇⋅u̅  = ∂x(u) + ∂y(v)

    Args:
        u (Array): the input array for the u direction
            Size = [Nx+1, Ny]
        v (Array): the input array for the v direction
            Size = [Nx, Ny-1]
        dx (float | Array): the stepsize for the x-direction
        dy (float | Array): the stepsize for the y-direction

    Returns:
        div (Array): the divergence
            Size = [Nx-1,Ny-1]

    """
    # ∂xu
    u, v = gradient_spherical(
        u=u,
        lat_coords=lat_coords,
        step_size_lon=step_size_lon,
        step_size_lat=step_size_lat,
        accuracy=accuracy,
        method=method,
    )
    u_lap = divergence_spherical(
        u=u,
        v=v,
        lat_coords=lat_coords,
        step_size_lat=step_size_lat,
        step_size_lon=step_size_lon,
        method=method,
        accuracy=accuracy,
    )

    return u_lap
