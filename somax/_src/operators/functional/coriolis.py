
from typing import Optional
import jax.numpy as jnp
from jaxtyping import Array

from finitevolx._src.constants import OMEGA, R_EARTH


def coriolis_fn(Y: Array, f0: Array | float=9.375e-5, beta: Array | float=1.754e-11, y0: Optional[float]=None) -> Array:
    """Beta-Plane Approximation for the coriolis parameter

    Eq:
        f = f₀ + β(y - y₀)

    where:
        f₀ = 2Ω sin θ₀
        β = (1/R) 2Ω cos θ₀

    Args:
        Y (Array): the grid coordinates for the Y-coordinates
        f0 (float): the Coriolis parameter [s^-1]
        beta (float): coriolis parameter gradient [m^-1 s^-1]
        y0 (float): the mean distance [m]

    Returns:
        f (Array): the Coriolis parameter
    """
    # take mean distance
    if y0 is None:
        y0 = jnp.mean(Y)

    # calculate coriolis parameter
    f = f0 + beta * (Y - y0)

    return f

def beta_plane(lat: Array, omega: float = OMEGA, radius: float = R_EARTH) -> Array:
    """Beta-Plane Approximation coefficient from the mean latitude

    Equation:
        β = (1/R) 2Ω cos θ₀

    Args:
        lat (Array): the mean latitude [degrees]
        omega (float): the rotation (default=...)
        radius (float): the radius of the Earth (default=...)

    Returns:
        beta (Array): the beta plane parameter
    """
    lat = jnp.deg2rad(lat)
    return (2 * omega / radius) * jnp.cos(lat)


def coriolis_param(lat: Array, omega: float = OMEGA) -> Array:
    """The Coriolis parameter

    Equation:
        f = 2Ω sin θ₀

    Args:
        lat (Array): the mean latitude [degrees]
        omega (Array): the rotation (default=...)

    Returns:
        Coriolis (float): the coriolis parameter
    """
    # calculate mean
    lat0 = jnp.mean(lat)
    # change coordinates
    lat0 = jnp.deg2rad(lat0)
    # calculate coriolis
    return 2.0 * omega * jnp.sin(lat0)
