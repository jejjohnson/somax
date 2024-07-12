from typing import Optional

from somax._src.domain.cartesian import CartesianDomain2D
from somax.interp import x_avg_2D
import jax.numpy as jnp
from jaxtyping import (
    Array,
    Float,
)

from somax._src.models.sw.params import SWMParams


def init_partition(domain: CartesianDomain2D):
    Y = domain.grid[..., 1]
    y = domain.coords_y
    Ny = domain.shape[1]
    Lx = domain.size[0]
    u0 = 10 * jnp.exp(-((Y - y[Ny // 2]) ** 2) / (0.02 * Lx) ** 2)
    return u0


def init_h0_jet(domain: CartesianDomain2D, params: SWMParams, u0: Optional[Array] = None):
    dy = domain.resolution[1]
    Lx, Ly = domain.size
    XY = domain.grid
    X, Y = XY[..., 0], XY[..., 1]

    if u0 is not None:
        h_geostrophy = jnp.cumsum(
            -dy * x_avg_2D(u0) * params.coriolis_param(Y) / params.gravity,
            axis=1,
        )

        h0 = h_geostrophy.mean()
    else:
        h_geostrophy = 0.0
        h0 = 0.0

    h0 = (
        params.depth
        + h_geostrophy
        # make sure h0 is centered around depth
        - h0
        # small perturbation
        + 0.2
        * jnp.sin(X / Lx * 10.0 * jnp.pi)
        * jnp.cos(Y / Ly * 8.0 * jnp.pi)
    )

    return h0


def init_h0_zonal_jet(domain):
    y = domain.coords_axis[1]
    Y = domain.grid_axis[1]

    return -jnp.tanh(20.0 * ((Y - jnp.mean(y)) / jnp.max(y))) * 400.0


def init_H_sea_mount(domain: CartesianDomain2D):
    dx, dy = domain.resolution
    Nx, Ny = domain.shape
    x, y = domain.coords_x, domain.coords_y
    XY = domain.grid
    X, Y = XY[..., 0], XY[..., 1]
    std_mountain = 40.0 * dy
    # Standard deviation of mountain (m)
    constant = 9250
    h0 = 100.0 * jnp.exp(
        -((X - jnp.mean(x)) ** 2.0 + (Y - 0.5 * jnp.mean(y)) ** 2.0)
        / (2.0 * std_mountain**2.0)
    )

    return h0
