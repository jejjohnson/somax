from typing import Optional

import numpy as np
import jax.numpy as jnp
from jaxtyping import (
    Array,
    ArrayLike,
    Float,
    PyTree,
)

from somax._src.domain.cartesian import CartesianDomain2D
from somax._src.models.sw.params import SWMParams
from somax.interp import x_avg_2D
from somax._src.constants import GRAVITY


def initial_h0(h_domain: CartesianDomain2D, model: PyTree) -> Array:
    """
    Calculate the initial value of h0 for the LinearSWM model.

    Args:
        h_domain (CartesianDomain2D): The Cartesian domain for h0.
        model (LinearSWM): The LinearSWM model.

    Returns:
        Array: The initial value of h0.

    """
    XY = h_domain.grid
    X, Y = XY[..., 0], XY[..., 1]

    Nx, Ny = h_domain.shape

    x = h_domain.coords_x
    y = h_domain.coords_y

    return model.depth + 1.0 * jnp.exp(
        -((X - x[Nx // 2]) ** 2) / model.rossby_radius**2
        - (Y - y[Ny - 2]) ** 2 / model.rossby_radius**2
    )


def init_partition(domain: CartesianDomain2D):
    Y = domain.grid[..., 1]
    y = domain.coords_y
    Ny = domain.shape[1]
    Lx = domain.size[0]
    u0 = 10 * jnp.exp(-((Y - y[Ny // 2]) ** 2) / (0.02 * Lx) ** 2)
    return u0


def init_h0_jet(
    domain: CartesianDomain2D, params: SWMParams, u0: Optional[Array] = None
):
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
        + 0.2 * jnp.sin(X / Lx * 10.0 * jnp.pi) * jnp.cos(Y / Ly * 8.0 * jnp.pi)
    )

    return h0


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


def init_h0_uniform_westerly(domain, model):
    mean_wind_speed = 20.0 # m/s
    f0 = model.coriolis_f0
    g = GRAVITY
    beta = model.coriolis_beta
    Y = domain.grid[..., 1]

    h0 = 10_000. -(mean_wind_speed*f0/g)*Y
    h0 -= - 0.5*(mean_wind_speed*beta/g)*Y**2

    return h0


def init_h0_sinusoidal(domain, model):
    mean_wind_speed = 20.0 # m/s
    g = GRAVITY
    beta = model.coriolis_beta
    y = domain.coords_y
    Y = domain.grid[..., 1]

    h0 = 10000.-350.*jnp.sin(Y/(jnp.max(y))*2.0*jnp.pi);

    return h0


def init_h0_equitorial_easterly(domain, model):
    """
    Initialize the initial condition for the equatorial easterly component of the height field.

    Args:
        domain: The domain object containing the grid information.
        model: The model object containing additional information.

    Returns:
        h0: The initial condition for the equatorial easterly component of the height field.
    """
    y = domain.coords_y
    Y = domain.grid[..., 1]

    h0 = 10000. - 50.*jnp.cos(Y*2*jnp.pi/jnp.max(y))

    return h0


def init_h0_zonal_jet(domain, model):
    y = domain.coords_y
    Y = domain.grid[..., 1]

    h0 = 10_000. - jnp.tanh(20.0 * ((Y - jnp.mean(y)) / jnp.max(y))) * 400.0

    return h0


def init_h0_gauss_blob(domain, model):
    dx, dy = domain.resolution
    y = domain.coords_y
    x = domain.coords_x
    XY = domain.grid
    X, Y = XY[..., 0], XY[..., 1]
    std_blob = 8.0 * dy # Standard deviation of blob (m)

    h0 = 9750.
    h0 += 1000. * jnp.exp(
        -(
            (X-0.25*jnp.mean(x))**2. +
            (Y-jnp.mean(y))**2.
            ) /
        (
            2.0 * std_blob**2.
            ))

    return h0


def init_h0_gauss_low(domain, model):
    dx, dy = domain.resolution
    y = domain.coords_y
    x = domain.coords_x
    XY = domain.grid
    X, Y = XY[..., 0], XY[..., 1]
    std_blob = 8.0 * dy # Standard deviation of blob (m)

    h0 = 9750.
    h0 -= 1000. * jnp.exp(
        -(
            (X-0.25*jnp.mean(x))**2. +
            (Y-jnp.mean(y))**2.
            ) /
        (
            2.0 * std_blob**2.
            ))

    return h0


def init_h0_cyclone_westerly(domain, model):
    dx, dy = domain.resolution
    f0 = model.coriolis_f0
    g = GRAVITY
    beta = model.coriolis_beta
    y = domain.coords_y
    x = domain.coords_x
    XY = domain.grid
    X, Y = XY[..., 0], XY[..., 1]

    mean_wind_speed = 25.0 # m/s
    std_blob = 7.0 * dy # Standard deviation of blob (m)
    h0 = 10000.-(mean_wind_speed*f0/g)*(Y-jnp.mean(y))
    h0 -= 0.5*(mean_wind_speed*beta/g)*Y**2
    h0 -= 500.*jnp.exp(-((X-0.5*jnp.mean(x))**2.+(Y-jnp.mean(y))**2.)/(2.*std_blob**2.))

    return h0


def init_h0_cyclone_westerly_v2(domain, model):
    dx, dy = domain.resolution
    f0 = model.coriolis_f0
    g = GRAVITY
    beta = model.coriolis_beta
    y = domain.coords_y
    x = domain.coords_x
    XY = domain.grid
    X, Y = XY[..., 0], XY[..., 1]

    max_wind_speed = 20.0 # m/s
    std_blob = 7.0 * dy # Standard deviation of blob (m)
    h0 = 10250.-(max_wind_speed*f0/g)*(Y-jnp.mean(y))**2./jnp.max(y)
    h0 -= 1000.*jnp.exp(
        -(0.25*(X-1.5*jnp.mean(x))**2. +
          (Y-0.5*jnp.mean(y))**2.
          ) /
        (2.*std_blob**2.)
        )

    return h0


def init_h0_step(domain, model):
    dx, dy = domain.resolution
    y = domain.coords_y
    x = domain.coords_x
    XY = domain.grid
    X, Y = XY[..., 0], XY[..., 1]
    h0 = 9750. * np.ones(domain.shape)
    h0[np.where((X<np.max(x)/5.) & (Y>np.max(y)/10.) & (Y<np.max(y)*0.9))] = 10500.
    return h0


def init_h0_sharp_shear(domain, model):
    g = GRAVITY
    f0 = model.coriolis_f0
    y = domain.coords_y
    Y = domain.grid[..., 1]
    mean_wind_speed = 50. # m/s
    h0 = (mean_wind_speed*f0/g)*np.abs(Y-np.mean(y))
    h0 = 1000 + h0-np.mean(h0)
    return h0

#     elif initial_conditions == REANALYSIS:
#        mat_contents = sio.loadmat('reanalysis.mat')
#        pressure = mat_contents['pressure'];
#        height = 0.99*pressure/g;
