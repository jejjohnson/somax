import math
import finitediffx as fdx
from finitevolx import (
    avg_pool,
    center_avg_2D,
)
import numpy as np
import jax.numpy as jnp
from fieldx._src.domain.domain import Domain
from jaxtyping import Float, Array


def init_tau(domain, tau0: float = 2.0e-5):
    """
    Args
    ----
        tau0 (float): wind stress magnitude m/s^2
            default=2.0e-5"""
    # initial TAU
    tau = np.zeros((2, domain.Nx[0], domain.Nx[1]))

    # create staggered coordinates (y-direction)
    y_coords = np.arange(domain.Nx[1]) + 0.5

    # create tau
    tau[0, :, :] = -tau0 * np.cos(2 * np.pi * (y_coords / domain.Nx[1]))

    return tau


def calculate_wind_forcing(tau, domain):
    # move from edges to nodes
    tau_x = avg_pool(tau[0], padding=((1, 0), (0, 0)), stride=1, mean_fn="arithmetic")
    tau_y = avg_pool(tau[1], padding=((0, 0), (1, 0)), stride=1, mean_fn="arithmetic")

    # compute finite difference
    # dF2dX = difference(tau_y, axis=0, step_size=domain.dx[0], )
    dF2dX = fdx.difference(
        tau_y, axis=0, step_size=domain.dx[0], accuracy=1, method="central"
    )
    dF1dY = fdx.difference(
        tau_x, axis=1, step_size=domain.dx[1], accuracy=1, method="central"
    )
    curl_stagg = dF2dX - dF1dY

    return center_avg_2D(curl_stagg.squeeze()[1:, 1:])


def calculate_wind_forcing(
    domain: Domain,
    H_0: float,
    tau0: float = 0.08 / 1_000.0,
) -> Float[Array, "Nx Ny"]:
    """
    Equation:
        F_wind: (τ₀ /H₀)(∂xτ−∂yτ)
    """

    Ly = domain.Lx[-1]

    # [Nx,Ny]
    y_coords = domain.grid_axis[-1]

    # center coordinates, cell centers
    # [Nx,Ny] --> [Nx-1,Ny-1]
    y_coords_center = center_avg_2D(y_coords)

    # calculate tau
    # analytical form! =]
    curl_tau = -tau0 * 2 * math.pi / Ly * jnp.sin(2 * math.pi * y_coords_center / Ly)

    # print_debug_quantity(curl_tau, "CURL TAU")

    wind_forcing = curl_tau / H_0

    return wind_forcing