import math

from fieldx._src.domain.domain import Domain
from finitevolx import (
    center_avg_2D,
    laplacian,
)
import jax
import jax.numpy as jnp
from jaxtyping import (
    Array,
    Float,
)


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


def calculate_bottom_drag(
    psi: Array,
    domain: Domain,
    H_z: float = 1.0,
    delta_ek: float = 2.0,
    f0: float = 9.375e-05,
    masks_psi=None,
) -> Array:
    """
    Equation:
        F_drag: (δₑf₀² / 2Hz)∇²ψN
    """

    # interior, vertex points on psi
    omega: Float[Array, "Nz Nx-2 Ny-2"] = jax.vmap(laplacian, in_axes=(0, None))(psi, domain.dx)

    # pad interior psi points
    omega: Float[Array, "Nz Nx Ny"]  = jnp.pad(
        omega, pad_width=((0, 0), (1, 1), (1, 1)), mode="constant", constant_values=0.0
    )

    if masks_psi is not None:
        omega *= masks_psi.values

    # plot_field(omega)
    # pad interior, center points on q
    # [Nx,Ny] --> [Nx-1,Ny-1]
    omega: Float[Array, "Nz Nx-1 Ny-1"] = jax.vmap(center_avg_2D)(omega)

    # calculate bottom drag coefficient
    bottom_drag_coeff: Float[Array, ""]  = delta_ek / H_z * f0 / 2.0

    # calculate bottom drag
    bottom_drag: Float[Array, "Nx Ny"]  = -bottom_drag_coeff * omega[-1]

    # plot_field(omega)
    return bottom_drag
