import jax.numpy as jnp
from jaxtyping import Array
from somax._src.domain.cartesian import CartesianDomain2D
from somax._src.models.linear_swm.model import LinearSWM
from somax._src.field.base import Field


def initial_h0(h_domain: CartesianDomain2D, model: LinearSWM) -> Field:

    XY = h_domain.grid
    X, Y = XY[..., 0], XY[..., 1]

    Nx, Ny = h_domain.shape

    x = h_domain.coords_x
    y = h_domain.coords_y

    h0 = model.depth + 1.0 * jnp.exp(
    -((X - x[Nx // 2]) ** 2) / model.rossby_radius**2
    - (Y - y[Ny - 2]) ** 2 / model.rossby_radius**2
    )
    return Field(h0, h_domain)


def initial_zeros(domain: CartesianDomain2D) -> Field:

    return Field(jnp.zeros(domain.shape), domain)
