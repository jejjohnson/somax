import jax.numpy as jnp
from jaxtyping import Array
from somax._src.domain.cartesian import CartesianDomain2D
from somax._src.models.linear_swm.model import LinearSWM


def initial_h0(h_domain: CartesianDomain2D, model: LinearSWM) -> Array:

    XY = h_domain.grid
    X, Y = XY[..., 0], XY[..., 1]

    Nx, Ny = h_domain.shape

    x = h_domain.coords_x
    y = h_domain.coords_y

    return model.depth + 1.0 * jnp.exp(
    -((X - x[Nx // 2]) ** 2) / model.rossby_radius**2
    - (Y - y[Ny - 2]) ** 2 / model.rossby_radius**2
    )