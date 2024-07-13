import jax.numpy as jnp
from jaxtyping import (
    Array,
    ArrayLike,
    Float,
    PyTree
)

from somax._src.constants import GRAVITY
from somax._src.domain.cartesian import CartesianDomain2D
# from somax._src.models.linear_swm.model import LinearSWM
from somax._src.operators.average import center_avg_2D
from somax._src.operators.difference import (
    x_diff_2D,
    y_diff_2D,
)


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


def calculate_u_rhs(
    h: Float[ArrayLike, "Dx Dy"],
    v: Float[ArrayLike, "Dx Dy+1"],
    dx: ArrayLike | Float[ArrayLike, "Dx-1"],
    coriolis_param: float | ArrayLike = 2e-4,
    gravity: float | ArrayLike = GRAVITY,
) -> Float[ArrayLike, "Dx-1 Dy"]:
    """
    Calculate the right-hand side (RHS) of the u-component of the velocity field.

    Eq:
        ∂u/∂t = fv - g ∂h/∂x

    Args:
        h (Float[Array, "Dx Dy"]): The height field.
        v (Float[Array, "Dx Dy+1"]): The v-component of the velocity field.
        dx (float | Float[Array, "Dx"]): The grid spacing in the x-direction.
        coriolis_param (float): The Coriolis parameter.
        gravity (float, optional): The acceleration due to gravity. Defaults to GRAVITY.

    Returns:
        Float[Array, "Dx-1 Dy"]: The RHS of the u-component of the velocity field.
    """

    # average
    v_on_u = center_avg_2D(v)
    dh_dx = x_diff_2D(h, step_size=dx, method="forward")

    u_rhs = coriolis_param * v_on_u - gravity * dh_dx

    return u_rhs


def calculate_v_rhs(
    h: Float[ArrayLike, "Dx Dy"],
    u: Float[ArrayLike, "Dx+1 Dy"],
    dy: float | Float[ArrayLike, "Dy"],
    coriolis_param: float | ArrayLike = 2e-4,
    gravity: float | ArrayLike = GRAVITY,
) -> Float[ArrayLike, "Dx Dy-1"]:
    """
    Calculate the right-hand side (RHS) of the v-equation in a linear shallow water model.

    Eq:
        ∂v/∂t = - fu - g ∂h/∂y

    Args:
        h (Float[Array, "Dx Dy"]): Array representing the height of the water.
        u (Float[Array, "Dx+1 Dy"]): Array representing the x-component of the velocity.
        dy (float | Float[Array, "Dy"]): Step size in the y-direction.
        coriolis_param (float, optional): Coriolis parameter. Defaults to 2e-4.
        gravity (float, optional): Gravity constant. Defaults to GRAVITY.

    Returns:
        Float[Array, "Dx Dy-1"]: Array representing the RHS of the v-equation.
    """

    u_on_v = center_avg_2D(u)
    dh_dy = y_diff_2D(h, step_size=dy)

    v_rhs = -coriolis_param * u_on_v - gravity * dh_dy
    return v_rhs


def calculate_h_rhs(
    u: Float[ArrayLike, "Dx+1 Dy"],
    v: Float[ArrayLike, "Dx Dy+1"],
    dx: float | Float[ArrayLike, "Dx"],
    dy: float | Float[ArrayLike, "Dy"],
    depth: float | ArrayLike = 100.0,
) -> Float[ArrayLike, "Dx Dy"]:
    """
    Calculates the right-hand side of the height equation for shallow water models.

    Eq:
       ∂h/∂t = - H (∂u/∂x + ∂v/∂y)

    Args:
        u (Float[Array, "Dx+1 Dy"]): The x-component of the velocity field.
        v (Float[Array, "Dx Dy+1"]): The y-component of the velocity field.
        dx (float | Float[Array, "Dx"]): The grid spacing in the x-direction.
        dy (float | Float[Array, "Dy"]): The grid spacing in the y-direction.
        depth (float, optional): The depth of the water. Defaults to 100.0.

    Returns:
        Float[Array, "Dx Dy"]: The right-hand side of the height equation.
    """

    du_dx = x_diff_2D(u, step_size=dx)
    dv_dy = y_diff_2D(v, step_size=dy)
    h_rhs = -depth * (du_dx + dv_dy)
    return h_rhs
