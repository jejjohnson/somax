from jaxtyping import Array, Float
from somax._src.operators.difference import x_diff_2D, y_diff_2D
from somax._src.operators.average import center_avg_2D
from somax._src.constants import GRAVITY


def calculate_u_rhs(
        h: Float[Array, "Dx Dy"],
        v: Float[Array, "Dx Dy+1"],
        dx: float | Float[Array, "Dx"],
        coriolis_param: float=2e-4,
        gravity: float=GRAVITY,
    ) -> Float[Array, "Dx-1 Dy"]:
    """
    Calculate the right-hand side (RHS) of the u-component of the velocity field.

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
        h: Float[Array, "Dx Dy"],
        u: Float[Array, "Dx+1 Dy"],
        dy: float | Float[Array, "Dy"],
        coriolis_param: float=2e-4,
        gravity: float=GRAVITY,
) -> Float[Array, "Dx Dy-1"]:
    """
    Calculate the right-hand side (RHS) of the v-equation in a linear shallow water model.

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

    v_rhs = - coriolis_param * u_on_v - gravity * dh_dy
    return v_rhs


def calculate_h_rhs(
        u: Float[Array, "Dx+1 Dy"],
        v: Float[Array, "Dx Dy+1"],
        dx: float | Float[Array, "Dx"],
        dy: float | Float[Array, "Dy"],
        depth: float=100.0,
) -> Float[Array, "Dx Dy"]:
    """
    Calculates the right-hand side of the height equation for shallow water models.
    
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
    h_rhs = - depth * (du_dx + dv_dy)
    return h_rhs
