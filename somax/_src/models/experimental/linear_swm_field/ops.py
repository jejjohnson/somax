from jaxtyping import Array, Float, ArrayLike
from somax._src.operators.difference import x_diff_2D, y_diff_2D
from somax._src.operators.average import center_avg_2D
from somax._src.constants import GRAVITY
from plum import dispatch
import jax
import jax.numpy as jnp
from functools import partial
from somax._src.field.base import (
    Field, pad_x_field, pad_y_field, interp_domain_field, x_diff_2D_field, y_diff_2D_field,
    interp_cart_domain_field,
    interp_reg_domain_field
)
from somax._src.domain.base import Domain


def calculate_u_rhs(
    h: Field,
    v: Field,
    u_domain: Domain,
    coriolis_param: float | ArrayLike = 2e-4,
    gravity: float | ArrayLike = GRAVITY,
    ) -> Field:
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
    # v_on_u = interp_domain_field(v, u_domain, extrap=False)
    v_on_u = interp_reg_domain_field(v, u_domain,  method="linear", fill_value=jnp.nan)

    # calculate difference

    dh_dx = x_diff_2D_field(h, derivative=1, method="backward")
    # dh_dx = h.diff_x(derivative=1, method="backward")
    # dh_dx = dh_dx.interp_domain(u.domain, extrap=False)
    # dh_dx = interp_domain_field(dh_dx, u_domain, extrap=False)
    dh_dx = interp_reg_domain_field(dh_dx, u_domain,  method="linear", fill_value=jnp.nan)

    u_rhs = coriolis_param * v_on_u - gravity * dh_dx

    return Field(u_rhs.values, u_domain)


def calculate_v_rhs(
    h: Field,
    u: Field,
    v_domain: Domain,
    coriolis_param: float | ArrayLike = 2e-4,
    gravity: float | ArrayLike = GRAVITY,
    ) -> Field:
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
    # u_on_v = u.interp_domain(v_domain, extrap=False)
    # u_on_v = interp_domain_field(u, v_domain, extrap=False)
    u_on_v = interp_reg_domain_field(u, v_domain, method="linear", fill_value=jnp.nan)
    
    # dh_dy = h.diff_y(derivative=1, method="backward")
    dh_dy = y_diff_2D_field(h, derivative=1, method="backward")
    # dh_dy = dh_dy.interp_domain(v.domain, extrap=False)
    # dh_dy = interp_domain_field(dh_dy, v_domain, method="linear", extrap=False)
    dh_dy = interp_reg_domain_field(dh_dy, v_domain, method="linear", fill_value=jnp.nan)

    v_rhs = - coriolis_param * u_on_v - gravity * dh_dy

    return Field(v_rhs.values, v_domain)



def calculate_h_rhs(
    u: Field,
    v: Field,
    h_domain: Domain,
    depth: float | ArrayLike = 100.0,
) -> Field:
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
    
    du_dx = x_diff_2D_field(u, derivative=1, method="backward") 
    dv_dy = y_diff_2D_field(v, derivative=1, method="backward")

    # du_dx = interp_domain_field(du_dx, h_domain, method="linear", extrap=False)
    # dv_dy = interp_domain_field(dv_dy, h_domain, method="linear", extrap=False)
    du_dx = interp_reg_domain_field(du_dx, h_domain,  method="linear", fill_value=jnp.nan)
    dv_dy = interp_reg_domain_field(dv_dy, h_domain,  method="linear", fill_value=jnp.nan)

    h_rhs = - depth * (du_dx + dv_dy)

    return Field(h_rhs.values, h_domain)
