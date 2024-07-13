from typing import Optional
import jax.numpy as jnp
from jaxtyping import (
    Array,
    ArrayLike,
    Float,
    PyTree,
)
from somax._src.masks.masks import FaceMask
from somax._src.constants import GRAVITY
from somax._src.domain.cartesian import CartesianDomain2D

# from somax._src.models.linear_swm.model import LinearSWM
from somax._src.operators.average import center_avg_2D
from somax._src.operators.difference import (
    x_diff_2D,
    y_diff_2D,
    curl_2D
)
from somax._src.reconstructions.base import reconstruct
from somax._src.boundaries.base import (
    zero_boundaries,
    no_flux_boundaries, 
    no_energy_boundaries,
    zero_gradient_boundaries
)


def calculate_u_linear_rhs(
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


def calculate_u_nonlinear_rhs(
    h: Float[Array, "Nx Ny"],
    q: Float[Array, "Nx+1 Ny+1"],
    vh_flux: Float[Array, "Nx Ny+1"],
    ke: Float[Array, "Nx Ny"],
    dx: float | Array,
    num_pts: int = 3,
    method: str = "wenoz",
    u_mask: Optional[FaceMask] = None,
    gravity: float=GRAVITY
):
    """
    Eq:
        work = g ∂h/∂x
        ke = 0.5 (u² + v²)
        ∂u/∂t = qhv - work - ke

    Notes:
        - uses reconstruction (5pt, improved weno) of q on vh flux
    """

    # pad arrays
    h_pad: Float[Array, "Nx+2 Ny"] = zero_gradient_boundaries(
        h, ((1, 1), (0, 0))
    )
    ke_pad: Float[Array, "Nx+2 Ny"] = no_energy_boundaries(ke, ((1, 1), (0, 0)))

    vh_flux_on_u: Float[Array, "Nx-1 Ny"] = center_avg_2D(vh_flux)

    # no flux padding
    vh_flux_on_u: Float[Array, "Nx+2 Ny"] = zero_gradient_boundaries(
        vh_flux_on_u, ((1, 1), (0, 0))
    )
    vh_flux_on_u = vh_flux_on_u.at[1:-1].set(-vh_flux_on_u[1:-1])

    qhv_flux_on_u: Float[Array, "Nx+1 Ny"] = reconstruct(
        q=q,
        u=vh_flux_on_u,
        u_mask=u_mask if u_mask is not None else None,
        dim=1,
        method=method,
        num_pts=num_pts,
    )

    # apply mask
    if u_mask is not None:
        qhv_flux_on_u *= u_mask.values

    # calculate work
    dh_dx: Float[Array, "Nx+1 Ny"] = x_diff_2D(h_pad, step_size=dx)
    work = gravity * dh_dx

    # calculate kinetic energy
    dke_on_u: Float[Array, "Nx+1 Ny"] = x_diff_2D(ke_pad, step_size=dx)
    # calculate u RHS
    u_rhs: Float[Array, "Nx+1 Ny"] = -work + qhv_flux_on_u - dke_on_u

    # apply mask
    if u_mask is not None:
        u_rhs *= u_mask.values

    return u_rhs


def calculate_v_linear_rhs(
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


def calculate_v_nonlinear_rhs(
    h: Float[Array, "Nx Ny"],
    q: Float[Array, "Nx+1 Ny+1"],
    uh_flux: Float[Array, "Nx+1 Ny"],
    ke: Float[Array, "Nx Ny"],
    dy: float | Array,
    num_pts: int = 3,
    method: str = "wenoz",
    v_mask: Optional[FaceMask] = None,
    gravity=GRAVITY
):
    """
    Eq:
        work = g ∂h/∂y
        ke = 0.5 (u² + v²)
        ∂v/∂t = - qhu - work - ke

    Notes:
        - uses reconstruction (5pt, improved weno) of q on uh flux
    """
    h_pad: Float[Array, "Nx Ny+2"] = zero_gradient_boundaries(
        h, ((0, 0), (1, 1))
    )
    ke_pad: Float[Array, "Nx Ny+2"] = no_energy_boundaries(ke, ((0, 0), (1, 1)))

    uh_flux_on_v: Float[Array, "Nx Ny-1"] = center_avg_2D(uh_flux)

    # assume no flux on boundaries
    uh_flux_on_v: Float[Array, "Nx Ny+2"] = zero_gradient_boundaries(
        uh_flux_on_v, ((0, 0), (1, 1))
    )
    uh_flux_on_v = uh_flux_on_v.at[1:-1].set(-uh_flux_on_v[1:-1])

    qhu_flux_on_v: Float[Array, "Nx Ny+1"] = reconstruct(
        q=q,
        u=uh_flux_on_v,
        u_mask=v_mask if v_mask is not None else None,
        dim=0,
        method=method,
        num_pts=num_pts,
    )

    # apply masks
    if v_mask is not None:
        qhu_flux_on_v *= v_mask.values

    # calculate work
    dh_dy: Float[Array, "Nx Ny+1"] = y_diff_2D(h_pad, step_size=dy)
    work = gravity * dh_dy

    # calculate kinetic energy
    dke_on_v: Float[Array, "Nx Ny+1"] = y_diff_2D(ke_pad, step_size=dy)

    # calculate u RHS
    v_rhs: Float[Array, "Nx Ny+1"] = -work - qhu_flux_on_v - dke_on_v

    # apply masks
    if v_mask is not None:
        v_rhs *= v_mask.values

    return v_rhs


def calculate_h_linear_rhs(
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


def calculate_h_nonlinear_rhs(
    uh_flux: Float[ArrayLike, "Dx+1 Dy"],
    vh_flux: Float[ArrayLike, "Dx Dy+1"],
    dx: float | Float[ArrayLike, "Dx"],
    dy: float | Float[ArrayLike, "Dy"],
    depth: float | ArrayLike = 100.0,
) -> Float[ArrayLike, "Dx Dy"]:
    """
    Calculates the right-hand side of the height equation for shallow water models.

    Eq:
        ∂h/∂t + ∂/∂x((H+h)u) + ∂/∂y((H+h)v) = 0

    Args:
        uh_flux (Float[ArrayLike, "Dx+1 Dy"]): The x-component of the velocity flux.
        vh_flux (Float[ArrayLike, "Dx Dy+1"]): The y-component of the velocity flux.
        dx (float | Float[ArrayLike, "Dx"]): The grid spacing in the x-direction.
        dy (float | Float[ArrayLike, "Dy"]): The grid spacing in the y-direction.
        depth (float | ArrayLike, optional): The depth of the water. Defaults to 100.0.

    Returns:
        Float[ArrayLike, "Dx Dy"]: The right-hand side of the height equation.
    """
    du_dx: Float[ArrayLike, "Dx Dy"] = x_diff_2D(uh_flux, step_size=dx)
    dv_dy: Float[ArrayLike, "Dx Dy"] = y_diff_2D(vh_flux, step_size=dy)
    h_rhs = -depth * (du_dx + dv_dy)
    return h_rhs


def potential_vorticity(
        h: Float[Array, "Nx+1 Ny+1"],
        u: Float[Array, "Nx+1 Ny"], 
        v: Float[Array, "Nx-1 Ny+1"],
        f: Float[Array, "Nx Ny"],
        dx: float | Float[ArrayLike, "Dx"],
        dy: float | Float[ArrayLike, "Dy"],
) -> Array:
    """
    Eq:
        ζ = ∂v/∂x - ∂u/∂y
        q = (ζ + f) / h
    """

    # relative vorticity, ζ = dv/dx - du/dy
    vort_r = curl_2D(u=u, v=v, step_size_x=dx, step_size_y=dy)

    # potential vorticity, q = (ζ + f) / h
    h_on_q = center_avg_2D(h)
    q: Float[Array, "Nx-1 Ny-1"] = (vort_r + f[1:-1,1:-1]) / h_on_q[1:-1,1:-1]

    return q