from typing import (
    List,
    Tuple,
)
from functools import partial
import jax
import equinox as eqx
import finitediffx as fdx
import jax.numpy as jnp
from jaxtyping import (
    Array,
    ArrayLike,
    Float,
)
import kernex as kex

from somax._src.operators.difference import x_diff_1D, x_diff_2D, y_diff_2D


def laplacian_2D(
    u: Float[Array, "... Dx Dy"],
    step_size_x: float | Float[Array, "Dx-1"] = 1.0,
    step_size_y: float | Float[Array, "Dy-1"] = 1.0,
) -> Float[Array, "... Dx-2 Dy-2"]:
    """
    Compute the 2D Laplacian of the given scalar field.

    Parameters:
        u (Float[Array, "... Dx Dy"]): The scalar field to compute the Laplacian of.
        step_size_x (float | Float[Array, "Dx-1"]): The step size in the x-direction. Default is 1.0.
        step_size_y (float | Float[Array, "Dy-1"]): The step size in the y-direction. Default is 1.0.

    Returns:
        Float[Array, "... Dx-2 Dy-2"]: The computed 2D Laplacian of the scalar field.
    """
    d2u_dx2 = x_diff_2D(
        u=u,
        step_size=step_size_x,
        derivative=2,
    )
    d2u_dy2 = y_diff_2D(
        u=u,
        step_size=step_size_y,
        derivative=2,
    )
    return d2u_dx2[..., 1:-1] + d2u_dy2[..., 1:-1, :]


def perpendicular_gradient_2D(
    u: Float[Array, "Nx Ny"],
    step_size_x: float | Float[Array, "Dx-1"] = 1.0,
    step_size_y: float | Float[Array, "Dy-1"] = 1.0,
) -> tuple[Float[Array, "Nx Ny-1"], Float[Array, "Nx-1 Ny"]]:
    """
    Calculates the geostrophic gradient for a staggered grid.

    The geostrophic gradient is calculated using the following equations:
        u velocity = -∂yΨ
        v velocity = ∂xΨ

    Args:
        u (Array): The input variable.
            Size = [Nx,Ny]
        step_size_x (float | Array): The step size for the x-direction.
        step_size_y (float | Array): The step size for the y-direction.

    Returns:
        tuple[Float[Array, "Nx Ny-1"], Float[Array, "Nx-1 Ny"]]: A tuple containing the geostrophic velocities.
            - du_dy (Array): The geostrophic velocity in the y-direction.
                Size = [Nx,Ny-1]
            - du_dx (Array): The geostrophic velocity in the x-direction.
                Size = [Nx-1,Ny]

    Note:
        For the geostrophic velocity, we need to multiply the derivative in the x-direction by -1.
    """

    du_dy = y_diff_2D(u=u, step_size=step_size_y, derivative=1)
    du_dx = x_diff_2D(u=u, step_size=step_size_x, derivative=1)
    return -du_dy, du_dx


def divergence_2D(
    u: Float[Array, "Dx+1 Dy"],
    v: Float[Array, "Dx Dy+1"],
    step_size_x: float | Float[Array, "Dx"] = 1.0,
    step_size_y: float | Float[Array, "Dy"] = 1.0,
    method: str = "backward",
) -> Float[Array, "Dx Dy"]:
    """Calculates the divergence for a staggered grid

    This function calculates the divergence of a vector field on a staggered grid.
    The divergence represents the net flow of a vector field out of a given point.
    It is calculated as the sum of the partial derivatives of the vector field with respect to each coordinate.

    Equation:
        ∇⋅u̅  = ∂x(u) + ∂y(v)

    Args:
        u (Array): The input array for the u direction. Size = [Nx+1, Ny]
        v (Array): The input array for the v direction. Size = [Nx, Ny-1]
        step_size_x (float | Array, optional): The step size for the x-direction. Defaults to 1.0.
        step_size_y (float | Array, optional): The step size for the y-direction. Defaults to 1.0.

    Returns:
        div (Array): The divergence of the vector field. Size = [Nx-1, Ny-1]
    """
    # ∂xu
    dudx = x_diff_2D(u=u, step_size=step_size_x, derivative=1, method=method)
    # ∂yv
    dvdx = y_diff_2D(u=v, step_size=step_size_y, derivative=1, method=method)

    return dudx + dvdx


def curl_2D(
    u: Float[Array, "Dx+1 Dy"],
    v: Float[Array, "Dx Dy+1"],
    step_size_x: float | Float[Array, "Dx"] = 1.0,
    step_size_y: float | Float[Array, "Dy"] = 1.0,
    method: str = "backward",
) -> Float[Array, "Nx-1 Ny-1"]:
    """
    Calculates the curl by using finite difference in the y and x direction for the u and v velocities respectively.

    Eqn:
        ζ = ∂v/∂x - ∂u/∂y

    Args:
        u (Array): The input array for the u direction. Size = [Nx, Ny-1].
        v (Array): The input array for the v direction. Size = [Nx-1, Ny].
        step_size_x (float | Array, optional): The step size for the x-direction. Defaults to 1.0.
        step_size_y (float | Array, optional): The step size for the y-direction. Defaults to 1.0.
        method (str, optional): The method to use for finite difference calculation. Defaults to "backward".

    Returns:
        zeta (Array): The relative vorticity. Size = [Nx-1, Ny-1].
    """
    # ∂xv
    dv_dx: Float[Array, "Dx-1 Dy+1"] = x_diff_2D(u=v, step_size=step_size_x, derivative=1, method=method)
    # ∂yu
    du_dy: Float[Array, "Dx+1 Dy-1"] = y_diff_2D(u=u, step_size=step_size_y, derivative=1, method=method)

    # slice appropriate axes
    dv_dx: Float[Array, "Nx-1 Ny-1"] = dv_dx[:, 1:-1]
    du_dy: Float[Array, "Nx-1 Ny-1"] = du_dy[1:-1, :]

    return dv_dx - du_dy


def det_jacobian(f, g, dx, dy):
    """Arakawa discretisation of Jacobian J(f,g).
    Scalar fields f and g must have the same dimension.
    Grid is regular and dx = dy."""

    dx_f = f[..., 2:, :] - f[..., :-2, :]
    dx_g = g[..., 2:, :] - g[..., :-2, :]
    dy_f = f[..., 2:] - f[..., :-2]
    dy_g = g[..., 2:] - g[..., :-2]

    return (
        (
            dx_f[..., 1:-1] * dy_g[..., 1:-1, :]
            - dx_g[..., 1:-1] * dy_f[..., 1:-1, :]
        )
        + (
            (
                f[..., 2:, 1:-1] * dy_g[..., 2:, :]
                - f[..., :-2, 1:-1] * dy_g[..., :-2, :]
            )
            - (
                f[..., 1:-1, 2:] * dx_g[..., 2:]
                - f[..., 1:-1, :-2] * dx_g[..., :-2]
            )
        )
        + (
            (
                g[..., 1:-1, 2:] * dx_f[..., 2:]
                - g[..., 1:-1, :-2] * dx_f[..., :-2]
            )
            - (
                g[..., 2:, 1:-1] * dy_f[..., 2:, :]
                - g[..., :-2, 1:-1] * dy_f[..., :-2, :]
            )
        )
    ) / (12.0 * dx * dy)