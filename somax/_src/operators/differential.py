import finitediffx as fdx
import jax
from jaxtyping import (
    Array,
    Float,
)

from somax._src.constants import GRAVITY
from somax.interp import (
    x_avg_2D,
    y_avg_2D,
)


def difference(
    u: Float[Array, "... D"], axis: int = 0, step_size: float = 1.0, derivative: int = 1
) -> Array:
    if derivative == 1:
        du: Float[Array, "... D-1"] = fdx.difference(
            u,
            step_size=step_size,
            axis=axis,
            accuracy=1,
            derivative=derivative,
            method="backward",
        )
        du = jax.lax.slice_in_dim(du, axis=axis, start_index=1, limit_index=None)
    elif derivative == 2:
        du: Float[Array, "... D-2"] = fdx.difference(
            u,
            step_size=step_size,
            axis=axis,
            accuracy=1,
            derivative=derivative,
            method="central",
        )
        du = jax.lax.slice_in_dim(du, axis=axis, start_index=1, limit_index=-1)
    else:
        msg = "Derivative must be 1 or 2"
        raise ValueError(msg)

    return du


def laplacian(u: Array, step_size: float | tuple[float, ...] | Array = 1) -> Array:
    msg = "Laplacian must be 2D or 3D"
    assert u.ndim in [2, 3], msg
    # calculate laplacian
    lap_u = fdx.laplacian(array=u, accuracy=1, step_size=step_size)

    # remove external dimensions
    lap_u = lap_u[1:-1, 1:-1]

    if u.ndim == 3:
        lap_u = lap_u[..., 1:-1]

    return lap_u


# ====
# TAKS - perpendicular gradient vs geostrophic gradient
# ====
def geostrophic_gradient(
    u: Float[Array, "Nx Ny"],
    dx: float | Array,
    dy: float | Array,
) -> tuple[Float[Array, "Nx Ny-1"], Float[Array, "Nx-1 Ny"]]:
    """Calculates the geostrophic gradient for a staggered grid

    Equation:
        u velocity = -∂yΨ
        v velocity = ∂xΨ

    Args:
        u (Array): the input variable
            Size = [Nx,Ny]
        dx (float | Array): the stepsize for the x-direction
        dy (float | Array): the stepsize for the y-direction

    Returns:
        du_dy (Array): the geostrophic velocity in the y-direction
            Size = [Nx,Ny-1]
        du_dx (Array): the geostrophic velocity in the x-direction
            Size = [Nx-1,Ny]

    Note:
        for the geostrophic velocity, we need to multiply the
        derivative in the x-direction by negative 1.
    """

    du_dy = difference(u=u, axis=1, step_size=dy, derivative=1)
    du_dx = difference(u=u, axis=0, step_size=dx, derivative=1)
    return - du_dy, du_dx


def divergence(
    u: Float[Array, "Nx+1 Ny"], 
    v: Float[Array, "Nx Ny+1"], 
    dx: float, dy: float) -> Array:
    """Calculates the divergence for a staggered grid

    Equation:
        ∇⋅u̅  = ∂x(u) + ∂y(v)

    Args:
        u (Array): the input array for the u direction
            Size = [Nx+1, Ny]
        v (Array): the input array for the v direction
            Size = [Nx, Ny-1]
        dx (float | Array): the stepsize for the x-direction
        dy (float | Array): the stepsize for the y-direction

    Returns:
        div (Array): the divergence
            Size = [Nx-1,Ny-1]

    """
    # ∂xu
    dudx: Float[Array, "Nx Ny"] = difference(u=u, axis=-2, step_size=dx, derivative=1)
    # ∂yv
    dvdx: Float[Array, "Nx Ny"]  = difference(u=v, axis=-1, step_size=dy, derivative=1)

    return dudx + dvdx


def relative_vorticity(
    u: Float[Array, "Nx+1 Ny"],
    v: Float[Array, "Nx Ny+1"],
    dx: float | Array,
    dy: float | Array,
) -> Float[Array, "Nx-1 Ny-1"]:
    """Calculates the relative vorticity by using
    finite difference in the y and x direction for the
    u and v velocities respectively

    Eqn:
        ζ = ∂v/∂x - ∂u/∂y

    Args:
        u (Array): the input array for the u direction
            Size = [Nx, Ny-1]
        v (Array): the input array for the v direction
            Size = [Nx-1, Ny]
        dx (float | Array): the stepsize for the x-direction
        dy (float | Array): the stepsize for the y-direction

    Returns:
        zeta (Array): the relative vorticity
            Size = [Nx-1,Ny-1]
    """
    # ∂xv
    dv_dx: Float[Array, "Nx-1 Ny+1"] = difference(
        u=v, axis=-2, step_size=dx, derivative=1
    )
    # ∂yu
    du_dy: Float[Array, "Nx+1 Ny-1"] = difference(
        u=u, axis=-1, step_size=dy, derivative=1
    )

    # slice approprate axies
    dv_dx: Float[Array, "Nx-1 Ny-1"] = dv_dx[..., 1:-1]
    du_dy: Float[Array, "Nx-1 Ny-1"] = du_dy[..., 1:-1, :]
    
    return dv_dx - du_dy


def kinetic_energy(
    u: Float[Array, "Nx+1 Ny"],
    v: Float[Array, "Nx Ny+1"],
) -> Float[Array, "Nx Ny"]:
    """Calculates the kinetic energy on the
    center points
    Eq:
        ke = 0.5 (u² + v²)

    Args:
        u (Array): the u-velocity for the input array on the
            left-right cell faces.
            Size = [Nx+1 Ny]
        v (Array): the v-velocity for the input array on the
            top-down cell faces.
            Size = [Nx Ny+1]

    Returns:
        ke (Array): the kinetic energy on the cell center
            Size = [Nx Ny]
    """

    # calculate squared components on cell centers
    u2_on_center: Float[Array, "Nx-1 Ny-1"] = x_avg_2D(u**2)
    v2_on_center: Float[Array, "Nx-1 Ny-1"] = y_avg_2D(v**2)

    # calculate kinetic energy
    ke = 0.5 * (u2_on_center + v2_on_center)
    return ke

# TAKS - F + Relative Vorticity
def absolute_vorticity(
    u: Float[Array, "Nx+1 Ny"],
    v: Float[Array, "Nx Ny+1"],
    dx: float | Array,
    dy: float | Array,
) -> Float[Array, "Nx-1 Ny-1"]:
    """Calculates the relative vorticity by using
    finite difference in the y and x direction for the
    u and v velocities respectively

    Eqn:
        ζ = ∂v/∂x + ∂u/∂y

    Args:
        u (Array): the input array for the u direction
            Size = [Nx, Ny-1]
        v (Array): the input array for the v direction
            Size = [Nx-1, Ny]
        dx (float | Array): the stepsize for the x-direction
        dy (float | Array): the stepsize for the y-direction

    Returns:
        zeta (Array): the absolute vorticity
            Size = [Nx-1,Ny-1]
    """
    # ∂xv
    dv_dx: Float[Array, "Nx-1 Ny-1"] = difference(
        u=v, axis=-2, step_size=dx, derivative=1
    )
    # ∂yu
    du_dy: Float[Array, "Nx-1 Ny-1"] = difference(
        u=u, axis=-1, step_size=dy, derivative=1
    )

    return dv_dx + du_dy


def bernoulli_potential(
    h: Float[Array, "Nx Ny"],
    u: Float[Array, "Nx+1 Ny"],
    v: Float[Array, "Nx Ny+1"],
    gravity: float = GRAVITY,
) -> Float[Array, "Nx Ny"]:
    """Calculates the Bernoulli work

    Eq:
        p = ke + gh
    where:
        ke = 0.5 (u² + v²)

    Args:
        h (Array): the height located at the center/nodes
            Size = [Nx, Ny]
        u (Array): the velocity located on the east/north faces
            Size = [Nx+1 Ny]
        v (Array): the velocity located on the north/east faces
            Size = [Nx Ny+1]
        gravity (float): the acceleration due to gravity constant
            Default = 9.81

    Returns:
        p (Array): the Bernoulli work
            Size = [Nx Ny]
    Example:
        >>> u, v, h = ...
        >>> p = bernoulli_potential(h=h, u=u, v=v)
    """

    # calculate kinetic energy
    ke: Float[Array, "Nx Ny"] = kinetic_energy(u=u, v=v)

    # calculate Berunoulli potential
    p: Float[Array, "Nx Ny"] = ke + gravity * h

    return p

laplacian_batch = jax.vmap(laplacian, in_axes=(0, None))