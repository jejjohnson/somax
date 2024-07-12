from jaxtyping import Float, Array
from somax._src.constants import GRAVITY
import jax.numpy as jnp


def speed(
    u: Array,
    v: Array,
) -> Array:
    """
    Calculates the kinetic energy on the center points.

    Eq:
        ke = sqrt (u² + v²)

    Args:
        u (Array): The u-velocity for the input array on the left-right cell faces.
            Size = [Nx+1, Ny]
        v (Array): The v-velocity for the input array on the top-down cell faces.
            Size = [Nx, Ny+1]

    Returns:
        ke (Array): The kinetic energy on the cell center.
            Size = [Nx, Ny]
    """
    # calculate kinetic energy
    return jnp.hypot(u, v)


def kinetic_energy(
    u: Array,
    v: Array,
) -> Array:
    """
    Calculates the kinetic energy on the center points.

    Eq:
        ke = 0.5 (u² + v²)

    Args:
        u (Array): The u-velocity for the input array on the left-right cell faces.
            Size = [Nx+1, Ny]
        v (Array): The v-velocity for the input array on the top-down cell faces.
            Size = [Nx, Ny+1]

    Returns:
        ke (Array): The kinetic energy on the cell center.
            Size = [Nx, Ny]
    """
    # calculate kinetic energy
    ke = 0.5 * (u**2 + v**2)
    return ke


def bernoulli_potential(
    h: Array,
    u: Array,
    v: Array,
    gravity: float = GRAVITY,
) -> Array:
    """Calculates the Bernoulli work.

    This function calculates the Bernoulli work using the equation:
        p = ke + gh
    where:
        ke = 0.5 (u² + v²)

    Args:
        h (Array): The height located at the center/nodes. Size = [Nx, Ny]
        u (Array): The velocity located on the east/north faces. Size = [Nx+1, Ny]
        v (Array): The velocity located on the north/east faces. Size = [Nx, Ny+1]
        gravity (float): The acceleration due to gravity constant. Default = 9.81

    Returns:
        p (Array): The Bernoulli work. Size = [Nx, Ny]

    Example:
        >>> u, v, h = ...
        >>> p = bernoulli_potential(h=h, u=u, v=v)
    """
    # calculate kinetic energy
    ke = kinetic_energy(u=u, v=v)

    # calculate Bernoulli potential
    p = ke + gravity * h

    return p
