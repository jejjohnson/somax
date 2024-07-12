from typing import Iterable, Union
from jaxtyping import Array
from somax._src.boundaries.base import padding


def free_slip_padding(u, axes: Iterable[Union[str | None]]):
    """This function pads the array with edge values to
    mimic a zero derivative. It mimics a frictionless
    boundary. This is known as free-slip.

    Eq:
        ∂yu(y=0=Ly)=0
        ∂xv(x=0=Lx)=0

    Args:
        u (Array): the input array
        grid (str):
            options = ["center", "node", "face_u", "face_v"

    Example:
        >>> u: Float[Array, "Nx+1 Ny"] = ...
        >>> u_: Float[Array, "Nx+3 Ny+2"] = free_slip_padding(u, axis=("both", "both"))
        >>> u_: Float[Array, "Nx+3 Ny"] = free_slip_padding(u, axis=("both", None))
        >>> u_: Float[Array, "Nx+2 Ny"] = free_slip_padding(u, axis=("left", None))
        >>> u_: Float[Array, "Nx+1 Ny+1"] = free_slip_padding(u, axis=("left", "right"))
    """
    return padding(u=u, axes=axes, mode="edge")


def no_slip_padding(u, axes: Iterable[Union[str | None]]):
    """This function pads the array with edge values to
    mimic a zero boundary. It mimics a vanishing tangential
    velocity at the boundary. This is known as no-slip.

    Eq:
        u(y=0=Ly)=0
        v(x=0=Lx)=0

    Args:
        u (Array): the input array
        grid (str):
            options = ["center", "node", "face_u", "face_v"

    Example:
        >>> u: Float[Array, "Nx+1 Ny"] = ...
        >>> u_: Float[Array, "Nx+3 Ny+2"] = no_slip_padding(u, axis=("both", "both"))
        >>> u_: Float[Array, "Nx+3 Ny"] = no_slip_padding(u, axis=("both", None))
        >>> u_: Float[Array, "Nx+2 Ny"] = no_slip_padding(u, axis=("left", None))
        >>> u_: Float[Array, "Nx+1 Ny+1"] = no_slip_padding(u, axis=("left", "right"))
    """
    return padding(u=u, axes=axes, mode="constant", constant_values=0.0)
