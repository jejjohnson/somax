from typing import Optional, Callable
from jaxtyping import Float, Array
from finitevolx import laplacian, x_avg_2D, y_avg_2D, center_avg_2D
import jax
import jax.numpy as jnp
from somax._src.utils.constants import GRAVITY


def potential_vorticity(
        psi: Float[Array, "Nx Ny"],
        step_size: float | tuple[float, ...] | Array = 1,
        alpha: float=1.0,
        beta: float=0.0,
        f: Optional[float | Array]= None,
        pad_bc_fn: Optional[Callable]=None,
) -> Float[Array, "Nx-1 Ny-1"]:
    """Calculates the potential vorticity according to the
    stream function and forces

    Eq:
        q = α∇²ψ + βψ + f

    Args:
        psi (Array): the stream function on the cell nodes/center
        step_size (float | tuple): the step size for the laplacian operator

    """
    # calculate laplacian
    q: Float[Array, "Nx-2 Ny-2"] = alpha * laplacian(psi, step_size=step_size)

    # pad with zeros
    if pad_bc_fn is not None:
        q: Float[Array, "Nx Ny"] = pad_bc_fn(q)
    else:
        q: Float[Array, "Nx Ny"] = jnp.pad(q, pad_width=((1,1),(1,1)), mode="constant", constant_values=0.0)

    # add beta term
    if beta != 0.0:
        q += beta * psi

    # add planetary vorticity
    if f is not None:
        q += f

    # move q from node/center to center/node
    q: Float[Array, "Nx-1 Ny-1"] = center_avg_2D(q)

    return q


def potential_vorticity_multilayer(
        psi: Float[Array, "Nz Nx Ny"],
        A: Float[Array, "Nm Nz"],
        step_size: float | tuple[float, ...] | Array = 1,
        alpha: float = 1.0,
        beta: float = 1.0,
        f: Optional[float | Array] = None,
        pad_bc_fn: Optional[Callable] = None,
) -> Float[Array, "Nz Nx-1 Ny-1"]:
    """Calculates the potential vorticity according to the
    stream function and forces

    Eq:
        qₖ = α∇²ψₖ + β(Aₖψₖ) + fₖ

    Args:
        psi (Array): the stream function on the cell nodes/center
        step_size (float | tuple): the step size for the laplacian operator

    """
    # calculate laplacian
    laplacian_batch = jax.vmap(laplacian, in_axes=(0, None))
    q: Float[Array, "Nz Nx-2 Ny-2"] = alpha * laplacian_batch(psi, step_size=step_size)

    # pad with zeros
    if pad_bc_fn is not None:
        q: Float[Array, "Nz Nx Ny"] = pad_bc_fn(q)
    else:
        pad_width = ((0,0),(1,1),(1,1))
        q: Float[Array, "Nz Nx Ny"] = jnp.pad(q, pad_width=pad_width, mode="constant", constant_values=0.0)

    # add beta term
    q += beta * jnp.einsum("lz,...zxy->...lxy", A, psi)

    # add planetary vorticity
    if f is not None:
        q += f

    # move q from node/center to center/node
    q: Float[Array, "Nl Nx-1 Ny-1"] = center_avg_2D(q)

    return q

def ssh_to_streamfn(ssh: Array, f0: float = 1e-5, g: float = GRAVITY) -> Array:
    """Calculates the ssh to stream function

    Eq:
        η = (g/f₀) Ψ

    Args:
        ssh (Array): the sea surface height [m]
        f0 (Array|float): the coriolis parameter
        g (float): the acceleration due to gravity

    Returns:
        psi (Array): the stream function
    """
    return (g / f0) * ssh


def streamfn_to_ssh(psi: Array, f0: float = 1e-5, g: float = GRAVITY) -> Array:
    """Calculates the stream function to ssh

    Eq:
        Ψ = (f₀/g) η

    Args:
        psi (Array): the stream function
        f0 (Array|float): the coriolis parameter
        g (float): the acceleration due to gravity

    Returns:
        ssh (Array): the sea surface height [m]
    """
    return (f0 / g) * psi
