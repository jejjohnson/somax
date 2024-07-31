from typing import Tuple
from jaxtyping import ArrayLike
import jax.numpy as jnp
from somax._src.dst.wavenumbers import poisson_dirichlet_wavenumber, poisson_neumann_wavenumber, poisson_periodic_wavenumber
import jax


def create_wavenumber_mat():
    return None


def create_laplace_dst_dirichlet_op(Nx, Ny, dx, dy, ):
    # create wavenumber matrix
    kx, ky = jnp.meshgrid(jnp.arange(Nx), jnp.arange(Ny), indexing="ij")
    
    # calculate Laplacian operator
    Lx = poisson_dirichlet_wavenumber(kx, Nx)
    Ly = poisson_dirichlet_wavenumber(ky, Ny)
    L_mat = - ((Lx/dx)**2 + (Ly/dy)**2)
    
    return L_mat


def create_helmholtz_dst_dirichlet_op(Nx, Ny, dx, dy, alpha: ArrayLike=1.0, beta: ArrayLike=0.0):
    L_mat = create_laplace_dst_dirichlet_op(Nx, Ny, dx, dy)
    
    return alpha * L_mat - alpha * beta


def dstI1D(x, axis=0, norm="ortho"):
    """
    1D type-I discrete sine transform.

    Parameters:
    - x: ndarray
        Input array.
    - norm: str, optional
        Normalization mode. Default is "ortho".

    Returns:
    - x: ndarray
        Transformed array.

    """
    num_dims = x.ndim
    x = jnp.swapaxes(x, axis, -1)
    N = x.shape
    padding = ((0, 0),) * (num_dims - 1) + ((1,1),)
    x = jnp.pad(x, pad_width=padding, mode="constant", constant_values=0.0)
    x = jnp.fft.irfft(-1j * x, axis=-1, norm=norm)
    x = jax.lax.slice_in_dim(x, 1, N[-1] + 1, axis=-1)
    x = jnp.swapaxes(x, -1, axis)
    return x


def dstI2D(x, norm="ortho"):
    """
    Perform a 2D type-I discrete sine transform on the input array.

    Parameters:
    - x: Input array to be transformed.
    - norm: Normalization mode. Default is "ortho".

    Returns:
    - Transformed array after applying the 2D type-I discrete sine transform.
    """
    assert x.ndim >= 2
    x = dstI1D(x, axis=-1, norm=norm)
    x = dstI1D(x, axis=-2, norm=norm)
    return x

# def dstI2D(x, axes: int | Tuple[int, ...], norm="ortho"):
#     """
#     Perform a 2D type-I discrete sine transform on the input array.

#     Parameters:
#     - x: Input array to be transformed.
#     - norm: Normalization mode. Default is "ortho".

#     Returns:
#     - Transformed array after applying the 2D type-I discrete sine transform.
#     """
#     assert x.ndim >= 2
#     if isinstance(axes, int):
#         axes = tuple(int,)
        
#     x = 
#     x = dstI1D(x, axis=-1, norm=norm)
#     x = dstI1D(x, axis=-2, norm=norm)
#     return x



def inverse_elliptic_dstI(f, operator_dst):
    """
    Inverse elliptic operator (e.g. Laplace, Helmoltz) using float32 discrete sine transform.

    Parameters:
    - f: Input array representing the elliptic operator.
    - operator_dst: The operator used in the discrete sine transform.

    Returns:
    - The result of applying the inverse elliptic operator using the discrete sine transform.
    """
    x = dstI2D(f)
    x /= operator_dst
    # print_debug_quantity(x)
    return dstI2D(x)