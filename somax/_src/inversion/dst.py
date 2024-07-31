from somax._src.operators.difference import laplacian_2D
from jaxtyping import Array
import jax.numpy as jnp
import jax


def compute_laplace_dstII(nx, ny, dx, dy) -> Array:
    """
    Compute the Laplace Discrete Sine Transform II (DST-II) of a 2D grid.
    
    Eq:
        H := ∇²

    Args:
        nx (int): Number of grid points in the x-direction.
        ny (int): Number of grid points in the y-direction.
        dx (float): Grid spacing in the x-direction.
        dy (float): Grid spacing in the y-direction.

    Returns:
        Array: The Laplace DST-II of the input grid.

    """
    x, y = jnp.meshgrid(jnp.arange(nx), jnp.arange(ny), indexing="ij")
    return (
        2 * (jnp.cos(jnp.pi / nx * x) - 1) / dx**2
        + 2 * (jnp.cos(jnp.pi / ny * y) - 1) / dy**2
    )
    
    
def compute_laplace_dstI(nx, ny, dx, dy) -> Array:
    """
    Compute the Laplace operator in the discrete sine transform domain.
    
    Eq:
        H := ∇²

    Args:
        nx (int): Number of grid points in the x-direction.
        ny (int): Number of grid points in the y-direction.
        dx (float): Grid spacing in the x-direction.
        dy (float): Grid spacing in the y-direction.

    Returns:
        Array: The Laplace operator in the discrete sine transform domain.

    """
    x, y = jnp.meshgrid(jnp.arange(1, nx), jnp.arange(1, ny), indexing="ij")
    return (
        2 * (jnp.cos(jnp.pi / (nx) * x) - 1) / dx**2
        + 2 * (jnp.cos(jnp.pi / (ny) * y) - 1) / dy**2
    )
    
    
def compute_helmholtz_distI(nx, ny, dx, dy, alpha=1.0, beta=0.0) -> Array:
    """
    Compute the Helmholtz distance for a given grid.
    
    Eq:
        H := (α∇² - β)

    Parameters:
    - nx (int): Number of grid points in the x-direction.
    - ny (int): Number of grid points in the y-direction.
    - dx (float): Grid spacing in the x-direction.
    - dy (float): Grid spacing in the y-direction.
    - alpha (float, optional): Scaling factor for the Laplace distance. Default is 1.0.
    - beta (float, optional): Constant offset. Default is 0.0.

    Returns:
    - Array: The computed Helmholtz distance.

    """
    return alpha * compute_laplace_dstI(nx=nx, ny=ny, dx=dx, dy=dy) - alpha * beta


def compute_helmholtz_distII(nx, ny, dx, dy, alpha=1.0, beta=0.0) -> Array:
    """
    Compute the Helmholtz distance for a given grid.
    
    Eq:
        H := (α∇² - β)

    Parameters:
    - nx (int): Number of grid points in the x-direction.
    - ny (int): Number of grid points in the y-direction.
    - dx (float): Grid spacing in the x-direction.
    - dy (float): Grid spacing in the y-direction.
    - alpha (float, optional): Scaling factor for the Laplace distance. Default is 1.0.
    - beta (float, optional): Constant offset. Default is 0.0.

    Returns:
    - Array: The computed Helmholtz distance.

    """
    return alpha * compute_laplace_dstII(nx=nx, ny=ny, dx=dx, dy=dy) - alpha * beta


def helmholtz_dirichlet_bc(u, dx, dy, alpha=1.0, beta=0.0):
    """
    Applies the Helmholtz Dirichlet boundary condition to the given function.

    Parameters:
    u (ndarray): The function to which the boundary condition is applied.
    dx (float): Step size in the x-direction.
    dy (float): Step size in the y-direction.
    beta (float): Coefficient for the boundary condition.

    Returns:
    ndarray: The result of applying the Helmholtz Dirichlet boundary condition.
    """
    lap_u = laplacian_2D(u, step_size_x=dx, step_size_y=dy)
    return alpha * lap_u - alpha * beta * u[1:-1, 1:-1]


def dstI1D(x, norm="ortho"):
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
    N = x.shape
    padding = ((0, 0),) * (num_dims - 1) + ((1, 1),)
    x = jnp.pad(x, pad_width=padding, mode="constant", constant_values=0.0)
    x = jnp.fft.irfft(-1j * x, axis=-1, norm=norm)
    x = jax.lax.slice_in_dim(x, 1, N[-1] + 1, axis=-1)
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
    x = dstI1D(x, norm=norm)
    x = jnp.transpose(x, axes=(-1, -2))
    x = dstI1D(x, norm=norm)
    x = jnp.transpose(x, axes=(-1, -2))
    return x


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
    # print_debug_quantity(x)
    x /= operator_dst
    # print_debug_quantity(x)
    return dstI2D(x)
