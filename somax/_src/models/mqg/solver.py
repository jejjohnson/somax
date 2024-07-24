from typing import Tuple
import equinox as eqx
import jax
from jaxtyping import Array, Float, ArrayLike
import jax.numpy as jnp
from somax._src.inversion.dst import inverse_elliptic_dstI, compute_helmholtz_distI
import einx
from somax._src.operators.average import center_avg_2D
from somax._src.masks.masks import MaskGrid


class DSTSolver(eqx.Module):
    homsol: Float[Array, "Nz Nx-1 Ny-1"] = eqx.field(static=True)
    homsol_mean: Float[Array, "Nz"] = eqx.field(static=True)
    helmholtz_mat: Float[Array, "Nz Nx-1 Ny-1"] = eqx.field(static=True)
    
    def __init__(self, mask_grid: MaskGrid, dx: float, dy: float, beta: ArrayLike):
        # create helmholtz dst
        Nx, Ny = mask_grid.node.shape
        H_mat = create_helmholtz_dst(Nx-1, Ny-1, dx, dy, beta)
        # get homogeneous solution
        homsol = compute_homogeneous_solution(Nx-1, Ny-1, beta, H_mat)
        
        self.helmholtz_mat = H_mat
        self.homsol = homsol
        
        homsol_center = jax.vmap(center_avg_2D)(homsol)
        homsol_center *= mask_grid.center.values
        homsol_mean = einx.mean("Nz [Nx Ny]", homsol_center)
        self.homsol_mean = homsol_mean


def create_helmholtz_dst(
    Nx, Ny, dx, dy, beta
) -> Array:
    """
    Create a Helmholtz DST matrix.

    Args:
        Nx (int): Number of grid points in the x-direction.
        Ny (int): Number of grid points in the y-direction.
        dx (float): Grid spacing in the x-direction.
        dy (float): Grid spacing in the y-direction.
        beta (float): Helmholtz parameter.

    Returns:
        Array: Helmholtz DST matrix.

    """
    # get Laplacian dst transform
    # print(domain.Nx, domain.dx)
    L_mat = compute_helmholtz_distI(
        Nx, Ny, dx, dy
    )
    # calculate helmholtz dst (broadcasting)
    H_mat = einx.subtract("Nx Ny, Nz -> Nz Nx Ny", L_mat, beta)

    return H_mat


def compute_homogeneous_solution(Nx: int, Ny: int, beta: Array, H_mat: Array):
    # create constant field
    constant_field = jnp.ones_like(H_mat)
    
    # pad with ones..
    num_dims = H_mat.ndim
    total_pad = ((0,0),) * (num_dims - 2) + ((1,1),(1,1))
    
    constant_field = jnp.pad(constant_field, total_pad, mode="constant", constant_values=1.0)

    # get homogeneous solution
    sol = jax.vmap(inverse_elliptic_dstI, in_axes=(0, 0))(
        constant_field[..., 1:-1, 1:-1], H_mat
    )

    # calculate the homogeneous solution
    sol = einx.dot("Nz Nx Ny, Nz -> Nz Nx Ny", sol, beta)
    homsol = constant_field + sol 

    return homsol