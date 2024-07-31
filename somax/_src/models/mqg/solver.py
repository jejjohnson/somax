from typing import Tuple
import equinox as eqx
import jax
from jaxtyping import Array, Float, ArrayLike
import jax.numpy as jnp
import numpy as np
from somax._src.dst.base import create_laplace_dst_dirichlet_op, inverse_elliptic_dstI
import einx
import einops
from somax._src.operators.average import center_avg_2D
from somax._src.masks.masks import MaskGrid
from somax._src.models.mqg.decomp import Mode2LayerTransformer
from somax._src.models.mqg.model import MQGParams


class DSTSolver(eqx.Module):
    beta: Array
    homsol: Float[Array, "Nz Nx-1 Ny-1"] = eqx.field(static=True)
    homsol_mean: Float[Array, "Nz"] = eqx.field(static=True)
    helmholtz_mat: Float[Array, "Nz Nx-1 Ny-1"] = eqx.field(static=True)
    transformer: Mode2LayerTransformer
    
    def __init__(
        self,
        mask_grid: MaskGrid,
        dx: float,
        dy: float,
        transformer: Mode2LayerTransformer,
        ):
        
        self.beta = transformer.lambda_sq
        # create helmholtz dst
        Nx, Ny = np.array(mask_grid.node.shape) - 1
        # print("node:", mask_grid.node.shape)
        H_mat = create_helmholtz_dst(Nx, Ny, dx, dy, self.beta)
        # print("H:", H_mat.shape)
        
        
        # get homogeneous solution
        homsol = compute_homogeneous_solution(Nx, Ny, self.beta, H_mat)
        
        self.helmholtz_mat = H_mat
        self.homsol = homsol
        
        homsol_center = jax.vmap(center_avg_2D)(homsol)
        # print("homsol_center:", homsol_center.shape)
        # print("q:", mask_grid.center.shape)
        homsol_center *= mask_grid.center.values
        homsol_mean = einx.mean("Nz [Nx Ny]", homsol_center)
        self.homsol_mean = homsol_mean
        self.transformer = transformer
        
    def solve(self, q: Array) -> Array:
        # transform 2 mode space
        src_modes = self.transformer.transform(q)
        # print_debug_quantity(src_modes, "DQ_MODES")
        
        # solve the Poisson problem
        # psi_modes = jax.vmap(inverse_elliptic_dstI, in_axes=(0, 0))(src_modes, self.helmholtz_mat)
        # print_debug_quantity(self.helmholtz_mat, "H_MAT")
        psi_modes = jax.vmap(inverse_elliptic_dstI, in_axes=(0,0))(src_modes, self.helmholtz_mat)
        psi_modes = jnp.pad(psi_modes, pad_width=((0, 0), (1, 1), (1, 1)), mode="constant", constant_values=0.0)
        
        # print_debug_quantity(psi_modes, "DPSI_MODES")
        
        # add homogeneous solution
        psi_modes_mean = jax.vmap(center_avg_2D)(psi_modes)
        alpha = - einx.mean("Nz [Nx Ny]", psi_modes_mean) / self.homsol_mean
        psi_modes += einx.multiply("Nz, Nz Nx Ny -> Nz Nx Ny", alpha, self.homsol)
        
        # print_debug_quantity(psi_modes, "DPSI_MODES (AFTER)")
        
        # transform 2 layer space
        psi = self.transformer.inverse_transform(psi_modes)
        # print_debug_quantity(psi, "DPSI")
        return psi
    
def print_debug_quantity(quantity, name=""):
    size = quantity.shape
    min_ = jnp.min(quantity)
    max_ = jnp.max(quantity)
    mean_ = jnp.mean(quantity)
    median_ = jnp.mean(quantity)
    jax.debug.print(
        f"{name}: {size} | Min: {min_:.6e} | Mean: {mean_:.6e} | Median: {median_:.6e} | Max: {max_:.6e}"
    )
    
    
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
    # # get Laplacian dst transform
    # L_mat = create_laplace_dst_dirichlet_op(
    #     Nx, Ny, dx, dy
    # )
    import math
    x, y = jnp.meshgrid(
        jnp.arange(1, Ny), jnp.arange(1, Nx), indexing='ij'
        )
    L_mat = 2*(jnp.cos(math.pi/Nx*x) - 1)/dx**2 + 2*(jnp.cos(math.pi/Ny*y) - 1)/dy**2
    # print_debug_quantity(L_mat, "L_MAT")
    # # calculate wave numbers
    # x, y = jnp.arange(Nx), np.arange(Ny)
    # kx, ky = jnp.meshgrid(x, y, indexing="ij")
    # lam_kx = - 2.0 * jnp.sin( (np.pi / 2.0) * (kx + 1) / (Nx + 1) ) / dx
    # lam_ky = - 2.0 * jnp.sin( (np.pi / 2.0) * (ky + 1) / (Ny + 1) ) / dy

    # # calculate Laplacian matrix
    # L_mat = - (lam_kx ** 2 + lam_ky ** 2)

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
    sol = jax.vmap(inverse_elliptic_dstI, in_axes=(0,0))(constant_field[..., 1:-1, 1:-1], H_mat)
    # sol = jax.vmap(inverse_elliptic_dstI, in_axes=(0, 0))(
    #     constant_field[..., 1:-1, 1:-1], H_mat
    # )
    
    sol = jnp.pad(sol, pad_width=((0,0),(1,1),(1,1)), mode="constant", constant_values=0.0)
    # print("SOL: ", sol.min(), sol.mean(), sol.max())

    # calculate the homogeneous solution
    homsol = constant_field + einx.dot("Nz Nx Ny, Nz -> Nz Nx Ny", sol, beta)
    # print("homsol: ", homsol.min(), homsol.mean(), homsol.max())

    return homsol

def print_debug_quantity(quantity, name=""):
    size = quantity.shape
    min_ = jnp.min(quantity)
    max_ = jnp.max(quantity)
    mean_ = jnp.mean(quantity)
    median_ = jnp.mean(quantity)
    jax.debug.print(
        f"{name}: {size} | Min: {min_:.6e} | Mean: {mean_:.6e} | Median: {median_:.6e} | Max: {max_:.6e}"
    )