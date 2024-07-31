import jax
import jax.numpy as jnp
from somax._src.operators.differential import perpendicular_gradient_2D, laplacian_2D, divergence_2D
from somax._src.operators.average import center_avg_2D
from somax._src.reconstructions.base import reconstruct
from jaxtyping import Array, Float, ArrayLike
from somax._src.masks.masks import MaskGrid
from somax._src.models.mqg.decomp import Mode2LayerTransformer
from functools import partial
import einx


def compute_q_from_psi(
    psi: Float[Array, "Nx+1 Ny+1"],
    masks: MaskGrid,
    dx: ArrayLike,
    dy: ArrayLike,
    Y: ArrayLike,
    y0: ArrayLike,
    beta: ArrayLike,
    transformer: Mode2LayerTransformer,
) -> Float[Array, "Nx Ny"]:
    
    # calculate q
    fn = partial(laplacian_2D, step_size_x=dx, step_size_y=dy)
    psi_lap = jax.vmap(fn)(psi)
    
    # add zero boundaries
    psi_lap = jnp.pad(psi_lap, ((0, 0), (1, 1), (1, 1)), mode="constant", constant_values=0.0)
    
    psi_beta = transformer.kappa * transformer.inverse_transform(psi)
    
    q_on_n = psi_lap - psi_beta
    
    # apply mask
    q_on_n *= masks.node.values
    
    # add beta term
    q_on_n = einx.add("Nz Nx Ny, Nx Ny -> Nz Nx Ny", q_on_n, beta * (Y - y0))
    
    # interpolate onto center values
    q = jax.vmap(center_avg_2D)(q_on_n)
    
    # apply masks
    q *= masks.center.values
    
    return q


def calculate_elliptical_rhs(
    psi: Float[Array, "Nx+1 Ny+1"],
    q: Float[Array, "Nx Ny"],
    dx: float = 1,
    dy: float = 1,
    masks: MaskGrid | None = None,
    num_pts: int = 5,
    method: str = "wenoz",
):
    """
    Velocity:
        u, v = -∂yΨ, ∂xΨ
    Advection:
        u̅⋅∇ q = ∂x(uq) + ∂y(vq)
    """
    # calculate velocities, uv    
    grads = perpendicular_gradient_2D(psi, step_size_x=dx, step_size_y=dy)
    
    u: Float[Array, "Nx Ny+1"] = grads[0]
    v: Float[Array, "Nx+1 Ny"] = grads[1]
    
    # calculate divergence
    uq_flux: Float[Array, "Nx Ny"] = reconstruct(
        q=q,
        u=u[1:-1, :],
        dim=0,
        u_mask=masks.face_u[1:-1, :] if masks.face_u is not None else None,
        method=method,
        num_pts=num_pts,
    )
    
    # calculate divergence
    vq_flux: Float[Array, "Nx Ny"] = reconstruct(
        q=q,
        u=v[:, 1:-1],
        dim=1,
        u_mask=masks.face_v[:, 1:-1] if masks.face_v is not None else None,
        method=method,
        num_pts=num_pts,
    )
    
    # zero flux!
    uq_flux = jnp.pad(uq_flux, pad_width=((1,1),(0,0)), mode="constant", constant_values=0.0)
    vq_flux = jnp.pad(vq_flux, pad_width=((0,0),(1,1)), mode="constant", constant_values=0.0)
    
    # calculate divergence
    div_flux = divergence_2D(u=uq_flux, v=vq_flux, step_size_x=dx, step_size_y=dy)
    
    return div_flux

batch_calculate_elliptical_rhs = jax.vmap(calculate_elliptical_rhs, in_axes=(0,0,None,None,None,None,None))
