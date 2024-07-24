from somax._src.operators.difference import perpendicular_gradient_2D
from somax._src.reconstructions.base import reconstruct
from jaxtyping import Array, Float
from somax._src.masks.masks import MaskGrid


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
    q_flux_on_u: Float[Array, "Nx Ny"] = reconstruct(
        q=q,
        u=u[1:-1, :],
        dim=0,
        u_mask=masks.face_u[1:-1, :] if masks.face_u is not None else None,
        method=method,
        num_pts=num_pts,
    )
    
    # calculate divergence
    q_flux_on_v: Float[Array, "Nx Ny"] = reconstruct(
        q=q,
        u=v[:, 1:-1],
        dim=1,
        u_mask=masks.face_v[:, 1:-1] if masks.face_v is not None else None,
        method=method,
        num_pts=num_pts,
    )
    
    # zero flux!
    q_flux_on_u = jnp.pad()
    
    return None