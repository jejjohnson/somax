from typing import Optional
from jaxtyping import Float, Array
from somax.masks import FaceMask
from somax.operators import reconstruct



def uv_center_flux(

    
    h: Float[Array, "Nx Ny"],
    u: Float[Array, "Nx-1 Ny"],
    v: Float[Array, "Nx Ny-1"],
    u_mask: Optional[FaceMask] = None,
    v_mask: Optional[FaceMask] = None,
    num_pts: int = 3,
    method: str = "wenoz",
):
    """
    Eq:
        (uh), (vh)
    Args:
        h (Array): the input array on the cell centers
            Size = [Nx, Ny]
        u (Array):
    """


    # calculate h fluxes
    uh_flux: Float[Array, "Nx-1 Ny"] = reconstruct(
        q=h[:, 1:-1], u=u, u_mask=u_mask, dim=0, num_pts=num_pts, method=method
    )
    vh_flux: Float[Array, "Nx Ny-1"] = reconstruct(
        q=h[1:-1, :], u=v, u_mask=v_mask, dim=1, num_pts=num_pts, method=method
    )

    # apply mask
    if u_mask is not None:
        uh_flux *= u_mask.values
    if v_mask is not None:
        vh_flux *= v_mask.values

    return uh_flux, vh_flux


def uv_node_flux(
    q: Float[Array, "Nx Ny"],
    u: Float[Array, "Nx-1 Ny"],
    v: Float[Array, "Nx Ny-1"],
    u_mask: Optional[FaceMask] = None,
    v_mask: Optional[FaceMask] = None,
    num_pts: int = 3,
    method: str = "wenoz",
):
    """
    Eq:
        ∇⋅[uq, vq] = uq, vq
    """

    # calculate h fluxes
    uh_flux: Float[Array, "Nx-1 Ny"] = reconstruct(
        q=q[:, 1:-1], u=u, u_mask=u_mask, dim=1, num_pts=num_pts, method=method
    )
    vh_flux: Float[Array, "Nx Ny-1"] = reconstruct(
        q=q[1:-1, :], u=v, u_mask=v_mask, dim=0, num_pts=num_pts, method=method
    )

    # apply mask
    if u_mask is not None:
        uh_flux *= u_mask.values
    if v_mask is not None:
        vh_flux *= v_mask.values

    return uh_flux, vh_flux
