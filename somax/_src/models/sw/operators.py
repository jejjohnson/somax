from typing import Optional

from fieldx._src.domain.domain import Domain
from finitevolx import (
    CenterMask,
    FaceMask,
    NodeMask,
    center_avg_2D,
    difference,
    reconstruct,
    relative_vorticity,
    x_avg_2D,
    y_avg_2D,
)

import jax.numpy as jnp
from jaxtyping import (
    Array,
    Float,
)

from somax._src.models.sw.params import SWMParams
from somax._src.boundaries import zero_gradient_boundaries, no_slip_boundaries, no_flux_boundaries


def calculate_uvh_flux(
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


def kinetic_energy(
    u: Float[Array, "Nx+1 Ny"],
    v: Float[Array, "Nx Ny+1"],
    center_mask: Optional[CenterMask] = None,
):
    """
    Eq:
        ke = 0.5 (u² + v²)
    """
    # calculate squared components
    u2_on_h: Float[Array, "Nx Ny"] = x_avg_2D(u**2)
    v2_on_h: Float[Array, "Nx Ny"] = y_avg_2D(v**2)

    # calculate kinetic energy
    ke_on_h: Float[Array, "Nx Ny"] = 0.5 * (u2_on_h + v2_on_h)

    # apply mask
    if center_mask is not None:
        ke_on_h *= center_mask.values

    return ke_on_h


def potential_vorticity(
    h: Float[Array, "Nx Ny"],
    u: Float[Array, "Nx+1 Ny"],
    v: Float[Array, "Nx Ny+1"],
    dx: Array,
    dy: Array,
    params: SWMParams,
    q_domain: Domain,
    node_mask: Optional[NodeMask] = None,
):
    """
    Eq:
        ζ = ∂v/∂x - ∂u/∂y
        q = (ζ + f) / h
    """
    # pad arrays
    # no gradient boundaries
    h_pad: Float[Array, "Nx+2 Ny+2"] = zero_gradient_boundaries(h, ((1, 1), (1, 1)))

    # no-slip boundaries for u,v
    u_pad: Float[Array, "Nx+1 Ny+2"] = no_slip_boundaries(u, ((0,0),(1,1)))
    v_pad: Float[Array, "Nx+2 Ny+1"] = no_slip_boundaries(v, ((1,1),(0,0)))


    # planetary vorticity, f
    f_on_q: Float[Array, "Nx+1 Ny+1"] = (
        params.coriolis_f0 + q_domain.grid_axis[1] * params.coriolis_beta
    )

    # relative vorticity, ζ = dv/dx - du/dy
    vort_r: Float[Array, "Nx+1 Ny+1"] = relative_vorticity(
        u=u_pad, v=v_pad, dx=dx, dy=dy
    )

    # potential vorticity, q = (ζ + f) / h
    h_on_q: Float[Array, "Nx+1 Ny+1"] = center_avg_2D(h_pad)
    q: Float[Array, "Nx+1 Ny+1"] = (vort_r + f_on_q) / h_on_q

    # apply masks
    if node_mask is not None:
        q *= node_mask.values

    return q


def h_linear_rhs(
    u: Float[Array, "Nx+1 Ny"],
    v: Float[Array, "Nx Ny+1"],
    dx,
    dy,
    params: SWMParams,
    center_mask: Optional[CenterMask] = None,
):
    """
    Eq:
       ∂h/∂t = - H (∂u/∂x + ∂v/∂y)
    """

    # calculate RHS terms
    du_dx: Float[Array, "Nx Ny"] = difference(u, step_size=dx, axis=0, derivative=1)
    dv_dy: Float[Array, "Nx Ny"] = difference(v, step_size=dy, axis=1, derivative=1)

    # calculate RHS
    h_rhs: Float[Array, "Nx Ny"] = -params.depth * (du_dx + dv_dy)

    # apply masks
    if center_mask is not None:
        h_rhs *= center_mask.values

    return h_rhs


def h_nonlinear_rhs(
    uh_flux: Float[Array, "Nx+1 Ny"],
    vh_flux: Float[Array, "Nx Ny1"],
    dx: float | Array,
    dy: float | Array,
    center_mask: Optional[CenterMask] = None,
):
    """
    Eq:
        ∂h/∂t + ∂/∂x((H+h)u) + ∂/∂y((H+h)v) = 0
    """

    # calculate RHS terms
    dhu_dx: Float[Array, "Nx Ny"] = difference(
        uh_flux, step_size=dx, axis=0, derivative=1
    )
    dhv_dy: Float[Array, "Nx Ny"] = difference(
        vh_flux, step_size=dy, axis=1, derivative=1
    )

    # calculate RHS
    h_rhs: Float[Array, "Nx Ny"] = -(dhu_dx + dhv_dy)

    # apply masks
    if center_mask is not None:
        h_rhs *= center_mask.values

    return h_rhs


def u_linear_rhs(
    h: Float[Array, "Nx Ny"],
    v: Float[Array, "Nx Ny+1"],
    dx: float | Array,
    params: SWMParams,
    u_mask: Optional[FaceMask] = None,
):
    """
    Eq:
        ∂u/∂t = fv - g ∂h/∂x
    """
    # pad arrays
    h_pad: Float[Array, "Nx+2 Ny"] = zero_gradient_boundaries(h, ((1, 1), (0, 0)))
    v_pad: Float[Array, "Nx+2 Ny+1"] = no_slip_boundaries(v, ((1,1),(0,0)))

    # calculate RHS terms
    v_avg: Float[Array, "Nx+1 Ny"] = center_avg_2D(v_pad)
    dh_dx: Float[Array, "Nx+1 Ny"] = difference(
        h_pad, step_size=dx, axis=0, derivative=1
    )

    # calculate RHS
    u_rhs: Float[Array, "Nx+1 Ny"] = params.coriolis_f0 * v_avg - params.gravity * dh_dx

    # apply masks
    if u_mask is not None:
        u_rhs *= u_mask.values

    return u_rhs


def u_nonlinear_rhs(
    h: Float[Array, "Nx Ny"],
    q: Float[Array, "Nx+1 Ny+1"],
    vh_flux: Float[Array, "Nx Ny+1"],
    ke: Float[Array, "Nx Ny"],
    params: SWMParams,
    dx: float | Array,
    num_pts: int = 3,
    method: str = "wenoz",
    u_mask: Optional[FaceMask] = None,
):
    """
    Eq:
        work = g ∂h/∂x
        ke = 0.5 (u² + v²)
        ∂u/∂t = qhv - work - ke

    Notes:
        - uses reconstruction (5pt, improved weno) of q on vh flux
    """

    # pad arrays
    h_pad: Float[Array, "Nx+2 Ny"] = zero_gradient_boundaries(h, ((1, 1), (0, 0)))
    ke_pad: Float[Array, "Nx+2 Ny"] = no_flux_boundaries(ke, ((1, 1), (0, 0)))

    vh_flux_on_u: Float[Array, "Nx-1 Ny"] = center_avg_2D(vh_flux)

    # no flux padding
    vh_flux_on_u: Float[Array, "Nx+2 Ny"] = no_flux_boundaries(vh_flux_on_u, ((1, 1), (0, 0)))


    qhv_flux_on_u: Float[Array, "Nx+1 Ny"] = reconstruct(
        q=q,
        u=vh_flux_on_u,
        u_mask=u_mask if u_mask is not None else None,
        dim=1,
        method=method,
        num_pts=num_pts,
    )

    # apply mask
    if u_mask is not None:
        qhv_flux_on_u *= u_mask.values

    # calculate work
    dh_dx: Float[Array, "Nx+1 Ny"] = difference(
        h_pad, step_size=dx, axis=0, derivative=1
    )
    work = params.gravity * dh_dx

    # calculate kinetic energy
    dke_on_u: Float[Array, "Nx+1 Ny"] = difference(
        ke_pad, step_size=dx, axis=0, derivative=1
    )

    # calculate u RHS
    u_rhs: Float[Array, "Nx+1 Ny"] = (
        - work + qhv_flux_on_u - dke_on_u
    )

    # apply mask
    if u_mask is not None:
        u_rhs *= u_mask.values

    return u_rhs


def v_linear_rhs(
    h: Float[Array, "Nx Ny"],
    u: Float[Array, "Nx+1 Ny"],
    dy: float | Array,
    params: SWMParams,
    v_mask: Optional[FaceMask] = None,
):
    """
    Eq:
        ∂v/∂t = - fu - g ∂h/∂y
    """
    # pad arrays
    h_pad: Float[Array, "Nx Ny+2"] = zero_gradient_boundaries(h, ((0, 0), (1, 1)))
    u_pad: Float[Array, "Nx+1 Ny+2"] = no_slip_boundaries(u, ((0,0),(1,1)))

    # calculate RHS terms
    u_avg: Float[Array, "Nx Ny+1"] = center_avg_2D(u_pad)
    dh_dy: Float[Array, "Nx Ny+1"] = difference(
        h_pad, step_size=dy, axis=1, derivative=1
    )

    # calculate RHS
    v_rhs: Float[Array, "Nx Ny+1"] = (
        -params.coriolis_f0 * u_avg - params.gravity * dh_dy
    )

    # apply masks
    if v_mask is not None:
        v_rhs *= v_mask.values

    return v_rhs



def v_nonlinear_rhs(
    h: Float[Array, "Nx Ny"],
    q: Float[Array, "Nx+1 Ny+1"],
    uh_flux: Float[Array, "Nx+1 Ny"],
    ke: Float[Array, "Nx Ny"],
    dy: float | Array,
    params: SWMParams,
    num_pts: int = 3,
    method: str = "wenoz",
    v_mask: Optional[FaceMask] = None,
):
    """
    Eq:
        work = g ∂h/∂y
        ke = 0.5 (u² + v²)
        ∂v/∂t = - qhu - work - ke

    Notes:
        - uses reconstruction (5pt, improved weno) of q on uh flux
    """
    h_pad: Float[Array, "Nx Ny+2"] = zero_gradient_boundaries(h, ((0, 0), (1, 1)))
    ke_pad: Float[Array, "Nx Ny+2"] = no_flux_boundaries(ke, ((0, 0), (1, 1)))

    uh_flux_on_v: Float[Array, "Nx Ny-1"] = center_avg_2D(uh_flux)

    # assume no flux on boundaries
    uh_flux_on_v: Float[Array, "Nx Ny+2"] = no_flux_boundaries(uh_flux_on_v, ((0, 0), (1, 1)))

    qhu_flux_on_v: Float[Array, "Nx Ny+1"] = reconstruct(
        q=q,
        u=uh_flux_on_v,
        u_mask=v_mask if v_mask is not None else None,
        dim=0,
        method=method,
        num_pts=num_pts,
    )

    # apply masks
    if v_mask is not None:
        qhu_flux_on_v *= v_mask.values

    # calculate work
    dh_dy: Float[Array, "Nx Ny+1"] = difference(
        h_pad, step_size=dy, axis=1, derivative=1
    )
    work = params.gravity * dh_dy

    # calculate kinetic energy
    dke_on_v: Float[Array, "Nx Ny+1"] = difference(
        ke_pad, step_size=dy, axis=1, derivative=1
    )

    # calculate u RHS
    v_rhs: Float[Array, "Nx Ny+1"] = (
        - work - qhu_flux_on_v - dke_on_v
    )

    # apply masks
    if v_mask is not None:
        v_rhs *= v_mask.values

    return v_rhs
