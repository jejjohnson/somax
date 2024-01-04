import functools as ft
from typing import Optional, Callable

import einops
from fieldx._src.domain.domain import Domain
from finitevolx import (
    MaskGrid,
    NodeMask,
    center_avg_2D,
    divergence,
    geostrophic_gradient,
    laplacian,
    reconstruct,
)
import jax
import jax.numpy as jnp
from jaxtyping import (
    Array,
    Float,
)

from somax._src.models.qg.domain import LayerDomain
from somax._src.models.qg.elliptical import DSTSolution
from somax._src.models.qg.params import QGParams
from somax._src.operators.dst import (
    inverse_elliptic_dst,
    inverse_elliptic_dst_cmm,
)

laplacian_batch = jax.vmap(laplacian, in_axes=(0, None))


def calculate_potential_vorticity(
    psi: Float[Array, "Nx Ny"],
    domain: Domain,
    layer_domain: LayerDomain,
    params: QGParams,
    masks_psi: Optional[MaskGrid] = None,
    masks_q: Optional[MaskGrid] = None,
) -> Array:
    # calculate laplacian [Nx,Ny] --> [Nx-2, Ny-2]
    psi_lap: Float[Array, "Nx-2 Ny-2"] = laplacian_batch(psi, domain.dx)

    # pad (zero boundaries)
    psi_lap = jnp.pad(
        psi_lap,
        pad_width=((0, 0), (1, 1), (1, 1)),
        mode="constant",
        constant_values=0.0,
    )

    # calculate beta term in helmholtz decomposition
    beta_lap = params.f0**2 * jnp.einsum("lm,...mxy->...lxy", layer_domain.A, psi)

    q = psi_lap - beta_lap

    # apply boundary conditions on psi
    if masks_psi:
        q *= masks_psi.values

    # [Nx,Ny] --> [Nx-1,Ny-1]
    q = jax.vmap(center_avg_2D)(q)

    # calculate beta-plane
    y_coords = center_avg_2D(domain.grid_axis[-1])
    f_y = params.beta * (y_coords - params.y0)

    q += f_y

    # apply boundary conditions on q
    if masks_q:
        q *= masks_q.values

    return q


def advection_rhs(
    q: Float[Array, "Nx-1 Ny-1"],
    psi: Float[Array, "Nx Ny"],
    dx: float,
    dy: float,
    num_pts: int = 1,
    method: str = "wenoz",
    masks_u: MaskGrid = None,
    masks_v: MaskGrid = None,
):
    """Calculates the advection term on the RHS of the Multilayer QG
    PDE. It assumes we use an arakawa C-grid whereby the potential
    vorticity term is on the cell centers, the zonal velocity is on the
    east-west cell faces, and the meridional velocity is on the
    north-south cell faces.

    Velocity:
        u, v = -∂yΨ, ∂xΨ
    Advection:
        u̅⋅∇ q = ∂x(uq) + ∂y(vq)


    This uses the conservative method which calculates the flux terms
    independently, (uq, vq) and then we calculate the partial
    derivatives.

    Args:
        q (Array): the potential vorticity term on the cell centers
            Size = [Nx-1, Ny-1]
        psi (Array): the stream function on the cell vertices
            Size = [Nx, Ny]
    Returns:
        div_flux (Array): the flux divergence on the cell centers
            Size = [Nx-1, Ny-1]

    """

    # calculate velocities
    # u, v = -∂yΨ, ∂xΨ
    u, v = geostrophic_gradient(u=psi, dx=dx, dy=dy)

    # calculate fluxes
    # Note: take interior points of velocities (+ masks)
    q_flux_on_u: Float[Array, "Nx-2 Ny-1"] = reconstruct(
        q=q,
        u=u[1:-1,:],
        dim=0,
        u_mask=masks_u[1:-1, :] if masks_u is not None else None,
        method=method,
        num_pts=num_pts,
    )
    q_flux_on_v: Float[Array, "Nx-1 Ny-2"] = reconstruct(
        q=q,
        u=v[:,1:-1],
        dim=1,
        u_mask=masks_v[:, 1:-1] if masks_v is not None else None,
        method=method,
        num_pts=num_pts,
    )

    # pad arrays to comply with velocities (cell faces)
    q_flux_on_u: Float[Array, "Nx Ny-1"] = jnp.pad(q_flux_on_u, pad_width=((1, 1), (0, 0)))
    q_flux_on_v: Float[Array, "Nx-1 Ny"] = jnp.pad(q_flux_on_v, pad_width=((0, 0), (1, 1)))

    # calculate divergence
    # ∂x(flux_u) + ∂y(flux_v) = div(flux_u, flux_v)
    div_flux: Float[Array, "Nx-1 Ny-1"] = divergence(q_flux_on_u, q_flux_on_v, dx, dy)

    return - div_flux


def batch_advection_rhs(q, psi, dx, dy, num_pts, method, masks_u, masks_v):    
    fn = jax.vmap(advection_rhs, in_axes=(0, 0, None, None, None, None, None, None))
    return fn(q, psi, dx, dy, num_pts, method, masks_u, masks_v)


def viscous_dissip(
    dq: Float[Array, "Nz Nx-1 Ny-1"],
    q: Float[Array, "Nz Nx-1 Ny-1"],
    domain: Domain,
    params: QGParams,
    masks: MaskGrid,
    capacitance_matrix,
) -> Float[Array, "Nz Nx Ny"]:

    y_coords = center_avg_2D(domain.grid_axis[-1])
    f_y = params.beta * (y_coords - params.y0)
    
    # harmonic dissipation (free slip; q=0 at the domain boundary)
    if params.a_2 != 0.:
        if capacitance_matrix == None:
            q_pad = jnp.pad(
                -(q - f_y),    # remove beta-plane vorticity
                pad_width=((0, 0), (1, 1), (1, 1)),
                mode="symmetric",
            )
            q_pad = q_pad.at[...,1:-1,1:-1].set(q - f_y)
            q_har = laplacian_batch(q_pad, domain.dx)
            dq += params.a_2 * q_har
        else:
            raise NotImplementedError("Dissipation is not implemented for non-rectangular domains.")
    
    # biharmonic dissipation
    if params.a_4 != 0.:
        raise NotImplementedError("Biharmonic dissipation is not implemented.")
        # q_pad = jnp.pad(
        #     q,
        #     pad_width=((0, 0), (2, 2), (2, 2)),
        #     mode="constant",
        #     constant_values=0.0,
        # )
        # q_har = laplacian_batch(q_pad, domain.dx)
        # q_bihar = laplacian_batch(q_har, domain.dx)
        # dq -= params.a_4 * q_bihar

    dq *= masks.center.values
    
    return dq


def equation_of_motion(
    q: Array,
    psi: Array,
    params: QGParams,
    domain: Domain,
    layer_domain: LayerDomain,
    forcing_fn: Callable,
    masks=None,
    capacitance_matrix=None,
) -> Array:
    
    # calculate advection
    dq = batch_advection_rhs(
        q, psi,
        domain.dx[-2], domain.dx[-1], 
        params.num_pts, 
        params.method,
        masks.face_u,
        masks.face_v
    )

    # add forces
    dq = forcing_fn(
        psi=psi, dq=dq, 
        domain=domain, 
        layer_domain=layer_domain,
        params=params, 
        masks=masks
    )

    # add dissipation (harmonic + biharmonic)
    dq = viscous_dissip(
        dq=dq, q=q, 
        domain=domain,
        params=params,
        masks=masks,
        capacitance_matrix=capacitance_matrix
    )

    # multiply by mask
    dq *= masks.center.values

    return dq


def calculate_psi_from_pv(
    q: Float[Array, "Nx-1 Ny-1"],
    layer_domain: LayerDomain,
    mask_node: NodeMask,
    dst_sol: DSTSolution,
) -> Float[Array, "Nx Ny"]:
    
    # get interior points (cell verticies interior)
    q_i: Float[Array, "Nx-2 Ny-2"] = jax.vmap(center_avg_2D)(q)
    
    # calculate helmholtz rhs
    helmholtz_rhs: Float[Array, "Nz Nx Ny"] = jnp.einsum(
        "lm, ...mxy -> ...lxy", layer_domain.A_layer_2_mode, q_i
    )

    # solve elliptical inversion problem
    if dst_sol.capacitance_matrix is not None:
        psi_modes: Float[Array, "Nz Nx Ny"] = inverse_elliptic_dst_cmm(
            rhs=helmholtz_rhs,
            H_matrix=dst_sol.H_mat,
            cap_matrices=dst_sol.capacitance_matrix,
            bounds_xids=mask_node.irrbound_xids,
            bounds_yids=mask_node.irrbound_yids,
            mask=mask_node.values,
        )
    else:
        psi_modes: Float[Array, "Nz Nx Ny"] = jax.vmap(
            inverse_elliptic_dst, in_axes=(0, 0)
        )(
            helmholtz_rhs, dst_sol.H_mat
        )

    # Add homogeneous solutions to ensure mass conservation
    psi_modes_i: Float[Array, "Nz Nx-1 Ny-1"] = jax.vmap(center_avg_2D)(psi_modes)

    psi_modes_i_mean: Float[Array, "Nz 1 1"] = einops.reduce(
        psi_modes_i, "... Nx Ny -> ... 1 1", reduction="mean"
    )
    
    alpha: Float[Array, "Nz 1 1"] = -psi_modes_i_mean / dst_sol.homsol_mean
    
    psi_modes += alpha * dst_sol.homsol
    
    psi: Float[Array, "Nz Nx Ny"] = jnp.einsum(
        "lm , ...mxy -> lxy", layer_domain.A_mode_2_layer, psi_modes
    )
    
    psi *= mask_node.values 
    
    return psi