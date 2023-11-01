import functools as ft
from typing import Optional

import einops
from fieldx._src.domain.domain import Domain
from finitevolx import (
    MaskGrid,
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
from somax._src.models.qg.forcing import (
    calculate_bottom_drag,
)
from somax._src.models.qg.params import QGParams
from somax._src.operators.dst import (
    inverse_elliptic_dst,
    inverse_elliptic_dst_cmm,
)

laplacian_batch = jax.vmap(laplacian, in_axes=(0, None))


def calculate_potential_vorticity(
    psi: Array,
    domain: Domain,
    layer_domain: LayerDomain,
    params: QGParams,
    masks_psi: Optional[MaskGrid] = None,
    masks_q: Optional[MaskGrid] = None,
) -> Array:
    # calculate laplacian [Nx,Ny] --> [Nx-2, Ny-2]
    psi_lap = laplacian_batch(psi, domain.dx)

    # pad
    # [Nx-2,Ny-2] --> [Nx,Ny]
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
    # [Nx,Ny] --> [Nx,Ny-1],[Nx-1,Ny]
    # u, v = -∂yΨ, ∂xΨ
    u, v = geostrophic_gradient(u=psi, dx=dx, dy=dy)

    # take interior points of velocities
    # [Nx,Ny-1] --> [Nx-2,Ny-1]
    u_i = u[..., 1:-1, :]
    # [Nx-1,Ny] --> [Nx-1,Ny-2]
    v_i = v[..., 1:-1]

    q_flux_on_u = reconstruct(
        q=q,
        u=u_i,
        dim=0,
        u_mask=masks_u[1:-1, :],
        method=method,
        num_pts=num_pts,
    )
    q_flux_on_v = reconstruct(
        q=q,
        u=v_i,
        dim=1,
        u_mask=masks_v[:, 1:-1],
        method=method,
        num_pts=num_pts,
    )

    # pad arrays to comply with velocities (cell faces)
    # [Nx-2,Ny-1] --> [Nx,Ny-1]
    q_flux_on_u = jnp.pad(q_flux_on_u, pad_width=((1, 1), (0, 0)))
    # [Nx-1,Ny-2] --> [Nx-1,Ny]
    q_flux_on_v = jnp.pad(q_flux_on_v, pad_width=((0, 0), (1, 1)))

    # calculate divergence
    # [Nx,Ny-1] --> [Nx-1,Ny-1]
    div_flux = divergence(q_flux_on_u, q_flux_on_v, dx, dy)

    return -div_flux, u, v, q_flux_on_u, q_flux_on_v


@ft.partial(jax.vmap, axis_name=("q", "psi"))
def batch_advection_rhs(q, psi, dx, dy, num_pts, method, masks_u, masks_v):
    return advection_rhs(
        q=q,
        psi=psi,
        dx=dx,
        dy=dy,
        num_pts=num_pts,
        method=method,
        masks_u=masks_u,
        masks_v=masks_v,
    )


def qg_rhs(
    q: Array,
    psi: Array,
    params: QGParams,
    domain: Domain,
    layer_domain: LayerDomain,
    dst_sol: DSTSolution,
    wind_forcing: Array,
    bottom_drag: Array,
    masks=None,
) -> Array:
    # calculate advection
    fn = jax.vmap(advection_rhs, in_axes=(0, 0, None, None, None, None, None, None))

    dq, u, v, q_flux_on_u, q_flux_on_v = fn(
        q, psi, domain.dx[-2], domain.dx[-1], 3, "wenoz", masks.face_u, masks.face_v
    )

    bottom_drag = calculate_bottom_drag(
        psi=psi,
        domain=domain,
        H_z=layer_domain.heights[-1],
        delta_ek=params.delta_ek,
        f0=params.f0,
        masks_psi=masks.node,
    )

    # add forces duh
    forces = jnp.zeros_like(dq)
    forces = forces.at[0].set(wind_forcing)
    forces = forces.at[-1].set(bottom_drag)
    # print_debug_quantity(forces, "FORCES")
    dq += forces

    # multiply by mask
    dq *= masks.center.values

    # get interior points (cell verticies interior)
    # [Nx-1,Ny-1] --> [Nx-2,Ny-2]
    dq_i = jax.vmap(center_avg_2D)(dq)

    # calculate helmholtz rhs
    # [Nx-2,Ny-2]
    helmholtz_rhs = jnp.einsum(
        "lm, ...mxy -> ...lxy", layer_domain.A_layer_2_mode, dq_i
    )

    # solve elliptical inversion problem
    # [Nx-2,Ny-2] --> [Nx,Ny]
    if dst_sol.capacitance_matrix is not None:
        # print_debug_quantity(dst_sol.capacitance_matrix, "CAPACITANCE MAT")
        dpsi_modes = inverse_elliptic_dst_cmm(
            rhs=helmholtz_rhs,
            H_matrix=dst_sol.H_mat,
            cap_matrices=dst_sol.capacitance_matrix,
            bounds_xids=masks.node.irrbound_xids,
            bounds_yids=masks.node.irrbound_yids,
            mask=masks.node.values,
        )
    else:
        dpsi_modes = jax.vmap(inverse_elliptic_dst, in_axes=(0, 0))(
            helmholtz_rhs, dst_sol.H_mat
        )

    # Add homogeneous solutions to ensure mass conservation
    # [Nx,Ny] --> [Nx-1,Ny-1]
    dpsi_modes_i = jax.vmap(center_avg_2D)(dpsi_modes)

    dpsi_modes_i_mean = einops.reduce(
        dpsi_modes_i, "... Nx Ny -> ... 1 1", reduction="mean"
    )

    # [Nz] / [Nx,Ny] --> [Nx,Ny]
    alpha = -dpsi_modes_i_mean / dst_sol.homsol_mean

    # [Nx,Ny]
    dpsi_modes += alpha * dst_sol.homsol

    # [Nx,Ny]
    dpsi = jnp.einsum("lm , ...mxy -> lxy", layer_domain.A_mode_2_layer, dpsi_modes)

    return dq, dpsi
