from typing import Optional, Callable

import einops
from somax.domain import Domain
from somax.masks import MaskGrid, NodeMask
from somax.interp import y_avg_2D, center_avg_2D
from somax._src.operators.differential import divergence_2D, perpendicular_gradient_2D, laplacian_2D
from somax._src.reconstructions.base import reconstruct
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

laplacian_batch = jax.vmap(laplacian_2D, in_axes=(0, None, None))


def calculate_potential_vorticity(
    psi: Float[Array, "Nx Ny"],
    domain: Domain,
    layer_domain: LayerDomain,
    params: QGParams,
    masks_psi: Optional[MaskGrid] = None,
    masks_q: Optional[MaskGrid] = None,
) -> Array:
    # calculate laplacian [Nx,Ny] --> [Nx-2, Ny-2]
    psi_lap: Float[Array, "Nx-2 Ny-2"] = laplacian_batch(psi, domain.dx[0], domain.dx[1])

    # pad (zero boundaries)
    psi_lap = jnp.pad(
        psi_lap,
        pad_width=((0, 0), (1, 1), (1, 1)),
        mode="constant",
        constant_values=0.0,
    )

    # calculate beta term in helmholtz decomposition
    beta_lap = params.f0**2 * jnp.einsum(
        "lm,...mxy->...lxy", layer_domain.A, psi
    )

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


def det_jacobian(f, g, dx, dy):
    """Arakawa discretisation of Jacobian J(f,g).
    Scalar fields f and g must have the same dimension.
    Grid is regular and dx = dy."""

    dx_f = f[..., 2:, :] - f[..., :-2, :]
    dx_g = g[..., 2:, :] - g[..., :-2, :]
    dy_f = f[..., 2:] - f[..., :-2]
    dy_g = g[..., 2:] - g[..., :-2]

    return (
        (
            dx_f[..., 1:-1] * dy_g[..., 1:-1, :]
            - dx_g[..., 1:-1] * dy_f[..., 1:-1, :]
        )
        + (
            (
                f[..., 2:, 1:-1] * dy_g[..., 2:, :]
                - f[..., :-2, 1:-1] * dy_g[..., :-2, :]
            )
            - (
                f[..., 1:-1, 2:] * dx_g[..., 2:]
                - f[..., 1:-1, :-2] * dx_g[..., :-2]
            )
        )
        + (
            (
                g[..., 1:-1, 2:] * dx_f[..., 2:]
                - g[..., 1:-1, :-2] * dx_f[..., :-2]
            )
            - (
                g[..., 2:, 1:-1] * dy_f[..., 2:, :]
                - g[..., :-2, 1:-1] * dy_f[..., :-2, :]
            )
        )
    ) / (12.0 * dx * dy)


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

    if method == "arakawa":
        q_pad = jnp.pad(
            -q,
            pad_width=((1, 1), (1, 1)),
            mode="symmetric",
        )
        q_pad = q_pad.at[..., 1:-1, 1:-1].set(q)
        q_pad = q_pad.at[..., 0, 0].set(q[0, 0])
        q_pad = q_pad.at[..., 0, -1].set(q[0, -1])
        q_pad = q_pad.at[..., -1, 0].set(q[-1, 0])
        q_pad = q_pad.at[..., -1, -1].set(q[-1, -1])
        div_flux: Float[Array, "Nx-2 Ny-2"] = det_jacobian(
            psi, center_avg_2D(q_pad), dx=dx, dy=dy
        )
        div_flux: Float[Array, "Nx-1 Ny-1"] = center_avg_2D(
            jnp.pad(div_flux, pad_width=((1, 1), (1, 1)), mode="edge")
        )

    else:
        # calculate velocities
        # u, v = -∂yΨ, ∂xΨ
        u, v = perpendicular_gradient_2D(u=psi, step_size_x=dx, step_size_y=dy)

        # calculate fluxes
        # Note: take interior points of velocities (+ masks)
        q_flux_on_u: Float[Array, "Nx-2 Ny-1"] = reconstruct(
            q=q,
            u=u[1:-1, :],
            dim=0,
            u_mask=masks_u[1:-1, :] if masks_u is not None else None,
            method=method,
            num_pts=num_pts,
        )
        q_flux_on_v: Float[Array, "Nx-1 Ny-2"] = reconstruct(
            q=q,
            u=v[:, 1:-1],
            dim=1,
            u_mask=masks_v[:, 1:-1] if masks_v is not None else None,
            method=method,
            num_pts=num_pts,
        )

        # pad arrays to comply with velocities (cell faces)
        q_flux_on_u: Float[Array, "Nx Ny-1"] = jnp.pad(
            q_flux_on_u, pad_width=((1, 1), (0, 0))
        )
        q_flux_on_v: Float[Array, "Nx-1 Ny"] = jnp.pad(
            q_flux_on_v, pad_width=((0, 0), (1, 1))
        )

        # calculate divergence
        # ∂x(flux_u) + ∂y(flux_v) = div(flux_u, flux_v)
        div_flux: Float[Array, "Nx-1 Ny-1"] = divergence_2D(
            q_flux_on_u, q_flux_on_v, dx, dy
        )

    return -div_flux


def batch_advection_rhs(q, psi, dx, dy, num_pts, method, masks_u, masks_v):
    fn = jax.vmap(
        advection_rhs, in_axes=(0, 0, None, None, None, None, None, None)
    )
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
    if params.a_2 != 0.0:
        if capacitance_matrix == None:
            q_pad = jnp.pad(
                -(q - f_y),  # remove beta-plane vorticity
                pad_width=((0, 0), (1, 1), (1, 1)),
                mode="symmetric",
            )
            q_pad = q_pad.at[..., 1:-1, 1:-1].set(q - f_y)
            q_har = laplacian_batch(q_pad, domain.dx)
            dq += params.a_2 * q_har
        else:
            raise NotImplementedError(
                "Dissipation is not implemented for non-rectangular domains."
            )

    # biharmonic dissipation
    if params.a_4 != 0.0:
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
        q,
        psi,
        domain.dx[-2],
        domain.dx[-1],
        params.num_pts,
        params.method,
        masks.face_u,
        masks.face_v,
    )

    # add forces
    dq = forcing_fn(
        psi=psi,
        dq=dq,
        domain=domain,
        layer_domain=layer_domain,
        params=params,
        masks=masks,
    )

    # add dissipation (harmonic + biharmonic)
    dq = viscous_dissip(
        dq=dq,
        q=q,
        domain=domain,
        params=params,
        masks=masks,
        capacitance_matrix=capacitance_matrix,
    )

    # multiply by mask
    dq *= masks.center.values

    return dq


def calculate_psi_from_pv(
    q: Float[Array, "Nx-1 Ny-1"],
    params: QGParams,
    domain: Domain,
    layer_domain: LayerDomain,
    mask_node: NodeMask,
    dst_sol: DSTSolution,
    remove_beta=True,
) -> Float[Array, "Nx Ny"]:

    # get interior points (cell verticies interior)
    if remove_beta == True:
        y_coords = center_avg_2D(domain.grid_axis[-1])
        f_y = params.beta * (y_coords - params.y0)
        q_i: Float[Array, "Nx-2 Ny-2"] = jax.vmap(center_avg_2D)(q - f_y)
    # elif remove_beta == "v":
    #     _, v = jax.vmap(geostrophic_gradient, in_axes=(0, None, None))(psi, domain.dx[-2], domain.dx[-1])
    #     bv = params.beta * jax.vmap(y_avg_2D)(v)
    #     q_i: Float[Array, "Nx-2 Ny-2"] = jax.vmap(center_avg_2D)(q - bv)
    else:
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
        )(helmholtz_rhs, dst_sol.H_mat)

    # Add homogeneous solutions to ensure mass conservation
    psi_modes_i: Float[Array, "Nz Nx-1 Ny-1"] = jax.vmap(center_avg_2D)(
        psi_modes
    )

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


from typing import Optional, Callable
from jaxtyping import Float, Array
from somax.interp import x_avg_2D, y_avg_2D, center_avg_2D
import jax
import jax.numpy as jnp
from somax._src.utils.constants import GRAVITY


def potential_vorticity(
    psi: Float[Array, "Nx Ny"],
    step_size: float | tuple[float, ...] | Array = 1,
    alpha: float = 1.0,
    beta: float = 0.0,
    f: Optional[float | Array] = None,
    pad_bc_fn: Optional[Callable] = None,
) -> Float[Array, "Nx-1 Ny-1"]:
    """Calculates the potential vorticity according to the
    stream function and forces

    Eq:
        q = α∇²ψ + βψ + f

    Args:
        psi (Array): the stream function on the cell nodes/center
        step_size (float | tuple): the step size for the laplacian operator

    """
    # calculate laplacian
    q: Float[Array, "Nx-2 Ny-2"] = alpha * laplacian_2D(psi, step_size_x=step_size[0], step_size_y=step_size[1])

    # pad with zeros
    if pad_bc_fn is not None:
        q: Float[Array, "Nx Ny"] = pad_bc_fn(q)
    else:
        q: Float[Array, "Nx Ny"] = jnp.pad(
            q, pad_width=((1, 1), (1, 1)), mode="constant", constant_values=0.0
        )

    # add beta term
    if beta != 0.0:
        q += beta * psi

    # add planetary vorticity
    if f is not None:
        q += f

    # move q from node/center to center/node
    q: Float[Array, "Nx-1 Ny-1"] = center_avg_2D(q)

    return q


def potential_vorticity_multilayer(
    psi: Float[Array, "Nz Nx Ny"],
    A: Float[Array, "Nm Nz"],
    step_size: float | tuple[float, ...] | Array = 1,
    alpha: float = 1.0,
    beta: float = 1.0,
    f: Optional[float | Array] = None,
    pad_bc_fn: Optional[Callable] = None,
) -> Float[Array, "Nz Nx-1 Ny-1"]:
    """Calculates the potential vorticity according to the
    stream function and forces

    Eq:
        qₖ = α∇²ψₖ + β(Aₖψₖ) + fₖ

    Args:
        psi (Array): the stream function on the cell nodes/center
        step_size (float | tuple): the step size for the laplacian operator

    """
    # calculate laplacian
    laplacian_batch = jax.vmap(laplacian_2D, in_axes=(0, None, None))
    q: Float[Array, "Nz Nx-2 Ny-2"] = alpha * laplacian_batch(
        psi, step_size[0], step_size[1]
    )

    # pad with zeros
    if pad_bc_fn is not None:
        q: Float[Array, "Nz Nx Ny"] = pad_bc_fn(q)
    else:
        pad_width = ((0, 0), (1, 1), (1, 1))
        q: Float[Array, "Nz Nx Ny"] = jnp.pad(
            q, pad_width=pad_width, mode="constant", constant_values=0.0
        )

    # add beta term
    q += beta * jnp.einsum("lz,...zxy->...lxy", A, psi)

    # add planetary vorticity
    if f is not None:
        q += f

    # move q from node/center to center/node
    q: Float[Array, "Nl Nx-1 Ny-1"] = center_avg_2D(q)

    return q


def ssh_to_streamfn(ssh: Array, f0: float = 1e-5, g: float = GRAVITY) -> Array:
    """Calculates the ssh to stream function

    Eq:
        η = (g/f₀) Ψ

    Args:
        ssh (Array): the sea surface height [m]
        f0 (Array|float): the coriolis parameter
        g (float): the acceleration due to gravity

    Returns:
        psi (Array): the stream function
    """
    return (g / f0) * ssh


def streamfn_to_ssh(psi: Array, f0: float = 1e-5, g: float = GRAVITY) -> Array:
    """Calculates the stream function to ssh

    Eq:
        Ψ = (f₀/g) η

    Args:
        psi (Array): the stream function
        f0 (Array|float): the coriolis parameter
        g (float): the acceleration due to gravity

    Returns:
        ssh (Array): the sea surface height [m]
    """
    return (f0 / g) * psi
