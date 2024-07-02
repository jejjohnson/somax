from typing import Optional

import einops
import equinox as eqx
from somax.domain import Domain
import jax
import jax.numpy as jnp
from jaxtyping import Array

from somax._src.models.qg.domain import LayerDomain
from somax._src.models.qg.params import QGParams
from somax._src.operators.functional.dst import (
    inverse_elliptic_dst,
    laplacian_dst,
)


class DSTSolution(eqx.Module):
    homsol: Array = eqx.static_field()
    homsol_mean: Array = eqx.static_field()
    H_mat: Array = eqx.static_field()
    capacitance_matrix: Optional[Array] = eqx.static_field()


def calculate_helmholtz_dst(
    domain: Domain, layer_domain: LayerDomain, params: QGParams
) -> Array:
    # get Laplacian dst transform
    # print(domain.Nx, domain.dx)
    L_mat = laplacian_dst(
        domain.Nx[0] - 2, domain.Nx[1] - 2, domain.dx[0], domain.dx[1]
    )
    # print_debug_quantity(L_mat, "L_MAT")
    # get beta term
    lambda_sq = einops.rearrange(layer_domain.lambda_sq, "Nz -> Nz 1 1")
    beta = params.f0**2 * lambda_sq

    # calculate helmholtz dst
    H_mat = L_mat - beta

    return H_mat


def compute_homogeneous_solution(u: Array, lambda_sq: Array, H_mat: Array):
    # create constant field
    constant_field = jnp.ones_like(u)

    # get homogeneous solution
    sol = jax.vmap(inverse_elliptic_dst, in_axes=(0, 0))(
        constant_field[..., 1:-1, 1:-1], H_mat
    )

    # calculate the homogeneous solution
    homsol = constant_field + sol * lambda_sq

    return homsol
