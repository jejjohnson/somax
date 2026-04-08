"""Precomputed transforms for multilayer models."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


def _build_coupling_matrix(
    H: tuple[float, ...],
    g_prime: tuple[float, ...],
) -> Float[Array, "nl nl"]:
    """Build the layer coupling matrix A from layer depths and reduced gravities.

    For an N-layer system, A is an NxN tridiagonal matrix encoding the
    vertical coupling between layers via the reduced gravities g'_k and
    layer depths H_k.

    Args:
        H: Layer depths (thickness) from top to bottom, length N.
        g_prime: Reduced gravities at each interface, length N.
            g_prime[0] is the free-surface reduced gravity (typically g),
            g_prime[k] for k>=1 are the interface reduced gravities.

    Returns:
        Coupling matrix A of shape (N, N).
    """
    nl = len(H)
    H_arr = jnp.array(H, dtype=jnp.float32)
    gp = jnp.array(g_prime, dtype=jnp.float32)

    A = jnp.zeros((nl, nl), dtype=jnp.float32)
    for k in range(nl):
        # Upper interface contribution
        A = A.at[k, k].add(-1.0 / (H_arr[k] * gp[k]))
        if k > 0:
            A = A.at[k, k - 1].add(1.0 / (H_arr[k] * gp[k]))
        # Lower interface contribution (if not bottom layer)
        if k < nl - 1:
            A = A.at[k, k].add(-1.0 / (H_arr[k] * gp[k + 1]))
            A = A.at[k, k + 1].add(1.0 / (H_arr[k] * gp[k + 1]))
    return A


class ModalTransform(eqx.Module):
    """Precomputed layer-to-mode and mode-to-layer transforms.

    Computed from physical parameters (H, g_prime, f0) via
    eigendecomposition of the layer coupling matrix A. Stored on the
    model as a static field and applied via ``to_modal`` / ``to_layer``.

    Attributes:
        Cl2m: Layer-to-mode projection matrix.
        Cm2l: Mode-to-layer reconstruction matrix.
        eigenvalues: Modal eigenvalues (related to 1/Rd^2).
        rossby_radii: Rossby deformation radii per mode [m].
    """

    Cl2m: Float[Array, "nl nl"]
    Cm2l: Float[Array, "nl nl"]
    eigenvalues: Array
    rossby_radii: Array

    @staticmethod
    def from_physics(
        H: tuple[float, ...],
        g_prime: tuple[float, ...],
        f0: float,
    ) -> ModalTransform:
        """Build transform from physical parameters.

        Args:
            H: Layer depths (top to bottom).
            g_prime: Reduced gravities at each interface.
            f0: Coriolis parameter [1/s].

        Returns:
            A ``ModalTransform`` with precomputed projection matrices.
        """
        A = _build_coupling_matrix(H, g_prime)
        eigenvalues, R = jnp.linalg.eigh(A)
        L = jnp.linalg.inv(R)
        rossby_radii = 1.0 / jnp.sqrt(jnp.abs(f0**2 * eigenvalues))
        return ModalTransform(
            Cl2m=L,
            Cm2l=R,
            eigenvalues=eigenvalues,
            rossby_radii=rossby_radii,
        )

    def to_modal(self, x: Float[Array, "nl ..."]) -> Float[Array, "nl ..."]:
        """Project from layer space to modal space."""
        return jnp.einsum("lm,m...->l...", self.Cl2m, x)

    def to_layer(self, x: Float[Array, "nl ..."]) -> Float[Array, "nl ..."]:
        """Reconstruct from modal space to layer space."""
        return jnp.einsum("lm,m...->l...", self.Cm2l, x)
