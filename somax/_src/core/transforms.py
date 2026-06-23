"""Stratification profiles and modal transforms for multilayer models."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from finitevolx import build_coupling_matrix
from jaxtyping import Array, Float


class StratificationProfile(eqx.Module):
    """Discrete vertical stratification for a layered ocean model.

    Stores layer thicknesses, reduced gravities, and (optionally) layer
    densities. Created from physical parameters via factory methods.

    Attributes:
        H: Layer resting thicknesses [m], shape ``(nl,)``, top to bottom.
        g_prime: Reduced gravities [m/s^2], shape ``(nl,)``.
            ``g_prime[i]`` is the reduced gravity at the interface above
            layer i.  For the top layer ``g_prime[0]`` equals full gravity
            (rigid-lid convention) or a free-surface reduced gravity.
        rho: Layer densities [kg/m^3], shape ``(nl,)``, or ``None``.
    """

    H: Array
    g_prime: Array
    rho: Array | None = None

    @property
    def nl(self) -> int:
        """Number of layers."""
        return self.H.shape[0]

    @property
    def total_depth(self) -> Array:
        """Total ocean depth [m] (JAX scalar, safe under jit)."""
        return jnp.sum(self.H)

    @staticmethod
    def from_N2_constant(
        N2: float,
        depth: float,
        n_layers: int,
        g: float = 9.81,
        rho0: float = 1025.0,
    ) -> StratificationProfile:
        """Build uniform stratification from a constant buoyancy frequency.

        Each layer has equal thickness ``depth / n_layers``. The reduced
        gravity between layers is derived from ``N^2 = -g/rho0 * drho/dz``:

            g_prime[k] = N^2 * H_k   for k >= 1
            g_prime[0] = g            (rigid-lid top interface)

        Args:
            N2: Buoyancy frequency squared [1/s^2].
            depth: Total ocean depth [m].
            n_layers: Number of layers.
            g: Full gravitational acceleration [m/s^2].
            rho0: Reference density [kg/m^3].

        Returns:
            A ``StratificationProfile`` instance.
        """
        H_val = depth / n_layers
        H = jnp.full(n_layers, H_val)
        g_prime_internal = N2 * H_val
        # Top interface: full gravity (rigid-lid convention)
        g_prime = jnp.concatenate(
            [jnp.array([g]), jnp.full(n_layers - 1, g_prime_internal)]
        )
        # Compute layer densities from N² = -g/rho0 * drho/dz
        drho = rho0 * N2 * H_val / g
        rho = rho0 + drho * jnp.arange(n_layers)
        return StratificationProfile(H=H, g_prime=g_prime, rho=rho)

    @staticmethod
    def from_N2_exponential(
        N2_surface: float,
        scale_depth: float,
        depth: float,
        n_layers: int,
        g: float = 9.81,
        rho0: float = 1025.0,
    ) -> StratificationProfile:
        """Build stratification from an exponential N^2(z) profile.

        N^2(z) = N2_surface * exp(z / scale_depth), where z <= 0
        (z=0 at the surface, z=-depth at the bottom).

        Args:
            N2_surface: Buoyancy frequency squared at the surface [1/s^2].
            scale_depth: e-folding depth [m] (positive value).
            depth: Total ocean depth [m].
            n_layers: Number of layers.
            g: Full gravitational acceleration [m/s^2].
            rho0: Reference density [kg/m^3].

        Returns:
            A ``StratificationProfile`` instance.
        """
        H_val = depth / n_layers
        H = jnp.full(n_layers, H_val)
        # N² at each interface (mid-points between layer centres)
        z_interfaces = -jnp.arange(1, n_layers) * H_val
        N2_interfaces = N2_surface * jnp.exp(z_interfaces / scale_depth)
        g_prime_internal = N2_interfaces * H_val
        g_prime = jnp.concatenate([jnp.array([g]), g_prime_internal])
        # Layer densities
        z_centres = -(jnp.arange(n_layers) + 0.5) * H_val
        N2_centres = N2_surface * jnp.exp(z_centres / scale_depth)
        drho = rho0 * N2_centres * H_val / g
        rho = rho0 + jnp.cumsum(drho)
        return StratificationProfile(H=H, g_prime=g_prime, rho=rho)

    @staticmethod
    def from_layers(
        H: tuple[float, ...] | list[float],
        g_prime: tuple[float, ...] | list[float],
        rho: tuple[float, ...] | list[float] | None = None,
    ) -> StratificationProfile:
        """Build stratification from explicit layer parameters.

        Args:
            H: Layer thicknesses [m], top to bottom.
            g_prime: Reduced gravities [m/s^2] at each interface.
            rho: Layer densities [kg/m^3], or None.

        Returns:
            A ``StratificationProfile`` instance.

        Raises:
            ValueError: If ``H`` and ``g_prime`` have different lengths,
                or if ``rho`` is provided with a different length.
        """
        if len(H) != len(g_prime):
            msg = f"H ({len(H)}) and g_prime ({len(g_prime)}) must have the same length"
            raise ValueError(msg)
        if rho is not None and len(rho) != len(H):
            msg = f"rho ({len(rho)}) must have the same length as H ({len(H)})"
            raise ValueError(msg)
        rho_arr = jnp.array(rho) if rho is not None else None
        return StratificationProfile(
            H=jnp.array(H),
            g_prime=jnp.array(g_prime),
            rho=rho_arr,
        )


class ModalTransform(eqx.Module):
    """Precomputed layer-to-mode and mode-to-layer transforms.

    Computed from physical parameters (H, g_prime, f0) via the
    eigendecomposition of the layer coupling matrix A (built by
    ``finitevolx.build_coupling_matrix``). A is non-symmetric for unequal
    layer thicknesses, so it is diagonalized through its symmetric similarity
    (see :meth:`from_physics`) rather than with a symmetric eigensolver.

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
        H: tuple[float, ...] | Array,
        g_prime: tuple[float, ...] | Array,
        f0: float,
    ) -> ModalTransform:
        """Build transform from physical parameters.

        Builds the layer coupling matrix with ``finitevolx.build_coupling_matrix``
        and diagonalizes it via its symmetric similarity (see body) so the modal
        transform is correct for unequal layer thicknesses (where the coupling
        matrix is not symmetric).

        Args:
            H: Layer depths (top to bottom).
            g_prime: Reduced gravities at each interface.
            f0: Coriolis parameter [1/s].

        Returns:
            A ``ModalTransform`` with precomputed projection matrices.
        """
        H_arr = jnp.asarray(H, dtype=float)
        gp_arr = jnp.asarray(g_prime, dtype=float)
        A = build_coupling_matrix(H_arr, gp_arr)

        # A = diag(1/H) @ B with B = diag(H) @ A symmetric, so A itself is NOT
        # symmetric for unequal layer thicknesses. Diagonalizing A with `eigh`
        # (which assumes symmetry and reads a single triangle) silently
        # decomposes the wrong matrix: the eigenvalues come out close, but the
        # eigenvectors do NOT diagonalize A, so the modal transform fails to
        # decouple the layers and the multilayer PV inversion solves the wrong
        # coupled elliptic problem. Diagonalize the symmetric similarity
        # S = D^(1/2) A D^(-1/2) (D = diag(H)) instead: it shares A's real,
        # non-negative eigenvalues, and A's layer eigenvectors are r = D^(-1/2) v
        # for S's orthonormal eigenvectors v. (MQGeometry uses a general
        # eigensolver for the same reason; the symmetric route is more accurate
        # and yields sorted, real eigenvalues without complex round-off.)
        sqrt_H = jnp.sqrt(H_arr)
        S = (sqrt_H[:, None] * A) / sqrt_H[None, :]
        S = 0.5 * (S + S.T)  # drop residual asymmetry from round-off
        eigenvalues, V = jnp.linalg.eigh(S)  # ascending, real; V orthonormal
        # A is positive semi-definite. eigh on the (correct) symmetric S keeps
        # the gravest eigenvalue accurate, but clamp to the physical floor of 0
        # as a guard: a *negative* barotropic eigenvalue would flip the sign of
        # the gravest-wavenumber denominators in the Helmholtz PV inversion
        # (lambda = f0^2 * eigenvalue) and diverge multilayer QG within ~2 weeks.
        eigenvalues = jnp.clip(eigenvalues, 0.0, None)
        Cm2l = V / sqrt_H[:, None]  # layer eigenvectors r = D^(-1/2) v (columns)
        Cl2m = jnp.linalg.inv(Cm2l)
        # Rossby radii: Rd = 1/(|f0| sqrt(lambda)); the gravest mode -> inf only
        # if its eigenvalue is exactly 0 (true rigid lid with no surface term).
        positive = eigenvalues > 0
        safe_eig = jnp.where(positive, eigenvalues, 1.0)
        finite_radii = 1.0 / (
            jnp.abs(jnp.asarray(f0, dtype=float)) * jnp.sqrt(safe_eig)
        )
        rossby_radii = jnp.where(positive, finite_radii, jnp.inf)
        return ModalTransform(
            Cl2m=Cl2m,
            Cm2l=Cm2l,
            eigenvalues=eigenvalues,
            rossby_radii=rossby_radii,
        )

    @staticmethod
    def from_stratification(
        strat: StratificationProfile,
        f0: float,
    ) -> ModalTransform:
        """Build transform from a stratification profile.

        Args:
            strat: A ``StratificationProfile`` instance.
            f0: Coriolis parameter [1/s].

        Returns:
            A ``ModalTransform`` with precomputed projection matrices.
        """
        return ModalTransform.from_physics(strat.H, strat.g_prime, f0)

    def to_modal(self, x: Float[Array, "nl ..."]) -> Float[Array, "nl ..."]:
        """Project from layer space to modal space."""
        return jnp.einsum("lm,m...->l...", self.Cl2m, x)

    def to_layer(self, x: Float[Array, "nl ..."]) -> Float[Array, "nl ..."]:
        """Reconstruct from modal space to layer space."""
        return jnp.einsum("lm,m...->l...", self.Cm2l, x)
