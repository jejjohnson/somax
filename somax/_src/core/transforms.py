"""Stratification profiles and modal transforms for multilayer models."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from finitevolx import build_coupling_matrix, decompose_vertical_modes
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

    Computed from physical parameters (H, g_prime, f0) via
    eigendecomposition of the layer coupling matrix A (delegated
    to ``finitevolx.build_coupling_matrix`` and
    ``finitevolx.decompose_vertical_modes``).

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

        Delegates to ``finitevolx.build_coupling_matrix`` and
        ``finitevolx.decompose_vertical_modes``.

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
        rossby_radii, Cl2m, Cm2l = decompose_vertical_modes(A, f0)
        eigenvalues, _ = jnp.linalg.eigh(A)
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
