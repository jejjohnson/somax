"""Stratification profiles and modal transforms for multilayer models."""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from finitevolx import build_coupling_matrix, decompose_vertical_modes
from jaxtyping import Array, Float, PyTree

from somax._src.core.scales import Scales


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


# ----------------------------------------------------------------------
# Affine state transforms (non-dimensionalisation and standardisation)
# ----------------------------------------------------------------------

#: Per-field ``(loc, scale)`` rules used by :meth:`StateAffine.from_scales`,
#: keyed by ``State`` field name. Each entry maps a ``Scales`` to the pair
#: that non-dimensionalises that field. Models may extend this mapping for
#: state fields of their own.
FIELD_SCALE_RULES: dict[str, Callable[[Scales], tuple[float, float]]] = {
    "u": lambda s: (0.0, s.U),
    "v": lambda s: (0.0, s.U),
    "w": lambda s: (0.0, s.U),
    "h": lambda s: (s.H, s.eta),
    "eta": lambda s: (0.0, s.eta),
    "q": lambda s: (0.0, s.vorticity),
    "psi": lambda s: (0.0, s.streamfunction),
    "zeta": lambda s: (0.0, s.vorticity),
}

#: Staggering location of each known state field, used to pick the right
#: C-grid mask in :meth:`StateAffine.from_samples`.
FIELD_MASK_LOCATION: dict[str, str] = {
    "u": "u",
    "v": "v",
    "q": "xy_corner",
    "zeta": "xy_corner",
}


class StateAffine(eqx.Module):
    """Per-leaf affine map on a ``State`` pytree: ``y = (x - loc) / scale``.

    One abstraction serves both jobs that need a change of state
    variables:

    * **Non-dimensionalisation** — ``loc`` and ``scale`` come from a
      :class:`~somax._src.core.scales.Scales` via :meth:`from_scales`,
      giving ``u' = u/U``, ``h' = (h - H)/dH``, ``q' = q/(U/L)``.
    * **Standardisation** — ``loc`` and ``scale`` are the sample mean
      and standard deviation from :meth:`from_samples`, per field or
      per gridpoint.

    They are the same map, so they compose (:meth:`compose`), invert,
    and can be used interchangeably by the DA flattening bridge, by
    ML input/output pipelines, and by ``ScaledModel``.

    Attributes:
        loc: Pytree matching the state's structure; each leaf is
            broadcastable against the corresponding field.
        scale: Pytree of the same structure. Leaves must be non-zero.

    Notes:
        ``loc`` and ``scale`` are ordinary pytrees, so a leaf may be a
        scalar (one number for the whole field), a per-layer column of
        shape ``(nl, 1, 1)``, or a full per-gridpoint array.
    """

    loc: PyTree
    scale: PyTree

    # ------------------------------------------------------------------
    # Core maps
    # ------------------------------------------------------------------

    def forward(self, state: PyTree) -> PyTree:
        """Map a state into transformed coordinates: ``(x - loc) / scale``."""
        return jtu.tree_map(
            lambda x, loc, scale: (x - loc) / scale, state, self.loc, self.scale
        )

    def inverse(self, state: PyTree) -> PyTree:
        """Map a transformed state back: ``x * scale + loc``."""
        return jtu.tree_map(
            lambda y, loc, scale: y * scale + loc, state, self.loc, self.scale
        )

    def log_det(self, state: PyTree) -> Array:
        """Log-determinant of the forward map's Jacobian at ``state``.

        The Jacobian of a fixed affine map is the constant diagonal
        ``1 / scale``, so this does not depend on the values in
        ``state`` — only on its shapes, which say how many elements a
        broadcast scalar ``scale`` covers.

        Args:
            state: A state, used for its leaf shapes only.

        Returns:
            Scalar ``-sum(n_elements * log(scale))`` over all leaves.
        """
        terms = jtu.tree_map(
            lambda x, scale: jnp.sum(
                jnp.broadcast_to(jnp.log(jnp.abs(scale)), jnp.shape(x))
            ),
            state,
            self.scale,
        )
        leaves = jtu.tree_leaves(terms)
        total = sum(leaves) if leaves else jnp.asarray(0.0)
        return -jnp.asarray(total)

    # flowjax-compatible signatures. Keeping these means a learned prior
    # can chain a flow onto a fixed physical scaling without an adapter
    # layer in between (see the optional ``somax[flows]`` extra).
    def transform_and_log_det(
        self, state: PyTree, condition: PyTree | None = None
    ) -> tuple[PyTree, Array]:
        """``(forward(state), log_det(state))``; ``condition`` is ignored."""
        del condition
        return self.forward(state), self.log_det(state)

    def inverse_and_log_det(
        self, state: PyTree, condition: PyTree | None = None
    ) -> tuple[PyTree, Array]:
        """``(inverse(state), -log_det(state))``; ``condition`` is ignored."""
        del condition
        return self.inverse(state), -self.log_det(state)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def identity(cls, state: PyTree) -> StateAffine:
        """The no-op transform matching ``state``'s structure."""
        return cls(
            loc=jtu.tree_map(lambda _: 0.0, state),
            scale=jtu.tree_map(lambda _: 1.0, state),
        )

    @classmethod
    def from_scales(
        cls,
        state_cls: type,
        scales: Scales,
        **per_field: float | tuple[float, float] | dict[str, Any],
    ) -> StateAffine:
        """Build the physical non-dimensionalising transform for a state type.

        Each field of ``state_cls`` is looked up in
        :data:`FIELD_SCALE_RULES`; a field with no rule gets the
        identity ``(0, 1)`` and a warning, since silently leaving a
        field dimensional is the kind of thing that produces a
        plausible-looking wrong answer.

        Args:
            state_cls: The ``State`` subclass to build the transform
                for. Its dataclass fields define the pytree structure.
            scales: Characteristic scales to derive ``loc``/``scale``
                from.
            **per_field: Overrides keyed by field name. A bare number
                sets ``scale`` and leaves ``loc`` at zero; a
                ``(loc, scale)`` pair or a ``{"loc": ..., "scale": ...}``
                mapping sets both. Values may be arrays, which is how
                a multilayer model supplies a per-layer ``(nl, 1, 1)``
                thickness scale.

        Returns:
            A ``StateAffine`` whose ``loc``/``scale`` are instances of
            ``state_cls``.

        Raises:
            ValueError: If an override names a field the state does not
                have, or if any resolved scale is zero.
        """
        fields = _state_field_names(state_cls)

        unknown = set(per_field) - set(fields)
        if unknown:
            raise ValueError(
                f"StateAffine.from_scales: {state_cls.__name__} has no field(s) "
                f"{sorted(unknown)}. Known fields: {sorted(fields)}."
            )

        locs: dict[str, Any] = {}
        scale_values: dict[str, Any] = {}
        undescribed: list[str] = []

        for name in fields:
            if name in per_field:
                loc, scale = _parse_override(name, per_field[name])
            elif name in FIELD_SCALE_RULES:
                loc, scale = FIELD_SCALE_RULES[name](scales)
            else:
                loc, scale = 0.0, 1.0
                undescribed.append(name)
            locs[name] = loc
            scale_values[name] = scale

        if undescribed:
            warnings.warn(
                f"StateAffine.from_scales: no scale rule for field(s) "
                f"{sorted(undescribed)} of {state_cls.__name__}; they are left "
                "dimensional (loc=0, scale=1). Pass an explicit override, or "
                "register a rule in FIELD_SCALE_RULES.",
                UserWarning,
                stacklevel=2,
            )

        _check_nonzero_scales(scale_values)
        return cls(loc=state_cls(**locs), scale=state_cls(**scale_values))

    @classmethod
    def from_samples(
        cls,
        states: PyTree,
        *,
        per_gridpoint: bool = False,
        eps: float = 1e-8,
        mask: Any | None = None,
    ) -> StateAffine:
        """Build a standardising transform from stacked samples.

        Args:
            states: A state pytree whose leaves carry a leading sample
                axis (time or ensemble member).
            per_gridpoint: If ``True``, reduce over the sample axis only,
                giving one ``(loc, scale)`` per gridpoint. If ``False``
                (default), reduce over every axis, giving one number per
                field.
            eps: Lower bound on the returned scale. A field the dynamics
                never touch has exactly zero sample variance, and
                dividing by it would produce inf/NaN.
            mask: Optional finitevolx ``Mask2D``/``Mask3D``. Dry cells
                are excluded from the statistics and are given the
                identity ``(0, 1)``, so masked fields round-trip
                unchanged and land sentinels cannot contaminate wet
                cells. The C-grid location is chosen per field via
                :data:`FIELD_MASK_LOCATION`.

        Returns:
            A ``StateAffine`` whose ``loc``/``scale`` share the state's
            structure.
        """
        names = _leaf_field_names(states)

        def moments(name: str, samples: Array) -> tuple[Array, Array]:
            wet = _field_mask(mask, name, samples, per_gridpoint=per_gridpoint)
            axis: int | tuple[int, ...]
            axis = 0 if per_gridpoint else tuple(range(jnp.ndim(samples)))
            return _masked_moments(samples, wet, axis=axis, eps=eps)

        pairs = _tree_map_with_names(moments, states, names)
        return cls(
            loc=jtu.tree_map(lambda p: p[0], pairs, is_leaf=_is_pair),
            scale=jtu.tree_map(lambda p: p[1], pairs, is_leaf=_is_pair),
        )

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    def compose(self, other: StateAffine) -> StateAffine:
        """Return the transform applying ``other`` first, then ``self``.

        ``self.compose(other).forward(x) == self.forward(other.forward(x))``.
        With ``y = (x - l2)/s2`` and ``z = (y - l1)/s1`` the composition
        is ``z = (x - (l2 + s2 * l1)) / (s1 * s2)``.

        Args:
            other: The transform applied first.

        Returns:
            The composed ``StateAffine``.
        """
        loc = jtu.tree_map(
            lambda l1, s2, l2: l2 + s2 * l1, self.loc, other.scale, other.loc
        )
        scale = jtu.tree_map(lambda s1, s2: s1 * s2, self.scale, other.scale)
        return StateAffine(loc=loc, scale=scale)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _state_field_names(state_cls: type) -> tuple[str, ...]:
    """Dataclass field names of an equinox ``Module`` subclass."""
    fields = getattr(state_cls, "__dataclass_fields__", None)
    if not fields:
        raise TypeError(
            f"StateAffine.from_scales: {state_cls!r} is not an equinox Module "
            "with dataclass fields."
        )
    return tuple(fields)


def _leaf_field_names(state: PyTree) -> list[str]:
    """Field names of ``state``'s leaves, in tree-flatten order.

    Falls back to empty names for pytrees that are not modules, which
    simply means no mask location is inferred for those leaves.
    """
    fields = getattr(type(state), "__dataclass_fields__", None)
    if not fields:
        return [""] * len(jtu.tree_leaves(state))
    names = []
    for name in fields:
        value = getattr(state, name)
        names.extend([name] * len(jtu.tree_leaves(value)))
    return names


def _tree_map_with_names(
    fn: Callable[[str, Array], Any], tree: PyTree, names: list[str]
) -> PyTree:
    """``tree_map`` that also passes each leaf's field name."""
    leaves, treedef = jtu.tree_flatten(tree)
    mapped = [fn(name, leaf) for name, leaf in zip(names, leaves, strict=True)]
    return jtu.tree_unflatten(treedef, mapped)


def _is_pair(x: Any) -> bool:
    return isinstance(x, tuple) and len(x) == 2


def _parse_override(
    name: str, value: float | tuple[float, float] | dict[str, Any]
) -> tuple[Any, Any]:
    """Normalise a ``from_scales`` override into a ``(loc, scale)`` pair."""
    if isinstance(value, dict):
        extra = set(value) - {"loc", "scale"}
        if extra:
            raise ValueError(
                f"StateAffine.from_scales: override for {name!r} has unexpected "
                f"key(s) {sorted(extra)}; expected 'loc' and/or 'scale'."
            )
        return value.get("loc", 0.0), value.get("scale", 1.0)
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(
                f"StateAffine.from_scales: tuple override for {name!r} must be "
                f"(loc, scale); got {len(value)} element(s)."
            )
        return value[0], value[1]
    return 0.0, value


def _check_nonzero_scales(scales: dict[str, Any]) -> None:
    """Reject a zero scale, which would make the transform non-invertible."""
    for name, value in scales.items():
        if jnp.ndim(value) == 0 and float(value) == 0.0:
            raise ValueError(
                f"StateAffine.from_scales: scale for {name!r} is zero, so the "
                "transform would not be invertible."
            )


def _field_mask(
    mask: Any | None,
    name: str,
    samples: Array,
    *,
    per_gridpoint: bool,
) -> Array | None:
    """Boolean wet-cell array for one field, broadcast against ``samples``."""
    del per_gridpoint
    if mask is None:
        return None
    location = FIELD_MASK_LOCATION.get(name, "h")
    wet = getattr(mask, location, None)
    if wet is None:  # pragma: no cover - defensive, masks carry these fields
        wet = mask.h
    return jnp.broadcast_to(wet, jnp.shape(samples))


def _masked_moments(
    samples: Array,
    wet: Array | None,
    *,
    axis: int | tuple[int, ...],
    eps: float,
) -> tuple[Array, Array]:
    """``(mean, std)`` over ``axis``, excluding dry cells, floored at ``eps``.

    Mirrors ``finitevolx.masked_moments``. Dry cells return ``(0, 1)``
    so that ``(x - loc) / scale`` leaves them untouched, and the floor
    is applied to the *variance* as ``eps**2`` rather than to the
    standard deviation: the value is identical, but ``sqrt`` has an
    infinite derivative at zero, so flooring afterwards would return a
    NaN gradient for a constant field.
    """
    if wet is None:
        count = jnp.asarray(
            math.prod(
                jnp.shape(samples)[a]
                for a in (axis if isinstance(axis, tuple) else (axis,))
            ),
            dtype=samples.dtype,
        )
        safe = samples
    else:
        safe = jnp.where(wet, samples, 0.0)
        count = jnp.sum(wet.astype(samples.dtype), axis=axis)

    empty = count == 0
    mean = jnp.where(
        empty, 0.0, jnp.sum(safe, axis=axis) / jnp.where(empty, 1.0, count)
    )

    dev = safe - jnp.expand_dims(mean, axis)
    if wet is not None:
        dev = jnp.where(wet, dev, 0.0)
    var = jnp.sum(dev**2, axis=axis) / jnp.where(empty, 1.0, count)
    std = jnp.sqrt(jnp.maximum(jnp.where(empty, 0.0, var), eps**2))

    return jnp.where(empty, 0.0, mean), jnp.where(empty, 1.0, std)
