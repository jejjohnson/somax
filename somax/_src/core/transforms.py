"""Stratification profiles and modal transforms for multilayer models."""

from __future__ import annotations

import dataclasses
import math
import warnings
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.core
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
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

#: ``(loc, scale)`` rule per *semantic kind* of field. A kind says what
#: a field means physically — which is what fixes its scaling — rather
#: than what it happens to be called.
SCALE_KIND_RULES: dict[str, Callable[[Scales], tuple[float, float]]] = {
    "velocity": lambda s: (0.0, s.U),
    "thickness": lambda s: (s.H, s.eta),
    "height_anomaly": lambda s: (0.0, s.eta),
    "vorticity": lambda s: (0.0, s.vorticity),
    "streamfunction": lambda s: (0.0, s.streamfunction),
    # A transported scalar has no scale in ``Scales`` — its amplitude
    # is set by the initial condition, not by the flow — so it is left
    # alone. Declared rather than left unknown so it does not warn.
    "tracer": lambda s: (0.0, 1.0),
}

#: Fallback kind per field *name*, for a state that declares nothing.
#: A name is only a hint: ``h`` is a total thickness in the nonlinear
#: shallow-water models but a height anomaly in the linear ones, and
#: ``u`` is a C-grid velocity in the ocean models but a T-point scalar
#: in the pde family. A state whose fields depart from these defaults
#: says so with :attr:`~somax._src.core.types.State.scale_kinds`, which
#: is consulted first.
DEFAULT_FIELD_KINDS: dict[str, str] = {
    "u": "velocity",
    "v": "velocity",
    "w": "velocity",
    "h": "thickness",
    "eta": "height_anomaly",
    "q": "vorticity",
    "zeta": "vorticity",
    "omega": "vorticity",
    "psi": "streamfunction",
}

#: Fallback C-grid staggering per field *name*, same caveat as above;
#: :attr:`~somax._src.core.types.State.mask_locations` wins.
DEFAULT_FIELD_MASK_LOCATION: dict[str, str] = {
    "u": "u",
    "v": "v",
    "q": "xy_corner",
    "zeta": "xy_corner",
}


def _field_kind(state_cls: type, name: str) -> str | None:
    """Semantic kind of ``state_cls.name``, or ``None`` if unknown."""
    declared = getattr(state_cls, "scale_kinds", {})
    if name in declared:
        return declared[name]
    return DEFAULT_FIELD_KINDS.get(name)


def _mask_location(state_cls: type | None, name: str) -> str:
    """C-grid location of ``state_cls.name``; T-points if unknown."""
    declared = getattr(state_cls, "mask_locations", {}) if state_cls else {}
    if name in declared:
        return declared[name]
    return DEFAULT_FIELD_MASK_LOCATION.get(name, "h")


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

        Each field of ``state_cls`` is resolved to a semantic kind —
        ``state_cls.scale_kinds`` first, then :data:`DEFAULT_FIELD_KINDS`
        — and scaled by the matching entry of :data:`SCALE_KIND_RULES`.
        A field with no kind gets the identity ``(0, 1)`` and a
        warning, since silently leaving a field dimensional is the kind
        of thing that produces a plausible-looking wrong answer.

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
            elif (kind := _field_kind(state_cls, name)) in SCALE_KIND_RULES:
                loc, scale = SCALE_KIND_RULES[kind](scales)
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
                "declare its kind in the state's scale_kinds.",
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
            eps: Lower bound on the returned scale, strictly positive.
                A field the dynamics never touch has exactly zero
                sample variance, and dividing by it would produce
                inf/NaN.
            mask: Optional finitevolx ``Mask2D``/``Mask3D``. Dry cells
                are excluded from the statistics and are given the
                identity ``(0, 1)``, so masked fields round-trip
                unchanged and land sentinels cannot contaminate wet
                cells. The C-grid location is taken from the state's
                :attr:`~somax._src.core.types.State.mask_locations`,
                falling back to :data:`DEFAULT_FIELD_MASK_LOCATION`.

                With ``per_gridpoint=False`` the wet-cell statistics
                are still a single number per field, but they are
                broadcast back over the grid so that dry cells keep
                the identity ``(0, 1)``.

        Returns:
            A ``StateAffine`` whose ``loc``/``scale`` share the state's
            structure.

        Raises:
            ValueError: If ``eps`` is not finite and strictly positive.
        """
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError(
                f"StateAffine.from_samples: eps must be a finite positive "
                f"number; got {eps!r}. It is the floor on the returned "
                f"scale, so zero leaves a constant field with a scale of "
                f"zero and a transform that cannot be inverted."
            )
        names = _leaf_field_names(states)

        state_cls = type(states)

        def moments(name: str, samples: Array) -> tuple[Array, Array]:
            wet = _field_mask(mask, name, samples, state_cls=state_cls)
            axis: int | tuple[int, ...]
            axis = 0 if per_gridpoint else tuple(range(jnp.ndim(samples)))
            mean, std = _masked_moments(samples, wet, axis=axis, eps=eps)
            if wet is None or per_gridpoint:
                return _Moments(mean, std)
            # Fieldwise reduction collapses to a scalar, which would
            # hand a dry cell the wet cells' statistics and so move its
            # land sentinel and count it in the log-determinant.
            # Broadcast back over the grid, identity on land.
            dry_safe = wet[0] if jnp.ndim(wet) == jnp.ndim(samples) else wet
            return _Moments(
                jnp.where(dry_safe, mean, 0.0),
                jnp.where(dry_safe, std, 1.0),
            )

        pairs = _tree_map_with_names(moments, states, names)
        return cls(
            loc=jtu.tree_map(lambda p: p.mean, pairs),
            scale=jtu.tree_map(lambda p: p.std, pairs),
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
    """Dataclass field names of an equinox ``Module`` subclass.

    ``dataclasses.fields`` rather than ``__dataclass_fields__``: the
    latter also lists ``ClassVar`` pseudo-fields, which is how a state
    declares its ``scale_kinds`` and ``mask_locations``, and those are
    metadata about the state rather than part of it.
    """
    if not getattr(state_cls, "__dataclass_fields__", None):
        raise TypeError(
            f"StateAffine.from_scales: {state_cls!r} is not an equinox Module "
            "with dataclass fields."
        )
    return tuple(field.name for field in dataclasses.fields(state_cls))


def _leaf_field_names(state: PyTree) -> list[str]:
    """Field names of ``state``'s leaves, in tree-flatten order.

    Falls back to empty names for pytrees that are not modules, which
    simply means no mask location is inferred for those leaves.
    """
    if not getattr(type(state), "__dataclass_fields__", None):
        return [""] * len(jtu.tree_leaves(state))
    names = []
    for name in _state_field_names(type(state)):
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


@dataclasses.dataclass(frozen=True)
class _Moments:
    """A ``(mean, std)`` pair produced by :meth:`StateAffine.from_samples`.

    A plain 2-tuple would be ambiguous: ``from_samples`` accepts any
    pytree, and a state that *is* a two-element tuple — or that holds
    one — would have its own structural node mistaken for a generated
    pair, splitting one child's statistics into ``loc`` and the
    other's into ``scale``. This type is not a registered pytree node,
    so ``tree_map`` treats it as an opaque leaf and only these pairs
    can be unpacked.
    """

    mean: Any
    std: Any


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
    """Reject a zero scale, which would make the transform non-invertible.

    Every entry is checked, not just scalars: the per-layer
    ``(nl, 1, 1)`` and per-gridpoint overrides are the whole point of
    allowing arrays here, and one zero among them divides by zero in
    :meth:`StateAffine.forward` and sends the log-determinant to
    infinity just as surely as a scalar zero would.

    Traced values are skipped — there is nothing to test at trace time,
    and raising on a tracer would make the constructor unusable inside
    ``jit``.
    """
    for name, value in scales.items():
        array = jnp.asarray(value)
        if isinstance(array, jax.core.Tracer):  # pragma: no cover - jit only
            continue
        values = np.asarray(array)
        if bool(np.any(values == 0.0)):
            where = "is zero" if array.ndim == 0 else "has a zero entry"
            raise ValueError(
                f"StateAffine.from_scales: scale for {name!r} {where}, so the "
                "transform would not be invertible."
            )
        # Non-finite is as fatal as zero, and the zero test misses it:
        # NaN contaminates every result, and an infinite scale maps a
        # finite value to zero with no way back. The log-determinant is
        # non-finite either way.
        if not bool(np.all(np.isfinite(values))):
            where = "is not finite" if array.ndim == 0 else "has a non-finite entry"
            raise ValueError(
                f"StateAffine.from_scales: scale for {name!r} {where}, so the "
                "transform would not be invertible."
            )


def _field_mask(
    mask: Any | None,
    name: str,
    samples: Array,
    *,
    state_cls: type | None = None,
) -> Array | None:
    """Boolean wet-cell array for one field, broadcast against ``samples``."""
    if mask is None:
        return None
    location = _mask_location(state_cls, name)
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
    # At least float32, whatever the samples are. ``float16`` saturates
    # at 65504, so a count over a 256x256 field overflows to infinity
    # and every mean it divides collapses silently to zero; the same
    # overflow corrupts the variance denominator.
    dtype = jnp.promote_types(jnp.asarray(samples).dtype, jnp.float32)
    if wet is None:
        count = jnp.asarray(
            math.prod(
                jnp.shape(samples)[a]
                for a in (axis if isinstance(axis, tuple) else (axis,))
            ),
            dtype=dtype,
        )
        safe = jnp.asarray(samples, dtype=dtype)
    else:
        safe = jnp.where(wet, samples, 0.0).astype(dtype)
        count = jnp.sum(wet.astype(dtype), axis=axis)

    empty = count == 0
    mean = jnp.where(
        empty, 0.0, jnp.sum(safe, axis=axis) / jnp.where(empty, 1.0, count)
    )

    dev = safe - jnp.expand_dims(mean, axis)
    if wet is not None:
        dev = jnp.where(wet, dev, 0.0)
    var = jnp.sum(dev**2, axis=axis) / jnp.where(empty, 1.0, count)

    # The doubled ``where`` keeps the zero away from ``sqrt`` on the
    # backward pass — its derivative there is infinite, and a constant
    # field is exactly what ``eps`` exists for — while the outer one
    # restores the exact value on the forward pass. The floor then goes
    # on the standard deviation itself, not on the variance as
    # ``eps**2``: squaring underflows for a small floor (``1e-30``
    # squares to ``1e-60``, which is zero in float32), putting back the
    # zero scale the floor is there to prevent.
    positive = var > 0.0
    std = jnp.where(positive, jnp.sqrt(jnp.where(positive, var, 1.0)), 0.0)
    std = jnp.maximum(std, eps)

    return jnp.where(empty, 0.0, mean), jnp.where(empty, 1.0, std)
