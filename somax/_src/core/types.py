"""Core type definitions for somax models."""

from __future__ import annotations

import dataclasses
import math
from typing import Any, ClassVar

import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, ArrayLike, PyTree
from paramax import AbstractUnwrappable, NonTrainable, Parameterize


class State(eqx.Module):
    """Base class for model state vectors.

    All model states should subclass this to enable interoperability
    with the somax model contract and JAX transformations.

    Two optional class attributes describe the fields to
    :class:`~somax._src.core.transforms.StateAffine`. Both are consulted
    before the name-based fallbacks, and both are per-state because a
    field name does not determine either answer: ``h`` is a total
    thickness in the nonlinear shallow-water models but a height
    anomaly in the linear ones, and ``u`` is a C-grid velocity in the
    ocean models but a T-point scalar in the pde family.

    Attributes:
        scale_kinds: Field name to semantic kind — one of
            ``"velocity"``, ``"thickness"``, ``"height_anomaly"``,
            ``"vorticity"``, ``"streamfunction"``. Fixes how
            ``StateAffine.from_scales`` non-dimensionalises the field.
        mask_locations: Field name to C-grid staggering — one of
            ``"h"``, ``"u"``, ``"v"``, ``"xy_corner"``, ``"w"``. Picks
            the mask that ``StateAffine.from_samples`` excludes dry
            cells with.
    """

    scale_kinds: ClassVar[dict[str, str]] = {}
    mask_locations: ClassVar[dict[str, str]] = {}


class Params(eqx.Module):
    """Base class for differentiable model parameters.

    Fields on Params subclasses are visible to ``jax.grad`` by default.
    Use ``eqx.field(static=True)`` for non-differentiable parameters.

    Fields may also hold a ``paramax`` wrapper instead of a bare array.
    :func:`positive` and :func:`interval` store a constrained value in an
    unconstrained space, and :func:`frozen` hides a value from gradients.
    A wrapped field is reconstituted by ``paramax.unwrap`` at RHS time —
    :meth:`somax.SomaxModel.build_terms` and
    :meth:`somax.SomaxModel.diagnose` both call it — so model code always
    sees the constrained value and never needs to know about the wrapper.

    Gradients of a wrapped field are taken **with respect to the
    unconstrained value**, not the constrained one. For a
    ``positive``-wrapped viscosity ``nu = softplus(r)`` the gradient lands
    on ``r``, so an optimiser stepping it can never drive ``nu`` negative.
    Chain-rule factors (``sigmoid(r)`` for ``positive``) mean the
    magnitudes differ from those of an unwrapped parameterisation; that
    is the point, and optimiser learning rates should be set accordingly.

    Example:
        >>> params = MyParams(lateral_viscosity=positive(100.0))
        >>> paramax.unwrap(params).lateral_viscosity  # 100.0
    """


class PhysConsts(eqx.Module):
    """Base class for frozen physical constants.

    All fields should be marked ``static=True`` so they are invisible
    to ``jax.grad`` and treated as compile-time constants.
    """


class Diagnostics(eqx.Module):
    """Base class for on-demand diagnostic quantities.

    Computed from a model state via ``model.diagnose(state)``.
    """

    def invariants(self) -> dict[str, Array]:
        """Conserved quantities this model advertises for drift tracking.

        Maps an invariant name to a scalar (or per-layer vector) that is
        *conserved* by the continuous dynamics — mass, total energy,
        potential enstrophy, PV Casimirs, momentum, … — so a monitor can
        track the relative drift ``(I(t) - I(0)) / I(0)`` over a run without
        knowing model specifics (see
        :class:`somax.monitor.ConservationDriftMonitor`).

        The base implementation returns an empty dict; models override it to
        expose the invariants their ``Diagnostics`` subclass carries. Whether
        a given invariant is *exactly* conserved depends on the scheme (e.g.
        the Arakawa Jacobian conserves energy/enstrophy up to time-truncation
        error; finite-volume upwind PV advection conserves mass to machine
        precision but dissipates energy/enstrophy implicitly), so callers
        should treat non-mass invariants as drift signals, not zero targets.

        Returns:
            Mapping from invariant name to its current value. Empty by
            default.
        """
        return {}


# ----------------------------------------------------------------------
# Constrained-parameter helpers
# ----------------------------------------------------------------------


def _inv_softplus(x: Array) -> Array:
    """Inverse of ``softplus``, computed stably for large ``x``.

    ``log(exp(x) - 1)`` overflows for moderate ``x``; the equivalent
    ``x + log1p(-exp(-x))`` does not.
    """
    return x + jnp.log(-jnp.expm1(-x))


@dataclasses.dataclass(frozen=True)
class _IntervalTransform:
    """``raw -> lower + (upper - lower) * sigmoid(raw)``.

    A module-level frozen dataclass rather than a closure built inside
    :func:`interval`. ``Parameterize`` keeps its transform as static
    pytree metadata, and two closures are never equal even when they
    capture the same bounds — so two independently built models with
    identical constraints would have different treedefs and could not
    be stacked into an ensemble with ``tree_map``. A frozen dataclass
    compares and hashes by its bounds, so equivalent wrappers stay
    structurally interchangeable.
    """

    lower: float
    upper: float

    def __call__(self, raw: Array) -> Array:
        return self.lower + (self.upper - self.lower) * jnn.sigmoid(raw)


def positive(value: ArrayLike) -> Parameterize:
    """Constrain a parameter to be strictly positive.

    The value is stored in softplus space and reconstituted as
    ``softplus(raw)`` by ``paramax.unwrap``, so gradient descent on the
    stored value cannot make it negative. Use it for quantities that are
    physically non-negative — lateral viscosity, bottom drag, wind
    amplitude — whenever they are being calibrated.

    The guarantee is exact in arithmetic and near-exact in floating
    point: ``softplus`` underflows to exactly ``0.0`` once the stored
    value falls below roughly ``-90`` in float32, so the constrained
    value is non-negative always and strictly positive everywhere an
    optimiser that has not already diverged will go. It is never
    negative.

    Args:
        value: The initial constrained value. Must be finite and
            strictly positive.

    Returns:
        A ``Parameterize`` that unwraps to ``value``.

    Raises:
        ValueError: If ``value`` is not strictly positive, which has no
            representation in softplus space.
    """
    array = jnp.asarray(value)
    # ``all`` of a positive predicate rather than ``any`` of its
    # negation: every comparison with NaN is false, so the negated form
    # would wave a NaN through and build a wrapper that unwraps to NaN.
    # Finiteness is tested separately because ``+inf`` satisfies
    # ``> 0``, stores an infinite raw value, and unwraps back to
    # ``inf`` — an overflowed calibration value is not a usable one.
    if not bool(jnp.all(jnp.isfinite(array) & (array > 0.0))):
        raise ValueError(
            f"positive(): value must be strictly positive and finite; got "
            f"{value!r}. A non-positive value has no softplus pre-image."
        )
    return Parameterize(jnn.softplus, _inv_softplus(array))


def interval(value: ArrayLike, lower: float, upper: float) -> Parameterize:
    """Constrain a parameter to the open interval ``(lower, upper)``.

    The value is stored in logit space and reconstituted as
    ``lower + (upper - lower) * sigmoid(raw)``. As with :func:`positive`,
    the bound is exact in arithmetic and saturating in floating point:
    far into either tail ``sigmoid`` reaches exactly 0 or 1, so the
    constrained value can land *on* a bound but never outside it.

    Args:
        value: The initial constrained value, strictly inside the interval.
        lower: Lower bound, exclusive. Must be finite.
        upper: Upper bound, exclusive. Must be finite.

    Returns:
        A ``Parameterize`` that unwraps to ``value``.

    Raises:
        ValueError: If the bounds are not finite or not ordered, or
            ``value`` lies outside the open interval.
    """
    if not (math.isfinite(lower) and math.isfinite(upper)):
        # A semi-infinite interval passes the ordering test but has no
        # usable logit: ``scaled`` collapses to 0 or 1, the stored raw
        # value becomes infinite, and unwrapping then evaluates
        # ``inf * sigmoid(-inf)``, which is NaN.
        raise ValueError(
            f"interval(): bounds must be finite; got ({lower!r}, {upper!r}). "
            f"A semi-infinite interval has no logit — use positive() for a "
            f"one-sided bound."
        )
    if not upper > lower:
        raise ValueError(
            f"interval(): upper must exceed lower; got ({lower!r}, {upper!r})."
        )
    array = jnp.asarray(value)
    # Stated positively so that NaN fails it — see :func:`positive`.
    if not bool(jnp.all(jnp.isfinite(array) & (array > lower) & (array < upper))):
        raise ValueError(
            f"interval(): value must lie strictly inside ({lower!r}, {upper!r}); "
            f"got {value!r}."
        )
    scaled = (array - lower) / (upper - lower)
    raw = jnp.log(scaled) - jnp.log1p(-scaled)
    return Parameterize(_IntervalTransform(float(lower), float(upper)), raw)


def frozen(value: ArrayLike) -> NonTrainable:
    """Hide a parameter from gradients while keeping it a runtime value.

    The leaf stays in the pytree but is cut out of the backward pass,
    so it behaves like an ``eqx.field(static=True)`` constant without
    having to be hashable or known at trace time. Its gradient comes
    back as exact zero rather than being absent.

    Args:
        value: The value to freeze.

    Returns:
        A ``NonTrainable`` wrapper that unwraps to ``value``.
    """
    return NonTrainable(jnp.asarray(value))


def as_parameter(value: ArrayLike | Parameterize | NonTrainable) -> Any:
    """Coerce a factory argument into a ``Params`` leaf.

    Model factories take plain numbers and convert them with
    ``jnp.asarray``, which cannot convert a paramax wrapper — so
    ``BarotropicQG.create(lateral_viscosity=positive(100.0))`` would
    fail, and a constrained model could only be built by surgery with
    ``eqx.tree_at``. Wrappers are passed through untouched; everything
    else is converted as before.

    Args:
        value: A number, array, or paramax wrapper.

    Returns:
        The wrapper unchanged, or ``jnp.asarray(value)``.
    """
    if isinstance(value, AbstractUnwrappable):
        return value
    return jnp.asarray(value)


def trainable_mask(tree: PyTree) -> PyTree:
    """Boolean pytree marking which leaves an optimiser may update.

    ``NonTrainable`` removes a leaf from the *backward* pass, so its
    gradient is exact zero — but a zero gradient is not the same as no
    update. A decoupled-weight-decay optimiser such as ``optax.adamw``
    computes its update from the parameter value as well as the
    gradient, so applying one to a whole model still drifts a
    :func:`frozen` constant. Pass this mask to ``optax.masked`` (or use
    it with ``eqx.partition``) to leave those leaves genuinely alone.

    Args:
        tree: Any pytree, typically a model.

    Returns:
        A pytree of the same structure whose leaves are ``True`` for
        trainable leaves and ``False`` under a ``NonTrainable``.

    Example:
        >>> optimiser = optax.masked(optax.adamw(1e-3), trainable_mask(model))
    """

    def mark(leaf: Any) -> Any:
        if isinstance(leaf, NonTrainable):
            return jtu.tree_map(lambda _: False, leaf)
        return True

    return jtu.tree_map(mark, tree, is_leaf=lambda x: isinstance(x, NonTrainable))
