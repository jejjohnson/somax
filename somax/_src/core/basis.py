"""Reduced-order, differentiable basis forcing.

A :class:`BasisForcing` pairs a small, differentiable coefficient vector (the
control you solve for in DA) with a fixed spatial dictionary, a temporal gate,
and a per-mode prior. The forcing field at any time is one elementwise gating
followed by one contraction::

    eps(., t) = Phi @ (w * b(t))

where ``Phi`` is the spatial dictionary (columns are basis functions sampled on
the grid), ``w`` the coefficients, and ``b(t)`` the per-atom temporal weights.

This module is deliberately **dependency-free**: the spatial dictionary is a
plain precomputed array (in production it comes from evaluating a geonnax basis
on a :class:`~somax._src.domain.domain.Domain`, but that is the caller's
concern), and the temporal bases here (:class:`ConstantInTime`,
:class:`FourierInTime`) are implemented directly in ``jax.numpy``. See
``content/notes/forcing_basis.md`` for the full design.

The seam that lets any :class:`~somax._src.core.forcing.ForcingProtocol` enter
a model right-hand side is :class:`ForcingTerm`: it lifts a forcing *field*
``(t, grid) -> field`` onto a state component as a *tendency*
``(t, state, args) -> tendency``, the contract the term algebra
(:mod:`somax._src.core.terms`) evaluates. The existing QG forcing
``dq = dq + tau0 * wind_forcing`` is the special case of a one-column
``BasisForcing`` lifted by ``ForcingTerm(..., place=add_to("q"))``.
"""

from __future__ import annotations

import abc
from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, Float, PyTree

from somax._src.core.forcing import ForcingProtocol
from somax._src.core.terms import Term


class SpatialBasis(eqx.Module):
    """A precomputed spatial dictionary plus the per-mode prior std.

    ``Phi`` holds the basis functions sampled on the (flattened) grid, one per
    column. In production it is produced by evaluating a geonnax basis on a
    ``Domain``; here it is any precomputed array, so the basis math stays out of
    somax. ``std`` is the prior standard deviation per mode (``Lambda^{1/2}``),
    supplied by the prior layer (a kernel spectral density of eigenvalues for a
    spectral basis, or a prescribed / wavenumber law for a frame).

    Attributes:
        Phi: Dictionary of shape ``(Ngrid, m)`` on the flattened grid.
        std: Per-mode prior std of shape ``(m,)``.
    """

    Phi: Float[Array, " Ngrid m"]
    std: Float[Array, " m"]

    @classmethod
    def from_array(
        cls,
        Phi: Float[Array, " Ngrid m"],
        std: Float[Array, " m"] | None = None,
    ) -> SpatialBasis:
        """Build from a precomputed dictionary, defaulting ``std`` to ones.

        Args:
            Phi: Dictionary of shape ``(Ngrid, m)``.
            std: Optional per-mode prior std; defaults to ``ones(m)``.

        Returns:
            A :class:`SpatialBasis`.
        """
        Phi = jnp.asarray(Phi)
        std = jnp.ones(Phi.shape[1]) if std is None else jnp.asarray(std)
        return cls(Phi=Phi, std=std)

    def synthesize(self, coeffs: Float[Array, " m"]) -> Float[Array, " Ngrid"]:
        """Expand coefficients into a flat field, ``Phi @ coeffs``."""
        return self.Phi @ coeffs

    def analyze(self, field: Float[Array, " Ngrid"]) -> Float[Array, " m"]:
        """Frame analysis ``Phi^T @ field`` (adjoint of synthesis).

        Exact inverse of :meth:`synthesize` only for an orthonormal basis.
        """
        return self.Phi.T @ field

    def prior_std(self) -> Float[Array, " m"]:
        """Return the per-mode prior std ``Lambda^{1/2}``."""
        return self.std


class TemporalBasis(eqx.Module):
    """Maps a scalar time to per-atom temporal weights ``b(t)``.

    Subclasses implement :meth:`weights`. In production these wrap geonnax
    temporal features; the two below are implemented directly so the slice has
    no external dependency.
    """

    @abc.abstractmethod
    def weights(self, t: float) -> Float[Array, " m"]:
        """Return per-atom temporal weights at time ``t``."""
        ...


class ConstantInTime(TemporalBasis):
    """Time-independent gate: ``b(t) = ones(m)`` (the ``Phi_t = I`` case).

    Attributes:
        m: Number of atoms.
    """

    m: int = eqx.field(static=True)

    def weights(self, t: float) -> Float[Array, " m"]:
        return jnp.ones((self.m,))


class FourierInTime(TemporalBasis):
    """Cosine temporal gate ``b_a(t) = cos(omega_a t + phase_a)``.

    A one-mode instance reproduces the temporal part of
    :class:`~somax._src.core.forcing.SeasonalWindForcing`. The frequencies and
    phases are fixed (not part of the control); they are stored as array leaves
    but excluded from gradients by :func:`control_filter`.

    Attributes:
        freqs: Angular frequencies ``omega`` of shape ``(m,)``.
        phases: Phase offsets of shape ``(m,)``.
    """

    freqs: Float[Array, " m"]
    phases: Float[Array, " m"]

    def weights(self, t: float) -> Float[Array, " m"]:
        return jnp.cos(self.freqs * t + self.phases)


class BasisForcing(ForcingProtocol):
    """Reduced-order forcing: a fixed space-time frame driven by a coefficient vector.

    The only learnable leaf is :attr:`coeffs` (the DA control); the dictionary,
    the temporal gate, and the prior std are fixed. ``__call__`` returns a field
    shaped to the model grid; :class:`ForcingTerm` lifts it onto a state
    component as a tendency.

    ``SeasonalWindForcing`` is the special case of a one-column dictionary
    (``Phi = tau0_pattern[:, None]``) with a one-mode :class:`FourierInTime`.

    Attributes:
        coeffs: Learnable control of shape ``(m,)`` (visible to ``jax.grad``).
        spatial: Fixed spatial dictionary and prior std.
        temporal: Fixed temporal gate.
        grid_shape: Field shape ``domain.Nx`` used to reshape the flat synthesis.
    """

    coeffs: Float[Array, " m"]
    spatial: SpatialBasis
    temporal: TemporalBasis
    grid_shape: tuple[int, ...] = eqx.field(static=True)

    def __call__(self, t: float, grid: eqx.Module | None = None) -> Array:
        """Evaluate the forcing field at time ``t`` (shaped to ``grid_shape``)."""
        b = self.temporal.weights(t)  # (m,)
        active = self.coeffs * b  # (m,)
        flat = self.spatial.synthesize(active)  # (Ngrid,)
        return flat.reshape(self.grid_shape)  # (Ny, Nx)

    def whiten(self, u: Float[Array, " m"]) -> BasisForcing:
        """Return a copy with ``coeffs = Lambda^{1/2} u`` (diagonal prior).

        The flow-prior note replaces this with a learned generative map. Used
        to precondition the variational control so the prior term is
        ``0.5 ||u||^2``.
        """
        w = self.spatial.prior_std() * u
        return eqx.tree_at(lambda f: f.coeffs, self, w)

    def regularization(self) -> Float[Array, " "]:
        """Prior penalty ``0.5 * sum (w / sigma)^2`` for the diagonal prior.

        The flow-prior note replaces this with ``-prior.log_prob(w)``. Added to
        the variational cost.
        """
        std = self.spatial.prior_std()
        return 0.5 * jnp.sum((self.coeffs / std) ** 2)


class TransformedForcing(ForcingProtocol):
    """Apply a pointwise transform to a base forcing (e.g. log-space synthesis).

    For lognormal variables (ocean colour) synthesise in log space and map back
    with the inverse, keeping the field positive.

    Attributes:
        base: The forcing whose output is transformed.
        inverse: Pointwise map applied to the base output (e.g. ``10 ** z``).
    """

    base: ForcingProtocol
    inverse: Callable[[Array], Array] = eqx.field(static=True)

    def __call__(self, t: float, grid: eqx.Module | None = None) -> Array:
        return self.inverse(self.base(t, grid))


def add_to(
    component: str,
    layer: int | None = None,
) -> Callable[[PyTree, Array], PyTree]:
    """Build a placement that adds a field onto one named state component.

    Mirrors the QG/SWM convention of writing forcing onto a single tendency
    component (and optionally a single layer), e.g. ``dq[0] += field``.

    Args:
        component: Name of the state attribute to add the field to (e.g. ``"q"``).
        layer: Optional layer index for a stacked component; ``None`` adds to
            the whole component.

    Returns:
        A ``place(zeros, field) -> tendency`` callable for :class:`ForcingTerm`.
    """

    def _place(zeros: PyTree, field: Array) -> PyTree:
        leaf = getattr(zeros, component)
        leaf = leaf.at[layer].add(field) if layer is not None else leaf + field
        return eqx.tree_at(lambda s: getattr(s, component), zeros, leaf)

    return _place


class ForcingTerm(Term):
    """Lift a :class:`ForcingProtocol` field into the term algebra as a tendency.

    Resolves the contract mismatch between a forcing
    (``(t, grid) -> field``) and a model right-hand-side term
    (``(t, state, args) -> tendency``). ``place`` writes the field onto the
    target state component, returning a tendency pytree that is zero everywhere
    else — the generalisation of QG's ``dq = dq.at[0].add(tau0 * wind_forcing)``.

    Attributes:
        forcing: The forcing whose field is lifted.
        place: A ``(zeros_tendency, field) -> tendency`` placement, typically
            from :func:`add_to`.
    """

    forcing: ForcingProtocol
    place: Callable[[PyTree, Array], PyTree] = eqx.field(static=True)

    def __call__(self, t: float, state: PyTree, args: PyTree | None = None) -> PyTree:
        field = self.forcing(t, None)
        zeros = jtu.tree_map(lambda leaf: leaf * 0.0, state)
        return self.place(zeros, field)


def control_filter(forcing: BasisForcing) -> BasisForcing:
    """Boolean filter selecting only ``coeffs`` for gradient updates.

    Use with :func:`equinox.partition` so optimisers update the control vector
    only, leaving the (large) dictionary, the temporal centres/widths, and the
    prior std fixed::

        diff, static = eqx.partition(forcing, control_filter(forcing))

    Args:
        forcing: The forcing whose ``coeffs`` should be the trainable leaves.

    Returns:
        A like-structured pytree of booleans, ``True`` only at ``coeffs``.
    """
    filt = jtu.tree_map(lambda _: False, forcing)
    return eqx.tree_at(lambda f: f.coeffs, filt, replace=True)
