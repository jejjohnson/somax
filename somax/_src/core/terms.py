"""Composable term algebra for somax model right-hand sides.

A PDE right-hand side is additive::

    d(state)/dt = advection(state) + diffusion(state) + coriolis(state) + forcing(t)

This module makes that decomposition first-class. Each additive
contribution is a :class:`Term` — an ``eqx.Module`` mapping
``(t, state, args) -> tendency`` (a pytree with the same structure as
``state``). Terms form a small closed algebra:

- ``a + b``            → :class:`Sum`     (additive composition)
- ``c * a`` / ``a * c``→ :class:`Scaled`  (scalar coefficient; differentiable)
- ``-a`` / ``a - b``   → sugar over ``Scaled`` / ``Sum``
- ``a @ b``            → :class:`Compose` (operator composition: ``a(b(state))``)

Terms are the somax-internal protocol; the *assembled* term tree is what
a :class:`~somax._src.core.model.SomaxModel` evaluates in its
``vector_field``, so the composed object satisfies the
``pipekit_cycle.ForwardModel`` protocol (via ``SomaxModel.step``) without
the algebra itself importing pipekit.

IMEX / operator splitting
-------------------------
Every term carries a ``kind`` label — ``"explicit"`` (default) or
``"implicit"``. Use :func:`implicit` / :func:`explicit` to tag a term.
:func:`partition` flattens a (possibly nested) :class:`Sum` and groups
its summands by kind, and :func:`build_diffrax_terms` assembles the
appropriate diffrax term object:

- a single :class:`diffrax.ODETerm` when every summand shares a kind, or
- a :class:`diffrax.MultiTerm` of (explicit, implicit) ``ODETerm`` s,

which an IMEX solver (``diffrax.KenCarp3``, ``diffrax.Sil3``, …) can
integrate by routing the stiff implicit part through its implicit stage.
This is the seam that lets a model "mix and match which time stepper
handles which term".

The coefficients on :class:`Scaled` are ordinary JAX leaves, so a term
tree is differentiable end-to-end — ``jax.grad`` of a loss with respect
to a diffusion coefficient flows straight through ``Scaled.coeff``.
"""

from __future__ import annotations

import abc
from collections.abc import Callable, Iterator

import diffrax as dfx
import equinox as eqx
import jax.tree_util as jtu
from jaxtyping import Array, PyTree


Kind = str  # "explicit" | "implicit"

EXPLICIT: Kind = "explicit"
IMPLICIT: Kind = "implicit"
MIXED: Kind = "mixed"


class Term(eqx.Module):
    """A single additive contribution to a model right-hand side.

    Subclasses implement :meth:`__call__` with the signature
    ``(t, state, args=None) -> tendency`` where ``tendency`` is a pytree
    matching the structure of ``state``.

    Terms compose via the arithmetic operators (:meth:`__add__`,
    :meth:`__mul__`, :meth:`__sub__`, :meth:`__neg__`, :meth:`__matmul__`).
    The :attr:`kind` property drives IMEX partitioning; leaf terms are
    ``"explicit"`` by default — wrap with :func:`implicit` to mark a term
    for the implicit stage of a splitting integrator.
    """

    @abc.abstractmethod
    def __call__(self, t: float, state: PyTree, args: PyTree | None = None) -> PyTree:
        """Return this term's contribution to the tendency at time ``t``."""
        ...

    # -- IMEX label ----------------------------------------------------

    @property
    def kind(self) -> Kind:
        """Integration kind for IMEX splitting (``"explicit"`` by default)."""
        return EXPLICIT

    # -- algebra -------------------------------------------------------

    def __add__(self, other: Term) -> Term:
        if other is _ZERO:
            return self
        return Sum.of(self, other)

    def __radd__(self, other: Term) -> Term:
        # Enables ``sum(terms)`` which seeds with the int 0.
        if other == 0 or other is _ZERO:
            return self
        return Sum.of(other, self)

    def __mul__(self, coeff: float | Array) -> Term:
        return Scaled(self, coeff)

    def __rmul__(self, coeff: float | Array) -> Term:
        return Scaled(self, coeff)

    def __neg__(self) -> Term:
        return Scaled(self, -1.0)

    def __sub__(self, other: Term) -> Term:
        return Sum.of(self, Scaled(other, -1.0))

    def __matmul__(self, inner: Term) -> Term:
        """Operator composition: ``(self @ inner)(state) == self(inner(state))``."""
        return Compose(self, inner)


class TermFn(Term):
    """Adapt a plain callable into a :class:`Term`.

    Useful for wrapping an existing model's per-physics function, or for
    quick composition in tests. The callable must accept
    ``(t, state, args)`` and return a tendency pytree.

    Args:
        fn: The vector-field callable.
        kind: IMEX label for the wrapped function. Defaults to
            ``"explicit"``.
    """

    fn: Callable[[float, PyTree, PyTree | None], PyTree]
    _kind: Kind = eqx.field(static=True, default=EXPLICIT)

    def __call__(self, t: float, state: PyTree, args: PyTree | None = None) -> PyTree:
        return self.fn(t, state, args)

    @property
    def kind(self) -> Kind:
        return self._kind


class Sum(Term):
    """Additive composition of terms: ``sum_i terms[i](t, state, args)``.

    Construct via the ``+`` operator or :meth:`of` (which flattens nested
    sums so ``(a + b) + c`` and ``a + (b + c)`` yield the same flat tree).

    Args:
        terms: The summands. Leaves are added leaf-wise across the pytree.
    """

    terms: tuple[Term, ...]

    @classmethod
    def of(cls, *terms: Term) -> Term:
        """Build a flattened :class:`Sum`, dropping :data:`_ZERO` summands.

        Returns the lone term unchanged when only one survives, so the
        algebra stays free of redundant single-element ``Sum`` nodes.
        """
        flat: list[Term] = []
        for t in terms:
            if t is _ZERO:
                continue
            if isinstance(t, Sum):
                flat.extend(t.terms)
            else:
                flat.append(t)
        if not flat:
            return _ZERO
        if len(flat) == 1:
            return flat[0]
        return cls(tuple(flat))

    def __call__(self, t: float, state: PyTree, args: PyTree | None = None) -> PyTree:
        contributions = [term(t, state, args) for term in self.terms]
        acc = contributions[0]
        for contribution in contributions[1:]:
            acc = jtu.tree_map(lambda a, b: a + b, acc, contribution)
        return acc

    @property
    def kind(self) -> Kind:
        kinds = {term.kind for term in self.terms}
        if len(kinds) == 1:
            return kinds.pop()
        return MIXED


class Scaled(Term):
    """A term multiplied by a scalar coefficient.

    The coefficient is an ordinary (non-static) field, so it is a JAX
    leaf visible to ``jax.grad`` / ``jax.jit`` — coefficients can be
    optimised or swept. Scaling delegates its :attr:`kind` to the inner
    term.

    Args:
        term: The term to scale.
        coeff: Scalar multiplier (may be a Python float or a JAX scalar).
    """

    term: Term
    coeff: float | Array

    def __call__(self, t: float, state: PyTree, args: PyTree | None = None) -> PyTree:
        inner = self.term(t, state, args)
        return jtu.tree_map(lambda leaf: self.coeff * leaf, inner)

    @property
    def kind(self) -> Kind:
        return self.term.kind


class Compose(Term):
    """Operator composition: ``Compose(outer, inner)(state) = outer(inner(state))``.

    This is the multiplicative ("product") side of the algebra, in the
    operator-composition sense rather than a pointwise product —
    e.g. a biharmonic operator is ``Compose(laplacian, laplacian)``.

    The inner term's output is fed back in as the ``state`` argument of
    the outer term (both must map a field to a like-shaped field). ``t``
    and ``args`` are threaded through unchanged.

    Args:
        outer: Applied second (to the inner result).
        inner: Applied first (to the incoming state).
    """

    outer: Term
    inner: Term

    def __call__(self, t: float, state: PyTree, args: PyTree | None = None) -> PyTree:
        return self.outer(t, self.inner(t, state, args), args)

    @property
    def kind(self) -> Kind:
        # Composition is treated as a single atomic leaf for IMEX
        # partitioning; if either factor is implicit the whole composite
        # routes to the implicit stage.
        if self.outer.kind == IMPLICIT or self.inner.kind == IMPLICIT:
            return IMPLICIT
        return EXPLICIT


class _Zero(Term):
    """The additive identity term (returns a zeros-like tendency).

    Singleton; use the module-level :data:`_ZERO`. Adding it to any term
    is a no-op, which keeps :meth:`Sum.of` and ``sum(...)`` clean.
    """

    def __call__(self, t: float, state: PyTree, args: PyTree | None = None) -> PyTree:
        return jtu.tree_map(lambda leaf: leaf * 0.0, state)


_ZERO = _Zero()


# ----------------------------------------------------------------------
# IMEX tagging
# ----------------------------------------------------------------------


class _Kinded(Term):
    """Wrap a term to override its IMEX :attr:`kind` label.

    Produced by :func:`implicit` / :func:`explicit`. Evaluation is a
    transparent pass-through to the wrapped term.
    """

    term: Term
    _kind: Kind = eqx.field(static=True)

    def __call__(self, t: float, state: PyTree, args: PyTree | None = None) -> PyTree:
        return self.term(t, state, args)

    @property
    def kind(self) -> Kind:
        return self._kind


def implicit(term: Term) -> Term:
    """Tag ``term`` for the implicit stage of an IMEX integrator."""
    return _Kinded(term, IMPLICIT)


def explicit(term: Term) -> Term:
    """Tag ``term`` for the explicit stage of an IMEX integrator."""
    return _Kinded(term, EXPLICIT)


# ----------------------------------------------------------------------
# Partitioning + diffrax bridge
# ----------------------------------------------------------------------


def _iter_summands(term: Term) -> Iterator[Term]:
    """Yield the flat additive summands of ``term`` (recursing into Sum)."""
    if isinstance(term, Sum):
        for sub in term.terms:
            yield from _iter_summands(sub)
    else:
        yield term


def partition(term: Term) -> tuple[Term | None, Term | None]:
    """Split ``term`` into ``(explicit, implicit)`` sub-terms by kind.

    Flattens nested :class:`Sum` nodes and groups the summands by their
    :attr:`Term.kind`. A non-Sum term is treated as a single summand.

    Args:
        term: The (possibly composite) term to split.

    Returns:
        ``(explicit_part, implicit_part)``. Either element is ``None``
        when no summand of that kind is present.
    """
    explicit_terms: list[Term] = []
    implicit_terms: list[Term] = []
    for summand in _iter_summands(term):
        if summand.kind == IMPLICIT:
            implicit_terms.append(summand)
        else:
            explicit_terms.append(summand)

    explicit_part = Sum.of(*explicit_terms) if explicit_terms else None
    implicit_part = Sum.of(*implicit_terms) if implicit_terms else None
    return explicit_part, implicit_part


def _as_vector_field(
    term: Term,
    state_fn: Callable[[PyTree], PyTree] | None,
) -> Callable[[float, PyTree, PyTree | None], PyTree]:
    """Wrap a term as a diffrax-style ``f(t, y, args)`` vector field.

    When ``state_fn`` is given it is applied to ``y`` before the term is
    evaluated — used to enforce boundary conditions at every stage of
    every (explicit / implicit) sub-term.
    """
    if state_fn is None:

        def _vf(t, y, args=None):
            return term(t, y, args)
    else:

        def _vf(t, y, args=None):
            return term(t, state_fn(y), args)

    return _vf


def build_diffrax_terms(
    term: Term,
    *,
    state_fn: Callable[[PyTree], PyTree] | None = None,
) -> dfx.AbstractTerm:
    """Build the diffrax term object for ``term``, IMEX-aware.

    - If every summand shares a single kind, returns one
      :class:`diffrax.ODETerm`.
    - If the term mixes explicit and implicit summands, returns a
      :class:`diffrax.MultiTerm` whose first element is the explicit
      ``ODETerm`` and whose second is the implicit ``ODETerm`` — the
      ordering expected by diffrax IMEX solvers
      (``KenCarp3``/``KenCarp4``/``Sil3``/``KenCarp5``).

    Args:
        term: The assembled right-hand-side term.
        state_fn: Optional state preprocessor applied to ``y`` before
            each sub-term evaluation (e.g. boundary-condition
            enforcement). Applied uniformly to the explicit and implicit
            parts so both integrator stages see a BC-consistent state.

    Returns:
        A diffrax term suitable for ``diffrax.diffeqsolve``.

    Raises:
        ValueError: If ``term`` has no summands (e.g. the zero term).
    """
    explicit_part, implicit_part = partition(term)

    if explicit_part is not None and implicit_part is not None:
        return dfx.MultiTerm(
            dfx.ODETerm(_as_vector_field(explicit_part, state_fn)),
            dfx.ODETerm(_as_vector_field(implicit_part, state_fn)),
        )
    single = explicit_part if explicit_part is not None else implicit_part
    if single is None:
        raise ValueError("cannot build diffrax terms from an empty term")
    return dfx.ODETerm(_as_vector_field(single, state_fn))


__all__ = [
    "EXPLICIT",
    "IMPLICIT",
    "MIXED",
    "Compose",
    "Kind",
    "Scaled",
    "Sum",
    "Term",
    "TermFn",
    "build_diffrax_terms",
    "explicit",
    "implicit",
    "partition",
]
