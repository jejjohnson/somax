"""Base model contract for all somax models."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import diffrax as dfx
import equinox as eqx
import jax.tree_util as jtu
from jaxtyping import PyTree


if TYPE_CHECKING:
    from somax._src.core.terms import Term


class SomaxModel(eqx.Module):
    """Abstract base class defining the somax model contract.

    All somax models follow this interface for interoperability with
    diffrax, ``jax.grad``, and downstream tools like fourdvarjax.

    Subclasses must implement:
        - ``vector_field``: the right-hand side of the ODE/PDE
        - ``apply_boundary_conditions``: boundary enforcement
    """

    @abc.abstractmethod
    def vector_field(
        self, t: float, state: PyTree, args: PyTree | None = None
    ) -> PyTree:
        """Compute the right-hand side (tendency) of the ODE/PDE.

        This method is diffrax-compatible and can be wrapped in an
        ``ODETerm`` for time integration.
        """
        ...

    @abc.abstractmethod
    def apply_boundary_conditions(self, state: PyTree) -> PyTree:
        """Apply boundary conditions to the state."""
        ...

    def build_terms(self) -> dfx.AbstractTerm:
        """Build diffrax term(s) for integration.

        The default wraps ``vector_field`` in an ``ODETerm``, applying
        boundary conditions to the state before each RHS evaluation.
        Override for SDE (``MultiTerm``) or IMEX splitting.
        """

        def _rhs(t, state, args=None):
            state = self.apply_boundary_conditions(state)
            return self.vector_field(t, state, args)

        return dfx.ODETerm(_rhs)

    def integrate(
        self,
        state0: PyTree,
        t0: float,
        t1: float,
        dt: float,
        **kw,
    ) -> dfx.Solution:
        """Forward integration using diffrax.

        Boundary conditions are applied to ``state0`` before starting
        and enforced at every RHS evaluation via ``build_terms``.

        Args:
            state0: Initial state.
            t0: Start time.
            t1: End time.
            dt: Initial time step.
            **kw: Passed to ``diffrax.diffeqsolve``. Supports
                ``solver``, ``saveat``, ``stepsize_controller``.
        """
        state0 = self.apply_boundary_conditions(state0)
        solver = kw.pop("solver", dfx.Tsit5())
        saveat = kw.pop("saveat", dfx.SaveAt(t1=True))
        stepsize_controller = kw.pop("stepsize_controller", dfx.ConstantStepSize())
        return dfx.diffeqsolve(
            terms=self.build_terms(),
            solver=solver,
            t0=t0,
            t1=t1,
            dt0=dt,
            y0=state0,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            **kw,
        )

    def step(
        self,
        state: PyTree,
        dt: float,
        *,
        t0: float = 0.0,
        **kw,
    ) -> PyTree:
        """Advance ``state`` by one increment ``dt`` and return the new state.

        This is the stepping primitive of the
        :class:`pipekit_cycle.ForwardModel` contract. A model supplies
        ``step`` and :attr:`state_signature` directly; the third protocol
        member — a default ``dt`` — is an adapter concern (a bare model
        has no inherent step size), so full ``ForwardModel`` conformance
        is provided by the :mod:`somax.operators` Operator wrapper, which
        carries ``dt`` alongside the model. None of this imports pipekit:
        the model is usable by ``pipekit_cycle.Cycle`` and by
        data-assimilation libraries (vardax, filterax) purely
        structurally.

        Unlike :meth:`integrate`, which returns a diffrax ``Solution``
        carrying a leading time axis, ``step`` returns a bare state
        pytree with the same structure as ``state``.

        Args:
            state: Current state pytree.
            dt: Increment to advance by. Used as both the integration
                window (``t0 -> t0 + dt``) and the initial solver step.
            t0: Absolute start time of the step. Autonomous models can
                ignore it; models with time-dependent forcing should
                thread it through. Defaults to ``0.0``.
            **kw: Forwarded to :meth:`integrate` (e.g. ``solver``,
                ``stepsize_controller``).

        Returns:
            The state advanced to ``t0 + dt``.
        """
        sol = self.integrate(
            state,
            t0=t0,
            t1=t0 + dt,
            dt=dt,
            saveat=dfx.SaveAt(t1=True),
            **kw,
        )
        # ``SaveAt(t1=True)`` stacks a length-1 leading time axis onto
        # every leaf; drop it to recover a bare state pytree.
        return jtu.tree_map(lambda leaf: leaf[-1], sol.ys)

    def diagnose(self, state: PyTree) -> PyTree:
        """Compute on-demand diagnostics from state.

        Override to return a ``Diagnostics`` instance.
        """
        return {}

    @property
    def state_signature(self) -> None:
        """No named-dimension signature — somax states are bare pytrees.

        Part of the :class:`pipekit_cycle.ForwardModel` contract (``step``,
        ``dt``, ``state_signature``). ``None`` means the model does not
        advertise a shape/dtype signature; structural, no pipekit import.
        Override to return a ``pipekit.Signature`` if a model tracks named
        dimensions.
        """
        return None


class TermModel(SomaxModel):
    """A :class:`SomaxModel` whose RHS is an assembled :class:`Term` tree.

    Instead of hand-writing ``vector_field``, a ``TermModel`` carries a
    composed term (a :class:`~somax._src.core.terms.Sum` of physics
    contributions). ``vector_field`` evaluates the tree, and
    ``build_terms`` delegates to
    :func:`~somax._src.core.terms.build_diffrax_terms`, which returns a
    :class:`diffrax.MultiTerm` when the tree mixes explicit and implicit
    summands — so an IMEX solver can route each physics term through the
    appropriate stage.

    Subclasses still own ``apply_boundary_conditions``. The base
    implementation here is a pass-through; override it to enforce BCs.

    Args:
        terms: The assembled right-hand-side term tree.

    Example:
        >>> rhs = AdvectionTerm(grid) + diffusivity * DiffusionTerm(grid)
        >>> model = MyTermModel(terms=rhs)
        >>> sol = model.integrate(state0, t0=0.0, t1=1.0, dt=0.01)
    """

    terms: Term

    def vector_field(
        self, t: float, state: PyTree, args: PyTree | None = None
    ) -> PyTree:
        """Evaluate the assembled term tree at ``(t, state, args)``."""
        return self.terms(t, state, args)

    def apply_boundary_conditions(self, state: PyTree) -> PyTree:
        """Pass-through by default; override to enforce boundary conditions."""
        return state

    def build_terms(self) -> dfx.AbstractTerm:
        """Build IMEX-aware diffrax terms from the term tree.

        Boundary conditions are applied before each term-tree evaluation,
        mirroring :meth:`SomaxModel.build_terms`. The explicit / implicit
        split (when present) is preserved so an IMEX solver integrates
        each part with the appropriate stage.
        """
        from somax._src.core.terms import build_diffrax_terms

        return build_diffrax_terms(self.terms, state_fn=self.apply_boundary_conditions)
