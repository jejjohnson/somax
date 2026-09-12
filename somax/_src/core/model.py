"""Base model contract for all somax models."""

from __future__ import annotations

import abc
import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import diffrax as dfx
import equinox as eqx
import jax.tree_util as jtu
import paramax
from jaxtyping import PyTree


if TYPE_CHECKING:
    from somax._src.core.terms import Term


#: Contract methods that read model parameters, and so must see them
#: unwrapped. :meth:`SomaxModel.__init_subclass__` wraps whichever of
#: these a subclass defines.
_UNWRAPPED_METHODS = ("vector_field", "diagnose", "apply_boundary_conditions")


def _unwrapping(implementation: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a contract method so it runs against unwrapped parameters."""

    @functools.wraps(implementation)
    def wrapper(self, *args: Any, **kwargs: Any) -> Any:
        # Call the raw implementation on the unwrapped model rather
        # than the method, which would re-enter this wrapper.
        return implementation(paramax.unwrap(self), *args, **kwargs)

    wrapper._somax_unwrapping = True
    return wrapper


class SomaxModel(eqx.Module):
    """Abstract base class defining the somax model contract.

    All somax models follow this interface for interoperability with
    diffrax, ``jax.grad``, and downstream tools like fourdvarjax.

    Subclasses must implement:
        - ``vector_field``: the right-hand side of the ODE/PDE
        - ``apply_boundary_conditions``: boundary enforcement

    Constrained parameters (see :func:`somax.positive`) are unwrapped
    on the way into ``vector_field`` and ``diagnose``, so a constrained
    model behaves exactly like its plain equivalent however it is
    called — through ``integrate``, through ``build_terms``, or by
    evaluating the right-hand side directly, which is what the
    differentiable-model tooling does. Unwrapping happens per call
    rather than once up front so that gradients flow to the *stored*
    unconstrained values, and it is a no-op for a model whose
    parameters are plain arrays.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for name in _UNWRAPPED_METHODS:
            implementation = cls.__dict__.get(name)
            if implementation is None or getattr(
                implementation, "_somax_unwrapping", False
            ):
                continue
            setattr(cls, name, _unwrapping(implementation))

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

        Constrained parameters are reconstituted per evaluation — see
        the class docstring.
        """

        def _rhs(t, state, args=None):
            model = paramax.unwrap(self)
            state = model.apply_boundary_conditions(state)
            return model.vector_field(t, state, args)

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

        Overrides receive an already-unwrapped model, because callers
        reach this through :meth:`diagnostics`. Calling ``diagnose``
        directly on a model with constrained parameters is a mistake;
        it raises rather than returning a wrong number, since arithmetic
        on a ``paramax`` wrapper is a type error.
        """
        return {}

    def diagnostics(self, state: PyTree) -> PyTree:
        """Diagnostics with constrained parameters reconstituted.

        The entry point to prefer over :meth:`diagnose`: it applies
        ``paramax.unwrap`` first, so a model carrying wrapped
        parameters reports the same diagnostics as the equivalent model
        carrying plain arrays. A no-op for unwrapped models.

        Somax's own monitors and runners spell this out as
        ``paramax.unwrap(model).diagnose(state)`` instead, because they
        accept any object exposing ``diagnose`` rather than requiring a
        ``SomaxModel``.
        """
        return paramax.unwrap(self).diagnose(state)

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

        model = paramax.unwrap(self)
        return build_diffrax_terms(
            model.terms, state_fn=model.apply_boundary_conditions
        )
