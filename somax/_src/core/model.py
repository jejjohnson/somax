"""Base model contract for all somax models."""

from __future__ import annotations

import abc

import diffrax as dfx
import equinox as eqx
import jax.tree_util as jtu
from jaxtyping import PyTree


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

        This single-step interface is what makes every somax model
        structurally satisfy the :class:`pipekit_cycle.ForwardModel`
        protocol (``step(state, dt) -> state``). Satisfaction is purely
        structural — somax does not import pipekit — so a model can be
        driven by the ``pipekit_cycle.Cycle`` runner and supplied as the
        forward model to data-assimilation libraries (vardax, filterax)
        without any subclassing.

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
