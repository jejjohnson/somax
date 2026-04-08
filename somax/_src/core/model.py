"""Base model contract for all somax models."""

from __future__ import annotations

import abc

import diffrax as dfx
import equinox as eqx
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

        Override for SDE (``MultiTerm``) or IMEX splitting.
        """
        return dfx.ODETerm(self.vector_field)

    def integrate(
        self,
        state0: PyTree,
        t0: float,
        t1: float,
        dt: float,
        **kw,
    ) -> dfx.Solution:
        """Forward integration using diffrax.

        Args:
            state0: Initial state.
            t0: Start time.
            t1: End time.
            dt: Initial time step.
            **kw: Passed to ``diffrax.diffeqsolve``. Supports
                ``solver``, ``saveat``, ``stepsize_controller``.
        """
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

    def diagnose(self, state: PyTree) -> PyTree:
        """Compute on-demand diagnostics from state.

        Override to return a ``Diagnostics`` instance.
        """
        return {}
