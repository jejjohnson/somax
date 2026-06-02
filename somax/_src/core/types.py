"""Core type definitions for somax models."""

from __future__ import annotations

import equinox as eqx
from jaxtyping import Array


class State(eqx.Module):
    """Base class for model state vectors.

    All model states should subclass this to enable interoperability
    with the somax model contract and JAX transformations.
    """


class Params(eqx.Module):
    """Base class for differentiable model parameters.

    Fields on Params subclasses are visible to ``jax.grad`` by default.
    Use ``eqx.field(static=True)`` for non-differentiable parameters.
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
