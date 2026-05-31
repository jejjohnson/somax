"""vardax forward-model adapter for somax models.

vardax's variational methods (``StrongFourDVar``, ``IncrementalFourDVar``,
``VarDACycle``, …) consume a ``pipekit_cycle.ForwardModel``: an object
exposing ``dt`` and ``step(state, dt) -> state``, rolled out over the
assimilation window with ``jax.lax.scan`` on a **flat** ``(N,)`` state
vector. somax models instead advance a *pytree* state. :class:`SomaxForwardModel`
bridges the two, mirroring :class:`somax.da.SomaxDynamics` (the filterax
ensemble analogue) but exposing the ``ForwardModel`` surface vardax expects.

``somax.operators.SomaxModelOp`` already satisfies ``ForwardModel`` — but on
*pytree* states, for ``pipekit_cycle.Cycle``. The variational solvers need a
*flat-vector* stepper (they ``lax.scan`` over ``(N,)`` control vectors), so
this adapter ravels/unravels around the model step via a state ``template``.

Importing this module requires the ``da`` dependency group (``uv sync
--group da``), which provides ``vardax``.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Float


class SomaxForwardModel(eqx.Module):
    r"""Adapt a somax model to vardax's flat-vector ``ForwardModel``.

    Satisfies the ``pipekit_cycle.ForwardModel`` protocol consumed by
    vardax's variational methods: ``dt`` (the fixed rollout step) and
    ``step(state, dt)`` on a flat ``(N_x,)`` vector. Each step unravels the
    flat state to the model's pytree state, advances one window with
    ``model.step(state, dt, t0=...)``, then re-ravels.

    The ``template`` is any representative state pytree (typically the initial
    condition); it fixes the flat <-> pytree layout. ``t0`` is the absolute
    start time threaded into ``model.step`` (autonomous models ignore it).

    Attributes:
        model: A somax model exposing ``step(state, dt, *, t0=...) -> state``.
        template: A representative state pytree defining the ravel layout.
        dt: Fixed integration window read by the variational solver's rollout.
        t0: Absolute start time passed to ``model.step``. Defaults to ``0.0``.
    """

    model: Any
    template: Any
    dt: float = eqx.field(static=True)
    t0: float = eqx.field(static=True, default=0.0)

    def step(
        self,
        state: Float[Array, " N_x"],
        dt: float,
    ) -> Float[Array, " N_x"]:
        """Advance a single flat state by ``dt`` (``ForwardModel`` contract)."""
        _, unravel = ravel_pytree(self.template)
        x = unravel(state)
        x_next = self.model.step(x, dt, t0=self.t0)
        flat, _ = ravel_pytree(x_next)
        return flat

    @property
    def state_signature(self) -> None:
        """No named-dimension signature — somax states are bare pytrees.

        Part of the ``pipekit_cycle.ForwardModel`` contract (``step`` / ``dt``
        / ``state_signature``); ``None`` means no advertised shape/dtype
        signature, matching :class:`somax.operators.SomaxModelOp`.
        """
        return None
