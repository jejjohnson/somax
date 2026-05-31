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

Time dependence: the ``ForwardModel`` contract is ``step(state, dt)`` with no
per-substep time, and vardax's rollout (``jax.lax.scan`` with a state-only
carry) gives the adapter no way to know which substep it is on. Every substep
is therefore integrated from the same absolute start time :attr:`t0`. This is
exact for **autonomous** models (Lorenz-63/96 and the somax cores, which
ignore ``t0``), but a model with explicitly time-dependent forcing would have
its later substeps evaluated at the wrong absolute time. Use this adapter with
autonomous models; ``t0`` sets the (single) window start, not a per-substep
clock.

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
    condition); it fixes the flat <-> pytree layout.

    Intended for **autonomous** models. The same fixed :attr:`t0` is passed on
    every ``step`` call, so a multi-step vardax rollout integrates every
    substep from the same absolute time (the ``ForwardModel`` contract carries
    no per-substep clock — see the module docstring). Autonomous models ignore
    ``t0`` so this is exact; a model with explicit time-dependent forcing would
    see its later substeps evaluated at the wrong absolute time.

    Attributes:
        model: A somax model exposing ``step(state, dt, *, t0=...) -> state``.
        template: A representative state pytree defining the ravel layout.
        dt: Fixed integration window read by the variational solver's rollout.
        t0: Absolute start time of the window, passed to ``model.step`` on
            every substep (not advanced per substep). Defaults to ``0.0``.
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
        """Advance a single flat state by ``dt`` (``ForwardModel`` contract).

        ``t0`` is held at :attr:`t0` (the window start), not advanced per
        substep — see the class docstring on autonomous-model use.
        """
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
