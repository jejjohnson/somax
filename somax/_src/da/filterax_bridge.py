"""filterax dynamics adapter for somax forward models.

``filterax`` propagates a *flat* state vector with
``AbstractDynamics.__call__(state, t0, t1)`` (vectorised over the ensemble
with ``eqx.filter_vmap``), whereas a somax model advances a *pytree* state
with ``step(state, dt, *, t0=...)``. :class:`SomaxDynamics` bridges the two.

Importing this module requires the ``da`` dependency group (``uv sync
--group da``), which provides ``filterax``.
"""

from __future__ import annotations

from typing import Any

from filterax import AbstractDynamics
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Float


class SomaxDynamics(AbstractDynamics):
    r"""Adapt a somax forward model to filterax's flat-vector dynamics.

    Wraps a somax model so an ensemble filter can drive it: unravel the flat
    state to the model's pytree state, advance one window with
    ``model.step(state, dt=t1 - t0, t0=t0)``, then re-ravel to a flat vector.

    The ``template`` is any representative state pytree (typically the initial
    condition); it fixes the flat <-> pytree layout. It is stored rather than
    the ``unravel`` closure so the module stays a clean equinox pytree under
    ``jit`` / ``filter_vmap``.

    Attributes:
        model: A somax model exposing ``step(state, dt, *, t0=...) -> state``
            (a bare ``SomaxModel`` or any object with that method).
        template: A representative state pytree defining the ravel layout.
    """

    model: Any
    template: Any
    #: Optional :class:`~somax.StateAffine`. Set it when the flat
    #: vectors come from ``state_to_vector``/``make_ensemble`` with a
    #: transform: without it the adapter would hand the model
    #: transformed values as if they were physical ones, advancing a
    #: near-zero standardised thickness as a real layer depth and
    #: corrupting the first forecast.
    transform: Any = None

    def __call__(
        self,
        state: Float[Array, " N_x"],
        t0: Float[Array, ""],
        t1: Float[Array, ""],
    ) -> Float[Array, " N_x"]:
        """Advance a single flat state from ``t0`` to ``t1``.

        When :attr:`transform` is set the incoming vector is in
        transformed coordinates, so it is mapped back before the model
        sees it and forward again afterwards — the filter keeps working
        in the coordinates ``make_ensemble`` produced, while the model
        keeps working in its own.
        """
        _, unravel = ravel_pytree(self.template)
        x = unravel(state)
        if self.transform is not None:
            x = self.transform.inverse(x)
        x_next = self.model.step(x, t1 - t0, t0=t0)
        if self.transform is not None:
            x_next = self.transform.forward(x_next)
        flat, _ = ravel_pytree(x_next)
        return flat
