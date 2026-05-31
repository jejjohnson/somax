"""Observation operators for somax data assimilation.

These satisfy ``filterax.AbstractObsOperator`` — ``H(state) -> obs`` on a flat
state vector, vectorised over the ensemble by the filter. The variational
(pytree-state) observation operators for vardax are added in Phase 4b.

Importing this module requires the ``da`` dependency group.
"""

from __future__ import annotations

import jax.numpy as jnp
from filterax import AbstractObsOperator
from jaxtyping import Array, Float, Int


class SubsampleObs(AbstractObsOperator):
    r"""Linear sub-sampling observation operator ``H(x) = x[indices]``.

    Observes a fixed subset of the flat state vector — the canonical sparse
    observation operator for twin DA experiments (e.g. observe every other
    grid point). Linear, so its tangent-linear is itself.

    Attributes:
        indices: 1-D integer array of observed positions in the flat state;
            its length is the observation dimension ``N_y``.
    """

    indices: Int[Array, " N_y"]

    def __init__(self, indices: Int[Array, " N_y"]):
        self.indices = jnp.asarray(indices)

    def __call__(self, state: Float[Array, " N_x"]) -> Float[Array, " N_y"]:
        """Select the observed components of a single flat state."""
        return state[self.indices]
