"""Pytree <-> flat-vector bridge for data assimilation.

filterax (and the variational control vector in vardax) operate on flat
``(N_x,)`` state vectors, whereas somax states are equinox pytrees. This
module centralises the conversion via :func:`jax.flatten_util.ravel_pytree`
so every DA adapter shares one definition of the state layout.

It is pure JAX (no DA dependency), so it is safe to import without the ``da``
dependency group; the filterax-specific adapters live alongside it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Float, PRNGKeyArray


def state_to_vector(
    state: Any,
) -> tuple[Float[Array, " N_x"], Callable[[Float[Array, " N_x"]], Any]]:
    """Flatten a state pytree to a 1-D vector plus its inverse.

    Args:
        state: A somax state pytree (e.g. an ``L96State`` or SWM state).

    Returns:
        ``(vector, unravel)`` where ``vector`` is the concatenated 1-D state
        and ``unravel(vector)`` reconstructs the original pytree. ``unravel``
        is pure and safe to call under ``jit`` / ``vmap``.
    """
    return ravel_pytree(state)


def make_ensemble(
    state: Any,
    key: PRNGKeyArray,
    *,
    size: int,
    std: float,
) -> Float[Array, "N_e N_x"]:
    """Build a Gaussian-perturbed flat ensemble around a base state.

    The canonical way to seed an ensemble filter for a twin experiment: take a
    (perturbed) background state and scatter ``size`` members around it.

    Args:
        state: Base state pytree; its flattened layout defines ``N_x``.
        key: PRNG key for the perturbations.
        size: Number of ensemble members ``N_e``.
        std: Standard deviation of the i.i.d. Gaussian perturbations.

    Returns:
        Flat ensemble of shape ``(size, N_x)`` suitable for a filterax
        filter's ``assimilate(init_ensemble, ...)``.
    """
    vec, _ = ravel_pytree(state)
    noise = std * jax.random.normal(key, (size, vec.size))
    return vec[None, :] + noise
