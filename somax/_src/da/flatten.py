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
    transform: Any | None = None,
) -> tuple[Float[Array, " N_x"], Callable[[Float[Array, " N_x"]], Any]]:
    """Flatten a state pytree to a 1-D vector plus its inverse.

    Args:
        state: A somax state pytree (e.g. an ``L96State`` or SWM state).
        transform: Optional :class:`~somax.StateAffine`. When given, the
            state is mapped into transformed coordinates before
            flattening and ``unravel`` maps back, so the flat vector a
            DA filter sees has comparable magnitudes across fields.
            Without it, ``h ~ 1e3 m`` and ``u ~ 1e-1 m/s`` share one
            vector and any scalar covariance over it is badly scaled.

    Returns:
        ``(vector, unravel)`` where ``vector`` is the concatenated 1-D state
        and ``unravel(vector)`` reconstructs the original pytree. ``unravel``
        is pure and safe to call under ``jit`` / ``vmap``.
    """
    if transform is None:
        return ravel_pytree(state)

    vector, unravel_transformed = ravel_pytree(transform.forward(state))

    def unravel(vec: Float[Array, " N_x"]) -> Any:
        return transform.inverse(unravel_transformed(vec))

    return vector, unravel


def make_ensemble(
    state: Any,
    key: PRNGKeyArray,
    *,
    size: int,
    std: float,
    transform: Any | None = None,
) -> Float[Array, "N_e N_x"]:
    """Build a Gaussian-perturbed flat ensemble around a base state.

    The canonical way to seed an ensemble filter for a twin experiment: take a
    (perturbed) background state and scatter ``size`` members around it.

    Args:
        state: Base state pytree; its flattened layout defines ``N_x``.
        key: PRNG key for the perturbations.
        size: Number of ensemble members ``N_e``.
        std: Standard deviation of the i.i.d. Gaussian perturbations.
        transform: Optional :class:`~somax.StateAffine`. The
            perturbation is applied in transformed coordinates, so one
            ``std`` means the same thing for every field: in physical
            space the members are spread by ``std * scale`` per field,
            giving a covariance of ``std**2 diag(scale**2)`` instead of
            an isotropic one that is meaningless across mixed units.
            The returned ensemble is in transformed coordinates, matching
            :func:`state_to_vector` called with the same transform.

    Returns:
        Flat ensemble of shape ``(size, N_x)`` suitable for a filterax
        filter's ``assimilate(init_ensemble, ...)``.
    """
    vec, _ = state_to_vector(state, transform)
    noise = std * jax.random.normal(key, (size, vec.size))
    return vec[None, :] + noise
