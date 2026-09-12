"""Bridge a fixed physical scaling into a flowjax normalising flow.

Optional: requires the ``flows`` extra (``pip install somax[flows]``).

:class:`~somax._src.core.transforms.StateAffine` already carries the
``transform_and_log_det`` / ``inverse_and_log_det`` pair flowjax
expects, but it acts on a ``State`` pytree while flowjax bijections act
on flat arrays of a declared ``shape``. :func:`to_flowjax` closes that
gap, so composing a learned flow onto a fixed physical scaling is a
one-liner:

    >>> bijection = Chain([flow, Invert(to_flowjax(transform, state))])
    >>> prior = Transformed(base_distribution, bijection)

flowjax is deliberately not a core dependency. Its own ``Affine``
forces the scale through a trainable softplus parameterisation, which
is the opposite of what a fixed physical scale wants, and the library
brings the whole flow stack (masked autoregressive layers, splines, a
training loop) along with it. The log-determinant only matters when
transforming densities; everywhere else in somax the affine map is
used directly.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import paramax
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, PyTree

from somax._src.core.transforms import StateAffine


try:  # pragma: no cover - exercised by the import-guard test
    from flowjax.bijections import AbstractBijection
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "somax.flows requires flowjax, which is an optional dependency. "
        "Install it with `pip install somax[flows]` (or "
        "`uv sync --extra flows`)."
    ) from exc


class StateAffineBijection(AbstractBijection):
    """A :class:`StateAffine` as a flat flowjax bijection.

    Wraps the pytree transform in the flat-array interface flowjax
    bijections use, flattening with the same ``ravel_pytree`` layout as
    :func:`somax._src.da.flatten.state_to_vector`. That shared layout is
    what lets one vector feed both a DA filter and a flow.

    The map is fixed, not trainable: ``loc`` and ``scale`` come from
    physical scales or sample statistics, and a flow composed on top
    learns the departure from that scaling rather than relearning the
    scaling itself. That is enforced rather than merely intended — the
    transform is held in a ``paramax.NonTrainable``, so when ``loc``
    and ``scale`` are arrays (which they are for
    :meth:`StateAffine.from_samples`, and for any array-valued field
    override) flowjax's training path cannot drift the normalisation
    along with the flow.

    Attributes:
        state_transform: The underlying pytree affine map. Named so
            rather than ``transform`` because ``AbstractBijection``
            already defines a ``transform()`` method, and a field of
            that name would shadow it.
        shape: Flat shape, ``(n_elements,)``.
        cond_shape: Always ``None`` — the map is unconditional.
    """

    state_transform: StateAffine
    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None = None
    unravel: Any = None

    def __init__(self, transform: StateAffine, state_example: PyTree) -> None:
        """Build from a transform and an example state.

        Args:
            transform: The affine map to wrap.
            state_example: A state with the structure and shapes the
                bijection will act on. Only its layout is used.
        """
        flat, unravel = ravel_pytree(state_example)
        self.state_transform = paramax.non_trainable(transform)
        self.shape = (flat.size,)
        self.cond_shape = None
        self.unravel = unravel

    @property
    def _map(self) -> StateAffine:
        """The affine map, with its non-trainable wrapper removed."""
        return paramax.unwrap(self.state_transform)

    def transform_and_log_det(
        self, x: Array, condition: Array | None = None
    ) -> tuple[Array, Array]:
        """Forward map on a flat vector, with the constant log-determinant."""
        del condition
        state = self.unravel(x)
        transformed, log_det = self._map.transform_and_log_det(state)
        flat, _ = ravel_pytree(transformed)
        return flat, jnp.asarray(log_det)

    def inverse_and_log_det(
        self, y: Array, condition: Array | None = None
    ) -> tuple[Array, Array]:
        """Inverse map on a flat vector, with the negated log-determinant."""
        del condition
        state = self.unravel(y)
        inverted, log_det = self._map.inverse_and_log_det(state)
        flat, _ = ravel_pytree(inverted)
        return flat, jnp.asarray(log_det)


def to_flowjax(transform: StateAffine, state_example: PyTree) -> StateAffineBijection:
    """Adapt a :class:`StateAffine` to the flowjax bijection interface.

    Args:
        transform: The pytree affine map to wrap.
        state_example: A state carrying the structure and shapes the
            bijection will act on.

    Returns:
        A flowjax-compatible bijection over flat vectors, using the same
        layout as :func:`somax._src.da.flatten.state_to_vector`.
    """
    return StateAffineBijection(transform, state_example)


def learned_prior(
    transform: StateAffine,
    state_example: PyTree,
    flow: Any,
    base: Any,
) -> Any:
    """Compose a fixed physical scaling with a learned flow into a prior.

    The flow models the *standardised* state, and the scaling maps its
    output back to physical units, so the flow learns only the
    departure from the physical scaling rather than relearning the
    scaling itself. The resulting log-density accounts for both, the
    affine part contributing the constant ``-sum(n log scale)``.

    A flowjax ``Transformed`` bijection runs base-to-data and a
    ``Chain`` applies its first element first, so the affine goes last
    and inverted: standardised coordinates are what the flow produces,
    and physical coordinates are what the prior is over.

    Args:
        transform: The fixed affine scaling.
        state_example: A state giving the flat layout.
        flow: A flowjax bijection acting on the flat vector.
        base: A flowjax distribution over the flat vector.

    Returns:
        A ``flowjax.distributions.Transformed`` over flat state vectors.
    """
    from flowjax.bijections import Chain, Invert
    from flowjax.distributions import Transformed

    to_physical = Invert(to_flowjax(transform, state_example))
    return Transformed(base, Chain([flow, to_physical]))
