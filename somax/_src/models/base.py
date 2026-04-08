"""DEPRECATED: Use ``somax.core.SomaxModel`` instead.

This module will be removed in a future release. All new models should
inherit from ``SomaxModel`` and implement ``vector_field`` and
``apply_boundary_conditions``.
"""

from __future__ import annotations

import equinox as eqx
from jaxtyping import PyTree


class DynamicalSystem(eqx.Module):
    """Legacy base class for dynamical systems.

    .. deprecated::
        Use :class:`somax.core.SomaxModel` instead.
    """

    def init_u0(self, domain: PyTree):
        raise NotImplementedError()

    def boundary(self, state: PyTree):
        raise NotImplementedError()

    def equation_of_motion(
        self, t: float, state: PyTree, args: PyTree | None = None
    ) -> PyTree:
        raise NotImplementedError()
