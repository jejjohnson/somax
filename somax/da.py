"""Public surface for somax's data-assimilation adapters.

Wires somax forward models into the ``jejjohnson`` DA stack — a thin adapter
layer so any somax model can be driven by an external DA solver.

Ensemble filtering with `filterax` (Phase 4a):

- :class:`SomaxDynamics` adapts a somax model's pytree ``step`` to filterax's
  flat-vector ``AbstractDynamics``.
- :class:`SubsampleObs` is a sparse observation operator on the flat state.

Variational assimilation with `vardax` (Phase 4b):

- :class:`SomaxForwardModel` adapts a somax model to vardax's flat-vector
  ``pipekit_cycle.ForwardModel`` (``dt`` + ``step``), consumed by
  ``StrongFourDVar`` / ``IncrementalFourDVar`` / ``VarDACycle``. Intended for
  autonomous models (the ``ForwardModel`` rollout carries no per-substep time;
  see :class:`~somax._src.da.vardax_bridge.SomaxForwardModel`). Structured
  background / observation covariances come from `gaussx` operators (e.g.
  ``LowRankUpdate``), which are ``lineax``-compatible.

Shared bridge:

- :func:`state_to_vector` / :func:`make_ensemble` convert somax pytree states
  to the flat vectors / ensembles both stacks expect.

Importing this module requires the optional ``da`` dependency group
(``uv sync --group da``), which provides ``filterax`` and ``vardax`` (and,
transitively, ``gaussx``). It is kept out of ``somax``'s top-level
``__init__`` so plain ``import somax`` never needs the DA stack.
"""

from __future__ import annotations

from somax._src.da import (
    SomaxDynamics,
    SomaxForwardModel,
    SubsampleObs,
    make_ensemble,
    state_to_vector,
)


__all__ = [
    "SomaxDynamics",
    "SomaxForwardModel",
    "SubsampleObs",
    "make_ensemble",
    "state_to_vector",
]
