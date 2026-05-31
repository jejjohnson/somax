"""Public surface for somax's data-assimilation adapters.

Wires somax forward models into the ``jejjohnson`` DA stack. This first piece
(Phase 4a) covers ensemble filtering with `filterax`:

- :class:`SomaxDynamics` adapts a somax model's pytree ``step`` to filterax's
  flat-vector ``AbstractDynamics``.
- :class:`SubsampleObs` is a sparse observation operator on the flat state.
- :func:`state_to_vector` / :func:`make_ensemble` bridge somax pytree states
  to the flat vectors / ensembles filterax expects.

Variational assimilation (vardax 4DVar) and structured `gaussx` covariances
arrive in Phase 4b.

Importing this module requires the optional ``da`` dependency group
(``uv sync --group da``), which provides ``filterax``. It is kept out of
``somax``'s top-level ``__init__`` so plain ``import somax`` never needs the
DA stack.
"""

from __future__ import annotations

from somax._src.da import (
    SomaxDynamics,
    SubsampleObs,
    make_ensemble,
    state_to_vector,
)


__all__ = [
    "SomaxDynamics",
    "SubsampleObs",
    "make_ensemble",
    "state_to_vector",
]
