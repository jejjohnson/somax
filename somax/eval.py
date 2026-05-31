"""Public surface for somax's reference-free evaluation metrics.

Field-level diagnostics computed on a model's own Cartesian grid — RMS
divergence, total enstrophy, kinetic energy, and geostrophic imbalance —
giving somax a quantitative evaluation surface for free-running simulations.
These need only the state itself (no ground-truth reference); reference-based
skill scores belong with the data-assimilation work in a later phase.

:func:`compute_eval_metrics` is the convenient entry point: it returns every
applicable metric for a ``(model, state)`` pair (and an empty dict for
non-fluid models), and is what the ``somax-sim`` runner folds into
``metrics.json``. The individual functions are exposed for ad-hoc analysis.

Pure JAX / finitevolx (both base dependencies), so importing this is cheap;
it is kept out of ``somax``'s top-level ``__init__`` only to mirror the
module-per-surface layout (cf. :mod:`somax.operators`).
"""

from __future__ import annotations

from somax._src.eval.metrics import (
    compute_eval_metrics,
    geostrophic_imbalance,
    kinetic_energy,
    rms_divergence,
    total_enstrophy,
)


__all__ = [
    "compute_eval_metrics",
    "geostrophic_imbalance",
    "kinetic_energy",
    "rms_divergence",
    "total_enstrophy",
]
