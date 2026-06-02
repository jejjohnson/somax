"""Public surface for somax's in-JIT fail-fast guards.

JIT/``vmap``/``grad``-safe tripwires (:func:`guard_finite`,
:func:`guard_positive`, :func:`guard_ceiling`) that return their input
unchanged but raise immediately — via :func:`equinox.error_if` — when a
physical invariant is violated inside a model's ``vector_field``. They halt
*at* the offending step rather than after a whole integration chunk.

Pure JAX / Equinox (a base dependency), so importing this is cheap; it is
kept out of ``somax``'s top-level ``__init__`` only to mirror the
module-per-surface layout (cf. :mod:`somax.eval`, :mod:`somax.operators`).
"""

from somax._src.guards import guard_ceiling, guard_finite, guard_positive


__all__ = [
    "guard_ceiling",
    "guard_finite",
    "guard_positive",
]
