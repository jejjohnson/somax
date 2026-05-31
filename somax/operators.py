"""Public surface for somax's pipekit ``Operator`` bridge.

Import this module to expose somax forward models as
:class:`pipekit.Operator` s with serializable flat configs. pipekit is a
base dependency, so no extra install step is needed. Kept out of
``somax``'s top-level ``__init__`` so plain ``import somax`` doesn't
import pipekit (the core stays structural-only) — ``import
somax.operators`` pulls it in.
"""

from __future__ import annotations

from somax._src.operators import Burgers2DOp, SomaxModelOp


__all__ = [
    "Burgers2DOp",
    "SomaxModelOp",
]
