"""Public surface for somax's pipekit ``Operator`` bridge.

Import this module (or install ``somax[sim]``) to expose somax forward
models as :class:`pipekit.Operator` s with serializable flat configs.
Kept out of ``somax``'s top-level ``__init__`` so ``import somax`` does
not require pipekit — only ``import somax.operators`` does.
"""

from __future__ import annotations

from somax._src.operators import Burgers2DOp, SomaxModelOp


__all__ = [
    "Burgers2DOp",
    "SomaxModelOp",
]
