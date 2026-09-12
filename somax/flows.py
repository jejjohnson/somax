"""Optional flowjax bridge — requires the ``flows`` extra.

``pip install somax[flows]`` (or ``uv sync --extra flows``) pulls in
flowjax; importing this module without it raises with that instruction
rather than a bare ``ModuleNotFoundError`` naming a package the user
never asked for.

See :mod:`somax._src.core.flows` for what the bridge does and why
flowjax is not a core dependency.
"""

from __future__ import annotations

from somax._src.core.flows import (
    StateAffineBijection,
    learned_prior,
    to_flowjax,
)


__all__ = [
    "StateAffineBijection",
    "learned_prior",
    "to_flowjax",
]
