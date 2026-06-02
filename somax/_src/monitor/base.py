"""Convenience base class for simulation monitors.

:class:`BaseMonitor` supplies inert defaults for all three lifecycle hooks so
a concrete monitor overrides only what it needs. It satisfies the
:class:`somax._src.monitor.protocol.Monitor` structural protocol.
"""

from __future__ import annotations

from typing import Any

from somax._src.monitor.protocol import ChunkInfo, MonitorVerdict


class BaseMonitor:
    """Inert base monitor — override only the hooks you care about.

    Subclasses typically set a class-level ``name`` and override
    :meth:`on_chunk_end`. The default hooks do nothing (and
    :meth:`on_chunk_end` returns an empty :class:`MonitorVerdict`), so a
    one-hook monitor stays minimal.
    """

    name: str = "monitor"

    def on_run_start(self, model: Any, state0: Any, *, spec: Any) -> None:
        """No-op by default."""

    def on_chunk_end(self, model: Any, state: Any, info: ChunkInfo) -> MonitorVerdict:
        """Inert by default — returns an empty verdict."""
        return MonitorVerdict()

    def on_run_end(self, model: Any, final_state: Any, *, ok: bool) -> None:
        """No-op by default."""
