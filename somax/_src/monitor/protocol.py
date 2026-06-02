"""Lifecycle protocol for in-process simulation monitors.

A :class:`Monitor` observes a ``somax-sim`` run at three lifecycle points —
run start, each diagnostic-chunk boundary, and run end — and may *request
termination*. It is the somax-side analog of an Oceananigans ``Callback`` /
``Diagnostic`` or a Keras callback, but JAX-aware: monitors run host-side
*between* JIT-compiled integration chunks, so they may hold ordinary Python
state and do arbitrary host work (logging, I/O, history).

This decouples DURING-simulation observability from the runner: the
non-finite abort and energy-growth warning that used to be inline ``if``
blocks in the chunk step become pluggable monitors (see
:mod:`somax._src.monitor.builtins`), and users can add their own without
editing the runner.

Severity is declared, not implicit: a monitor either returns a
:class:`MonitorVerdict` with ``terminate=True`` (FAIL-HARD — downstream state
is meaningless) or simply records metrics/messages (MONITOR).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Protocol, runtime_checkable


@dataclasses.dataclass(frozen=True)
class ChunkInfo:
    """Context handed to a monitor at a diagnostic-chunk boundary.

    Args:
        index: Diagnostic-chunk index just completed (0-based).
        n_chunks: Total number of diagnostic chunks in the run.
        t0: Simulation time at the chunk start (seconds).
        t1: Simulation time at the chunk end (seconds).
        wall_seconds: Wallclock seconds spent integrating this chunk.
        is_snapshot: Whether the chunk endpoint is a snapshot save boundary.
        stats: Diffrax solver stats for this chunk (e.g.
            ``num_accepted_steps`` / ``num_rejected_steps`` / ``result``), or
            an empty dict if the solver did not expose them. Consumed by
            :class:`somax.monitor.SolverHealthMonitor`.
    """

    index: int
    n_chunks: int
    t0: float
    t1: float
    wall_seconds: float
    is_snapshot: bool
    stats: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class MonitorVerdict:
    """A monitor's response to a chunk. Inert by default.

    Args:
        metrics: Scalar metrics to merge into the run's per-chunk diagnostics
            (name -> value). Empty by default.
        messages: Lines to emit to ``run.log`` (prefixed with the monitor
            name). Empty by default.
        terminate: Whether the monitor requests a clean stop of the run.
        reason: Why termination was requested. Required (non-``None``) when
            ``terminate`` is ``True``; ignored otherwise.
    """

    metrics: dict[str, float] = dataclasses.field(default_factory=dict)
    messages: tuple[str, ...] = ()
    terminate: bool = False
    reason: str | None = None


@runtime_checkable
class Monitor(Protocol):
    """Structural protocol for a simulation monitor.

    Implementations need a ``name`` and the three lifecycle hooks. Most
    monitors only care about one hook, so :class:`somax.monitor.BaseMonitor`
    supplies inert defaults — subclass it and override what you need rather
    than implementing this Protocol directly.
    """

    name: str

    def on_run_start(self, model: Any, state0: Any, *, spec: Any) -> None:
        """Called once before integration begins."""
        ...

    def on_chunk_end(self, model: Any, state: Any, info: ChunkInfo) -> MonitorVerdict:
        """Called after each diagnostic chunk; may request termination."""
        ...

    def on_run_end(self, model: Any, final_state: Any, *, ok: bool) -> None:
        """Called once after integration ends (``ok`` False if it failed)."""
        ...
