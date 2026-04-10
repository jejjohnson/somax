"""Run-log plumbing for somax-sim, built on loguru.

Every somax-sim invocation writes a structured ``run.log`` to the
output directory. The file contains the per-chunk physical diagnostics
emitted by the chunked-integration path plus a periodic "alive" tick
so users tailing the file can confirm liveness during slow chunks.

Architecture
------------

:func:`start_run_log` does two things:

1. Adds a loguru file sink at ``<output_dir>/run.log`` with
   level=DEBUG and a filter that only accepts records tagged with
   ``extra={"run_log": True}``. The file contains *only* run-log
   lines, even though they share the same logger as the rest of the
   application.

2. Starts a small daemon thread that emits an ``"alive (...)"`` line
   every ``alive_interval`` seconds via the bound logger. The thread
   keeps the user informed even when a single chunk takes longer than
   the interval (large grids, slow JIT compile, etc.).

The bound logger returned by ``start_run_log`` is the canonical
emitter for chunk diagnostics. Callers should use it directly:

    rl = start_run_log(output_dir, label="somax-sim/run")
    try:
        rl.log.debug("integration starting ...")
        # ... run chunks, emit rl.log.debug(...) per chunk ...
    finally:
        stop_run_log(rl)

All run-log lines are at DEBUG level so they only appear on stderr
when the user passes ``--verbose``; otherwise stderr stays clean and
the file is the only place to see them.
"""

from __future__ import annotations

import contextlib
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger


# ----------------------------------------------------------------------
# Run-log record filter
# ----------------------------------------------------------------------


def _is_run_log_record(record: dict[str, Any]) -> bool:
    """Loguru sink filter: True if the record was bound with run_log=True."""
    return bool(record["extra"].get("run_log", False))


# Format used for the run-log file. Plain (no colors) so it's easy to
# tail and grep. The label comes from `bind(label=...)` and identifies
# which run / mode emitted the line.
RUN_LOG_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | {extra[label]: <16} | {message}"
)

# Internal default for the safety-net "alive" tick. Not user-tunable —
# 10s is short enough to confirm liveness during a slow chunk and long
# enough that the file doesn't grow without bound on multi-hour runs.
DEFAULT_ALIVE_INTERVAL_SECONDS = 10.0


# ----------------------------------------------------------------------
# Public state object
# ----------------------------------------------------------------------


@dataclass
class RunLogContext:
    """Bag of resources held while run-log writing is active.

    Attributes:
        log: Loguru logger pre-bound with ``run_log=True`` and the run
            label. Use ``log.debug(...)`` to emit a line.
        sink_id: Loguru sink id for the run-log file. Pass to
            ``logger.remove(sink_id)`` to detach the sink.
        alive_thread: Background "still alive" thread, or ``None`` if
            no periodic ticking was requested.
    """

    log: Any  # loguru.Logger — bound
    sink_id: int
    alive_thread: _AliveThread | None


class _AliveThread:
    """Daemon thread that emits a periodic 'alive' line via the bound logger.

    The thread is the safety net for the case where a single chunk runs
    longer than its expected duration — without it, the user would see
    the most recent chunk's line and then silence until the next chunk
    completes. With it, every ``interval`` seconds the file gets a fresh
    timestamped line so the user knows the process is still healthy.

    Args:
        bound_log: Loguru logger pre-bound with the run_log tag.
        interval: Seconds between alive ticks.
    """

    def __init__(self, bound_log: Any, *, interval: float) -> None:
        self.bound_log = bound_log
        self.interval = interval
        self._stop = threading.Event()
        self._t_start = time.perf_counter()
        self._thread = threading.Thread(
            target=self._run, name="somax-sim-alive", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(2.0 * self.interval, 5.0))

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            elapsed = time.perf_counter() - self._t_start
            self.bound_log.debug(f"alive (wallclock={elapsed:.1f}s)")


# ----------------------------------------------------------------------
# Lifecycle
# ----------------------------------------------------------------------


def start_run_log(
    output_dir: str | Path,
    *,
    label: str,
    alive_interval: float = DEFAULT_ALIVE_INTERVAL_SECONDS,
    enable_alive_thread: bool = True,
) -> RunLogContext:
    """Set up the run-log file sink + optional alive thread.

    Truncates any existing ``run.log`` at ``output_dir`` so a fresh
    run starts with a clean file.

    Args:
        output_dir: Run output directory. The run-log file is written
            to ``<output_dir>/run.log``.
        label: Tag prepended to every line; identifies the run / mode.
        alive_interval: Seconds between alive ticks. Must be > 0.
        enable_alive_thread: If False, skip the periodic alive thread
            (the file sink is still added — chunks still log to it).

    Returns:
        :class:`RunLogContext` to be passed to :func:`stop_run_log`.
    """
    if alive_interval <= 0:
        raise ValueError(f"alive_interval must be > 0 (got {alive_interval})")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sink_id = logger.add(
        out_dir / "run.log",
        level="DEBUG",
        format=RUN_LOG_FORMAT,
        filter=_is_run_log_record,
        mode="w",  # truncate per run
        enqueue=False,
    )
    bound = logger.bind(run_log=True, label=label)
    bound.debug("started")

    alive: _AliveThread | None = None
    if enable_alive_thread:
        alive = _AliveThread(bound, interval=alive_interval)
        alive.start()

    return RunLogContext(log=bound, sink_id=sink_id, alive_thread=alive)


def stop_run_log(ctx: RunLogContext, *, final_message: str | None = None) -> None:
    """Tear down the run-log sink + alive thread.

    Safe to call multiple times; subsequent calls are no-ops once the
    sink id is detached.

    Args:
        ctx: RunLogContext returned by :func:`start_run_log`.
        final_message: Optional summary line emitted before the sink is
            detached (e.g. ``"finished cleanly"`` or ``"FAILED: ..."``).
    """
    if ctx.alive_thread is not None:
        ctx.alive_thread.stop()
    if final_message is not None:
        ctx.log.debug(final_message)
    with contextlib.suppress(ValueError):
        # ValueError → sink already removed (idempotent stop).
        logger.remove(ctx.sink_id)
