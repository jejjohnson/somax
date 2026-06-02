"""Public surface for somax's in-process simulation monitors.

A :class:`Monitor` observes a ``somax-sim`` run at three lifecycle points —
run start, each diagnostic-chunk boundary, and run end — and may request a
clean termination. This is the architectural unlock for DURING-simulation
observability: the runner's non-finite abort and energy-growth warning are
now pluggable :class:`NonFiniteMonitor` / :class:`EnergyGrowthMonitor`
monitors, and users add their own (conservation drift, solver health,
throughput, wallclock watchdog, or a custom :class:`BaseMonitor` subclass)
without editing the runner.

Monitors run host-side *between* JIT-compiled integration chunks, so they may
hold ordinary Python state. Pair them with :mod:`somax.guards` (in-JIT
fail-fast tripwires) for halts that must happen *at* the offending step.

Example::

    from somax.monitor import BaseMonitor, MonitorVerdict, default_monitors
    from somax._src.cli._run import simulate

    class MaxSpeedMonitor(BaseMonitor):
        name = "max_speed"
        def __init__(self, ceiling=50.0):
            self.ceiling = ceiling
        def on_chunk_end(self, model, state, info):
            umax = float(jnp.nanmax(jnp.abs(state.u)))
            if umax > self.ceiling:
                return MonitorVerdict(metrics={"u_max": umax}, terminate=True,
                                      reason=f"|u|={umax:.1f} > {self.ceiling}")
            return MonitorVerdict(metrics={"u_max": umax})

    simulate(spec, "runs/dg",
             monitors=[*default_monitors(), MaxSpeedMonitor(ceiling=20.0)])

Kept out of ``somax``'s top-level ``__init__`` to mirror the
module-per-surface layout (cf. :mod:`somax.eval`, :mod:`somax.guards`).
"""

from somax._src.monitor import (
    BaseMonitor,
    ChunkInfo,
    ConservationDriftMonitor,
    EnergyGrowthMonitor,
    Monitor,
    MonitorVerdict,
    NonFiniteMonitor,
    SolverHealthMonitor,
    ThroughputMonitor,
    WatchdogMonitor,
    default_monitors,
)


__all__ = [
    "BaseMonitor",
    "ChunkInfo",
    "ConservationDriftMonitor",
    "EnergyGrowthMonitor",
    "Monitor",
    "MonitorVerdict",
    "NonFiniteMonitor",
    "SolverHealthMonitor",
    "ThroughputMonitor",
    "WatchdogMonitor",
    "default_monitors",
]
