"""Built-in monitors shipped with the :mod:`somax.monitor` protocol.

These reproduce the observability that used to be inline in the runner's
chunk step (the non-finite abort and the energy-growth warning) as pluggable
monitors, and add new ones that consume the conservation invariants
(:meth:`somax.Diagnostics.invariants`) and the diffrax solver stats.

Severity per monitor:

============================ ==================
Monitor                      Severity
============================ ==================
``NonFiniteMonitor``         FAIL-HARD
``EnergyGrowthMonitor``      MONITOR (-> FAIL-HARD with ``hard_factor``)
``ConservationDriftMonitor`` MONITOR (-> FAIL-HARD with ``rtol_fail``)
``SolverHealthMonitor``      MONITOR
``ThroughputMonitor``        MONITOR
``WatchdogMonitor``          FAIL-HARD
============================ ==================
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from somax._src.monitor.base import BaseMonitor
from somax._src.monitor.protocol import ChunkInfo, MonitorVerdict


def _state_field_arrays(state: Any) -> dict[str, np.ndarray]:
    """Best-effort map of state field name -> numpy array (eqx.Module)."""
    out: dict[str, np.ndarray] = {}
    fields = getattr(state, "__dataclass_fields__", None)
    if not fields:
        return out
    for name in fields:
        try:
            value = getattr(state, name)
            arr = np.asarray(value)
        except (TypeError, ValueError):
            continue
        if np.issubdtype(arr.dtype, np.number):
            out[name] = arr
    return out


class NonFiniteMonitor(BaseMonitor):
    """FAIL-HARD: terminate the run if any state field holds NaN/Inf.

    The pluggable form of the runner's original non-finite abort. A
    non-finite state makes everything downstream meaningless, so this always
    requests termination.
    """

    name = "non_finite"

    def on_chunk_end(self, model: Any, state: Any, info: ChunkInfo) -> MonitorVerdict:
        bad = [
            name
            for name, arr in _state_field_arrays(state).items()
            if not np.all(np.isfinite(arr))
        ]
        if bad:
            return MonitorVerdict(
                terminate=True,
                reason=f"non-finite values in state field(s): {', '.join(sorted(bad))}",
            )
        return MonitorVerdict()


class EnergyGrowthMonitor(BaseMonitor):
    """MONITOR (optionally FAIL-HARD): flag large energy jumps between chunks.

    Warns when the run's energy-like scalar grows by more than ``factor``
    relative to the previous chunk — an early instability signal before a NaN
    appears. If ``hard_factor`` is set, growth beyond it requests termination.

    Args:
        factor: Warn when ``|E(t)| > factor * |E(t-1)|``. Defaults to 10.0.
        hard_factor: If not ``None``, terminate when growth exceeds this.
    """

    name = "energy_growth"
    _ENERGY_KEYS = (
        "total_energy",
        "energy",
        "total_kinetic_energy",
        "kinetic_energy",
    )

    def __init__(self, factor: float = 10.0, hard_factor: float | None = None):
        self.factor = factor
        self.hard_factor = hard_factor
        self._prev: float | None = None

    def _energy(self, model: Any, state: Any) -> float | None:
        # Source an energy-like scalar from the model's diagnostics. Prefer the
        # advertised invariants(), but fall back to any top-level scalar field
        # named like an energy on the Diagnostics object — this matches the
        # legacy inline warning, which read the flattened diagnostics, so the
        # early-warning still fires for models that do not override
        # invariants() (Lorenz, linear SWM, Navier-Stokes, Burgers, ...).
        try:
            diag = model.diagnose(state)
        except Exception:
            return None
        try:
            invariants = diag.invariants()
        except Exception:
            invariants = {}
        for key in self._ENERGY_KEYS:
            if key in invariants:
                value = float(np.asarray(invariants[key]))
                return value if np.isfinite(value) else None
        for key in self._ENERGY_KEYS:
            field = getattr(diag, key, None)
            if field is None:
                continue
            arr = np.asarray(field)
            if arr.ndim == 0:
                value = float(arr)
                return value if np.isfinite(value) else None
        return None

    def on_run_start(self, model: Any, state0: Any, *, spec: Any) -> None:
        self._prev = self._energy(model, state0)

    def on_chunk_end(self, model: Any, state: Any, info: ChunkInfo) -> MonitorVerdict:
        cur = self._energy(model, state)
        prev = self._prev
        if cur is not None:
            self._prev = cur
        if prev is None or cur is None or abs(prev) == 0.0:
            return MonitorVerdict()
        growth = abs(cur) / abs(prev)
        if growth <= self.factor:
            return MonitorVerdict()
        msg = f"energy grew {growth:.1f}x from previous chunk"
        if self.hard_factor is not None and growth > self.hard_factor:
            return MonitorVerdict(
                metrics={"energy_growth": growth},
                messages=(msg,),
                terminate=True,
                reason=f"{msg} (> hard_factor {self.hard_factor})",
            )
        return MonitorVerdict(metrics={"energy_growth": growth}, messages=(msg,))


class ConservationDriftMonitor(BaseMonitor):
    """MONITOR (optionally FAIL-HARD): track drift of conserved invariants.

    Records the relative drift ``|I(t) - I(0)| / |I(0)|`` for every invariant
    the model advertises via :meth:`somax.Diagnostics.invariants`. Warns above
    ``rtol_warn``; if ``rtol_fail`` is set, terminates when the worst drift
    exceeds it.

    Mass is conserved to machine precision (set a tight tolerance); energy /
    enstrophy / Casimirs drift under implicit numerical dissipation, so use a
    generous ``rtol_warn`` and read the metric as "quantify the dissipation",
    not "drive to zero".

    Args:
        rtol_warn: Relative-drift warning threshold. Defaults to 1e-2.
        rtol_fail: If not ``None``, terminate when the worst drift exceeds it.
    """

    name = "conservation"

    def __init__(self, rtol_warn: float = 1e-2, rtol_fail: float | None = None):
        self.rtol_warn = rtol_warn
        self.rtol_fail = rtol_fail
        self._ref: dict[str, float] = {}

    def _invariants(self, model: Any, state: Any) -> dict[str, float]:
        try:
            raw = model.diagnose(state).invariants()
        except Exception:
            return {}
        return {k: float(np.asarray(v)) for k, v in raw.items()}

    def on_run_start(self, model: Any, state0: Any, *, spec: Any) -> None:
        self._ref = self._invariants(model, state0)

    def on_chunk_end(self, model: Any, state: Any, info: ChunkInfo) -> MonitorVerdict:
        cur = self._invariants(model, state)
        metrics: dict[str, float] = {}
        messages: list[str] = []
        worst = 0.0
        for key, ref in self._ref.items():
            if abs(ref) < 1e-30 or key not in cur:
                continue
            drift = abs(cur[key] - ref) / abs(ref)
            metrics[f"drift_{key}"] = drift
            worst = max(worst, drift)
            if drift > self.rtol_warn:
                messages.append(f"{key} drifted {drift:.2%}")
        terminate = self.rtol_fail is not None and worst > self.rtol_fail
        return MonitorVerdict(
            metrics=metrics,
            messages=tuple(messages),
            terminate=terminate,
            reason=(f"conservation drift {worst:.2%}" if terminate else None),
        )


class SolverHealthMonitor(BaseMonitor):
    """MONITOR: surface diffrax per-chunk solver statistics.

    Consumes the solver stats the runner now threads onto :class:`ChunkInfo`
    (via a per-chunk ``stats`` attribute set by the runner). Reports the
    rejected-step rate — often the earliest blow-up signal, before any field
    check trips — and flags a non-success diffrax result.
    """

    name = "solver_health"

    def on_chunk_end(self, model: Any, state: Any, info: ChunkInfo) -> MonitorVerdict:
        stats = info.stats
        if not stats:
            return MonitorVerdict()
        accepted = stats.get("num_accepted_steps")
        rejected = stats.get("num_rejected_steps")
        metrics: dict[str, float] = {}
        messages: list[str] = []
        if accepted is not None and rejected is not None:
            total = float(accepted) + float(rejected)
            if total > 0:
                rate = float(rejected) / total
                metrics["rejected_step_rate"] = rate
                if rate > 0.5:
                    messages.append(f"high rejected-step rate {rate:.0%}")
        if stats.get("result_successful") is False:
            messages.append("solver reported a non-successful result")
        return MonitorVerdict(metrics=metrics, messages=tuple(messages))


class ThroughputMonitor(BaseMonitor):
    """MONITOR: report simulated seconds per wallclock second per chunk.

    A sudden drop flags a recompilation or a host-side stall.
    """

    name = "throughput"

    def on_chunk_end(self, model: Any, state: Any, info: ChunkInfo) -> MonitorVerdict:
        if info.wall_seconds <= 0:
            return MonitorVerdict()
        sim_span = info.t1 - info.t0
        throughput = sim_span / info.wall_seconds
        return MonitorVerdict(metrics={"sim_s_per_wall_s": throughput})


class WatchdogMonitor(BaseMonitor):
    """FAIL-HARD: terminate if cumulative wallclock exceeds a ceiling.

    A wallclock budget for the whole run. The runner already ticks an
    alive-thread; this enforces a hard ceiling and stops cleanly at the next
    chunk boundary rather than running indefinitely.

    Args:
        max_wall_s: Maximum cumulative wallclock seconds before termination.
    """

    name = "watchdog"

    def __init__(self, max_wall_s: float):
        self.max_wall_s = max_wall_s
        self._t0: float | None = None

    def on_run_start(self, model: Any, state0: Any, *, spec: Any) -> None:
        self._t0 = time.perf_counter()

    def on_chunk_end(self, model: Any, state: Any, info: ChunkInfo) -> MonitorVerdict:
        if self._t0 is None:
            return MonitorVerdict()
        elapsed = time.perf_counter() - self._t0
        if elapsed > self.max_wall_s:
            return MonitorVerdict(
                metrics={"elapsed_wall_s": elapsed},
                terminate=True,
                reason=(
                    f"wallclock {elapsed:.0f}s exceeded ceiling {self.max_wall_s:.0f}s"
                ),
            )
        return MonitorVerdict(metrics={"elapsed_wall_s": elapsed})


def default_monitors() -> list[BaseMonitor]:
    """The runner's default monitor set — preserves legacy behavior.

    ``NonFiniteMonitor`` (the original abort) plus ``EnergyGrowthMonitor`` (the
    original 10x warning), now pluggable. ``ConservationDriftMonitor``,
    ``SolverHealthMonitor`` and ``ThroughputMonitor`` add record-only
    diagnostics on top without changing run outcomes.
    """
    return [
        NonFiniteMonitor(),
        EnergyGrowthMonitor(factor=10.0),
        ConservationDriftMonitor(),
        SolverHealthMonitor(),
        ThroughputMonitor(),
    ]
