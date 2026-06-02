"""Unit tests for the somax.monitor protocol and built-in monitors.

These exercise the monitor objects directly with lightweight fakes — no
diffrax integration — so they run in the fast PR lane.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from somax.monitor import (
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


class _FakeState(eqx.Module):
    u: jnp.ndarray


class _FakeDiagnostics(eqx.Module):
    invs: dict

    def invariants(self) -> dict:
        return self.invs


class _FakeModel(eqx.Module):
    """Model whose ``diagnose`` returns caller-controlled invariants."""

    invs: dict = eqx.field(static=True)

    def diagnose(self, state):
        return _FakeDiagnostics(invs=self.invs)


class _PlainModel:
    """Non-eqx fake so invariant values can be JAX arrays (vector tests)."""

    def __init__(self, invs: dict) -> None:
        self._invs = invs

    def diagnose(self, state):
        return _FakeDiagnostics(invs=self._invs)


def _info(index=0, n_chunks=4, t0=0.0, t1=1.0, wall=0.1, snap=True, stats=None):
    return ChunkInfo(
        index=index,
        n_chunks=n_chunks,
        t0=t0,
        t1=t1,
        wall_seconds=wall,
        is_snapshot=snap,
        stats=stats or {},
    )


class TestProtocolConformance:
    def test_base_monitor_is_monitor(self) -> None:
        assert isinstance(BaseMonitor(), Monitor)

    def test_builtins_are_monitors(self) -> None:
        for mon in default_monitors():
            assert isinstance(mon, Monitor)

    def test_base_monitor_inert_defaults(self) -> None:
        verdict = BaseMonitor().on_chunk_end(None, None, _info())
        assert verdict == MonitorVerdict()
        assert not verdict.terminate


class TestNonFiniteMonitor:
    def test_clean_state_passes(self) -> None:
        v = NonFiniteMonitor().on_chunk_end(None, _FakeState(u=jnp.ones(4)), _info())
        assert not v.terminate

    def test_nan_state_terminates(self) -> None:
        state = _FakeState(u=jnp.array([1.0, jnp.nan]))
        v = NonFiniteMonitor().on_chunk_end(None, state, _info())
        assert v.terminate
        assert v.reason is not None and "non-finite" in v.reason


class TestEnergyGrowthMonitor:
    def test_warns_above_factor(self) -> None:
        mon = EnergyGrowthMonitor(factor=10.0)
        model_lo = _FakeModel(invs={"total_energy": 1.0})
        model_hi = _FakeModel(invs={"total_energy": 100.0})
        mon.on_run_start(model_lo, None, spec=None)
        v = mon.on_chunk_end(model_hi, None, _info())
        assert v.messages and "grew" in v.messages[0]
        assert not v.terminate  # no hard_factor

    def test_hard_factor_terminates(self) -> None:
        mon = EnergyGrowthMonitor(factor=2.0, hard_factor=10.0)
        mon.on_run_start(_FakeModel(invs={"total_energy": 1.0}), None, spec=None)
        v = mon.on_chunk_end(_FakeModel(invs={"total_energy": 50.0}), None, _info())
        assert v.terminate

    def test_stable_energy_no_warning(self) -> None:
        mon = EnergyGrowthMonitor(factor=10.0)
        mon.on_run_start(_FakeModel(invs={"total_energy": 1.0}), None, spec=None)
        v = mon.on_chunk_end(_FakeModel(invs={"total_energy": 1.1}), None, _info())
        assert not v.messages and not v.terminate

    def test_falls_back_to_diagnostics_field(self) -> None:
        """Models without an energy invariant still get the early warning via
        a top-level scalar ``energy`` field on the Diagnostics object."""

        class _DiagWithEnergyField(eqx.Module):
            energy: jnp.ndarray

            def invariants(self) -> dict:
                return {}  # no energy in invariants -> must fall back

        class _ModelEnergyField(eqx.Module):
            e: float = eqx.field(static=True)

            def diagnose(self, state):
                return _DiagWithEnergyField(energy=jnp.asarray(self.e))

        mon = EnergyGrowthMonitor(factor=10.0)
        mon.on_run_start(_ModelEnergyField(e=1.0), None, spec=None)
        v = mon.on_chunk_end(_ModelEnergyField(e=100.0), None, _info())
        assert v.messages and "grew" in v.messages[0]

    def test_vector_energy_invariant_summed(self) -> None:
        """A per-layer vector energy invariant is summed, not float()'d."""
        mon = EnergyGrowthMonitor(factor=10.0)
        mon.on_run_start(
            _PlainModel(invs={"total_energy": jnp.array([0.5, 0.5])}), None, spec=None
        )
        v = mon.on_chunk_end(
            _PlainModel(invs={"total_energy": jnp.array([60.0, 60.0])}), None, _info()
        )
        assert v.messages and "grew" in v.messages[0]


class TestConservationDriftMonitor:
    def test_records_drift(self) -> None:
        mon = ConservationDriftMonitor(rtol_warn=1e-2)
        mon.on_run_start(_FakeModel(invs={"mass": 100.0}), None, spec=None)
        v = mon.on_chunk_end(_FakeModel(invs={"mass": 101.0}), None, _info())
        assert v.metrics["drift_mass"] == 0.01
        assert not v.terminate

    def test_warns_above_rtol(self) -> None:
        mon = ConservationDriftMonitor(rtol_warn=1e-3)
        mon.on_run_start(_FakeModel(invs={"mass": 100.0}), None, spec=None)
        v = mon.on_chunk_end(_FakeModel(invs={"mass": 110.0}), None, _info())
        assert v.messages

    def test_fail_terminates(self) -> None:
        mon = ConservationDriftMonitor(rtol_warn=1e-3, rtol_fail=0.05)
        mon.on_run_start(_FakeModel(invs={"mass": 100.0}), None, spec=None)
        v = mon.on_chunk_end(_FakeModel(invs={"mass": 200.0}), None, _info())
        assert v.terminate

    def test_zero_reference_skipped(self) -> None:
        mon = ConservationDriftMonitor()
        mon.on_run_start(_FakeModel(invs={"x": 0.0}), None, spec=None)
        v = mon.on_chunk_end(_FakeModel(invs={"x": 5.0}), None, _info())
        assert "drift_x" not in v.metrics

    def test_vector_invariant_summed_not_crash(self) -> None:
        """Per-layer vector invariants are summed, not float()'d (no crash)."""
        mon = ConservationDriftMonitor(rtol_warn=1e-2)
        mon.on_run_start(
            _PlainModel(invs={"mass": jnp.array([60.0, 40.0])}), None, spec=None
        )
        # total 100 -> 101 = 1% drift
        v = mon.on_chunk_end(
            _PlainModel(invs={"mass": jnp.array([61.0, 40.0])}), None, _info()
        )
        assert v.metrics["drift_mass"] == 0.01
        assert not v.terminate


class TestSolverHealthMonitor:
    def test_no_stats_inert(self) -> None:
        v = SolverHealthMonitor().on_chunk_end(None, None, _info(stats={}))
        assert v == MonitorVerdict()

    def test_rejected_rate(self) -> None:
        stats = {"num_accepted_steps": 8, "num_rejected_steps": 2}
        v = SolverHealthMonitor().on_chunk_end(None, None, _info(stats=stats))
        assert v.metrics["rejected_step_rate"] == 0.2

    def test_high_rejection_warns(self) -> None:
        stats = {"num_accepted_steps": 1, "num_rejected_steps": 9}
        v = SolverHealthMonitor().on_chunk_end(None, None, _info(stats=stats))
        assert v.messages

    def test_unsuccessful_result_warns(self) -> None:
        v = SolverHealthMonitor().on_chunk_end(
            None, None, _info(stats={"result_successful": False})
        )
        assert v.messages


class TestThroughputMonitor:
    def test_reports_throughput(self) -> None:
        v = ThroughputMonitor().on_chunk_end(
            None, None, _info(t0=0.0, t1=100.0, wall=2.0)
        )
        assert v.metrics["sim_s_per_wall_s"] == 50.0

    def test_zero_wall_inert(self) -> None:
        v = ThroughputMonitor().on_chunk_end(None, None, _info(wall=0.0))
        assert v == MonitorVerdict()


class TestWatchdogMonitor:
    def test_terminates_past_ceiling(self) -> None:
        mon = WatchdogMonitor(max_wall_s=-1.0)  # already exceeded
        mon.on_run_start(None, None, spec=None)
        v = mon.on_chunk_end(None, None, _info())
        assert v.terminate

    def test_within_budget_ok(self) -> None:
        mon = WatchdogMonitor(max_wall_s=1e9)
        mon.on_run_start(None, None, spec=None)
        v = mon.on_chunk_end(None, None, _info())
        assert not v.terminate


def test_default_monitors_membership() -> None:
    names = {m.name for m in default_monitors()}
    assert {"non_finite", "energy_growth", "conservation"} <= names
