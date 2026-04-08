"""Tests for SimulationCheckpointer."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from somax._src.core.checkpoint import SimulationCheckpointer
from somax.core import Params, State


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _orbax_available() -> bool:
    try:
        import orbax.checkpoint  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Minimal model stub for checkpointing tests
# ---------------------------------------------------------------------------


class MyState(State):
    x: jnp.ndarray


class MyParams(Params):
    alpha: jnp.ndarray


class StubModel:
    """Minimal object with a .params attribute."""

    def __init__(self, params):
        self.params = params


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSimulationCheckpointer:
    def test_should_save(self):
        ckpt = SimulationCheckpointer(
            checkpoint_dir="/tmp/ckpt", checkpoint_interval=10
        )
        assert not ckpt.should_save(0)
        assert not ckpt.should_save(5)
        assert ckpt.should_save(10)
        assert ckpt.should_save(20)

    def test_latest_step_no_dir(self, tmp_path):
        ckpt = SimulationCheckpointer(
            checkpoint_dir=str(tmp_path / "nonexistent"),
            checkpoint_interval=10,
        )
        assert ckpt.latest_step() is None

    def test_latest_step_empty_dir(self, tmp_path):
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        ckpt = SimulationCheckpointer(
            checkpoint_dir=str(ckpt_dir), checkpoint_interval=10
        )
        assert ckpt.latest_step() is None

    def test_latest_step_with_checkpoints(self, tmp_path):
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        (ckpt_dir / "step_00000010").mkdir()
        (ckpt_dir / "step_00000020").mkdir()
        (ckpt_dir / "step_00000005").mkdir()

        ckpt = SimulationCheckpointer(
            checkpoint_dir=str(ckpt_dir), checkpoint_interval=10
        )
        assert ckpt.latest_step() == 20

    @pytest.mark.skipif(
        not _orbax_available(),
        reason="orbax-checkpoint not installed",
    )
    def test_save_and_restore(self, tmp_path):
        """Round-trip save/restore with orbax."""
        ckpt_dir = tmp_path / "checkpoints"
        ckpt = SimulationCheckpointer(
            checkpoint_dir=str(ckpt_dir), checkpoint_interval=10
        )

        state = MyState(x=jnp.array([1.0, 2.0, 3.0]))
        params = MyParams(alpha=jnp.array(0.5))
        model = StubModel(params=params)

        ckpt.save(step=10, state=state, model=model)

        target_state = MyState(x=jnp.zeros(3))
        target_model = StubModel(params=MyParams(alpha=jnp.zeros(())))
        restored_state, restored_params, restored_step = ckpt.restore(
            step=10, target_state=target_state, target_model=target_model
        )
        assert restored_step == 10
        assert jnp.allclose(restored_state.x, state.x)
        assert jnp.isclose(restored_params.alpha, params.alpha)
