"""somax models as pipekit Operators with serializable flat configs.

Exercises the pipekit bridge (``somax.operators``): flat-primitive
``get_config``, faithful ``pipekit.serial`` round-trip, ForwardModel
conformance, pipeline composition, and driving ``pipekit_cycle.Cycle``.

These require pipekit (the ``somax[sim]`` extra); they're skipped if it
isn't installed.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest


pytest.importorskip("pipekit")
pytest.importorskip("pipekit_cycle")

from pipekit import dumps, loads
from pipekit_cycle import Cycle, ForwardModel

from somax._src.models.pde2d.burgers import Burgers2DState
from somax.operators import Burgers2DOp, SomaxModelOp


def _gaussian_state(model) -> Burgers2DState:
    grid = model.grid
    x = jnp.arange(grid.Nx) * grid.dx
    y = jnp.arange(grid.Ny) * grid.dy
    X, Y = jnp.meshgrid(x, y)
    g = jnp.exp(-0.5 * (((X - 1.0) / 0.3) ** 2 + ((Y - 1.0) / 0.3) ** 2))
    return Burgers2DState(u=g, v=g)


# ----------------------------------------------------------------------
# Config: flat primitives only
# ----------------------------------------------------------------------


def test_get_config_is_flat_primitives():
    op = Burgers2DOp(nx=16, ny=16, Lx=2.0, Ly=2.0, nu=0.05, dt=0.001)
    config = op.get_config()
    assert config == {
        "nx": 16,
        "ny": 16,
        "Lx": 2.0,
        "Ly": 2.0,
        "nu": 0.05,
        "method": "upwind1",
        "imex": False,
        "dt": 0.001,
    }
    # Every value is a JSON primitive (this is what makes serial work).
    assert all(isinstance(v, int | float | str | bool) for v in config.values())


def test_model_not_in_config():
    op = Burgers2DOp(nx=8, ny=8)
    assert "model" not in op.get_config()
    assert "_model" not in op.get_config()


# ----------------------------------------------------------------------
# pipekit.serial round-trip
# ----------------------------------------------------------------------


def test_serial_round_trip_config():
    op = Burgers2DOp(nx=16, ny=16, nu=0.05, method="upwind1", imex=False, dt=0.002)
    restored = loads(dumps(op))
    assert isinstance(restored, Burgers2DOp)
    assert restored.get_config() == op.get_config()


def test_serial_round_trip_steps_identically():
    op = Burgers2DOp(nx=16, ny=16, nu=0.05, dt=0.001)
    restored = loads(dumps(op))
    state = _gaussian_state(op.model)
    out_a = op(state)
    out_b = restored(state)
    assert jnp.allclose(out_a.u, out_b.u, atol=1e-12)
    assert jnp.allclose(out_a.v, out_b.v, atol=1e-12)


def test_imex_flag_round_trips():
    op = Burgers2DOp(nx=8, ny=8, imex=True)
    restored = loads(dumps(op))
    assert restored.get_config()["imex"] is True
    # The rebuilt model carries the IMEX split.
    import diffrax as dfx

    assert isinstance(restored.model.build_terms(), dfx.MultiTerm)


# ----------------------------------------------------------------------
# ForwardModel conformance + stepping
# ----------------------------------------------------------------------


def test_is_forward_model():
    op = Burgers2DOp(nx=8, ny=8)
    assert isinstance(op, ForwardModel)


def test_apply_advances_one_dt():
    op = Burgers2DOp(nx=16, ny=16, nu=0.05, dt=0.001)
    state = _gaussian_state(op.model)
    via_apply = op(state)
    via_step = op.step(state, 0.001)
    via_model = op.model.step(state, 0.001)
    assert jnp.allclose(via_apply.u, via_step.u, atol=1e-12)
    assert jnp.allclose(via_apply.u, via_model.u, atol=1e-12)


def test_base_class_exported():
    assert issubclass(Burgers2DOp, SomaxModelOp)


# ----------------------------------------------------------------------
# pipekit composition + Cycle
# ----------------------------------------------------------------------


def test_sequential_composition_is_two_steps():
    op = Burgers2DOp(nx=16, ny=16, nu=0.05, dt=0.001)
    state = _gaussian_state(op.model)
    composed = (op | op)(state)
    manual = op(op(state))
    assert jnp.allclose(composed.u, manual.u, atol=1e-12)
    assert jnp.allclose(composed.v, manual.v, atol=1e-12)


def test_drives_pipekit_cycle():
    op = Burgers2DOp(nx=16, ny=16, nu=0.05, dt=0.001)
    state0 = _gaussian_state(op.model)

    cycle = Cycle(step_op=op, n_steps=3, save_history=True)
    carrier, _ = cycle(state0, None)

    # Manual three-step reference.
    manual = state0
    for _ in range(3):
        manual = op.step(manual, op.dt)

    assert jnp.allclose(carrier.u, manual.u, atol=1e-10)
    assert jnp.allclose(carrier.v, manual.v, atol=1e-10)
    assert len(cycle.history) == 3
