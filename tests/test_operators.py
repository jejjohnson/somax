"""somax models as pipekit Operators with serializable flat configs.

Exercises the pipekit bridge (``somax.operators``): flat-primitive
``get_config``, faithful ``pipekit.serial`` round-trip, ForwardModel
conformance, pipeline composition, and driving ``pipekit_cycle.Cycle``.

These require pipekit (a base somax dependency); the ``importorskip``
guard below is belt-and-suspenders.
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


# ----------------------------------------------------------------------
# General registry-driven wrapper (any scenario x model pair)
# ----------------------------------------------------------------------


def _small_linear_swm_op():
    return SomaxModelOp.from_registry(
        "double_gyre",
        "linear_swm",
        dt=30.0,
        scenario_params={"grid": {"nx": 8, "ny": 8}},
        model_params={"params": {"lateral_viscosity": 100.0, "bottom_drag": 1e-6}},
    )


def test_from_registry_wraps_any_model():
    op, state0 = _small_linear_swm_op()
    assert isinstance(op, SomaxModelOp)
    assert type(op.model).__name__ == "LinearShallowWater2D"
    # registry initial state has the SWM (h, u, v) structure
    assert hasattr(state0, "h") and hasattr(state0, "u") and hasattr(state0, "v")


def test_from_registry_op_is_forward_model_and_steps():
    op, state0 = _small_linear_swm_op()
    assert isinstance(op, ForwardModel)
    stepped = op(state0)
    assert jnp.all(jnp.isfinite(stepped.h))
    # _apply == step(dt) == model.step(dt)
    assert jnp.allclose(stepped.h, op.model.step(state0, op.dt).h, atol=1e-12)


def test_from_registry_op_drives_cycle():
    op, state0 = _small_linear_swm_op()
    carrier, _ = Cycle(step_op=op, n_steps=2)(state0, None)
    manual = op.step(op.step(state0, op.dt), op.dt)
    assert jnp.allclose(carrier.h, manual.h, atol=1e-10)


def test_general_wrapper_is_not_round_trippable():
    # The general wrapper holds a non-primitive eqx.Module: no faithful
    # serial round-trip (forbid_in_yaml), and an empty auto-config.
    op, _ = _small_linear_swm_op()
    assert op.forbid_in_yaml is True
    assert op.get_config() == {}


def test_flat_subclass_still_round_trips():
    # Burgers2DOp re-enables the round-trip the general wrapper forgoes.
    op = Burgers2DOp(nx=8, ny=8, nu=0.05, dt=0.001)
    assert op.forbid_in_yaml is False
    assert loads(dumps(op)).get_config() == op.get_config()
