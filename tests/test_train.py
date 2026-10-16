"""``somax.train``: jit-native rollouts and the rollout loss."""

from __future__ import annotations

from collections.abc import Iterator

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pipekit_jax import JaxOperator

from somax._src.core.terms import TermFn
from somax._src.models.swm.linear_2d import LinearSW2DState
from somax._src.models.swm.linear_swm_terms import (
    LinearSWM2DTermModel,
    LinearSWMCoriolis,
    LinearSWMGravityWave,
)
from somax.operators import OperatorTerm
from somax.train import rollout, rollout_loss, state_mse


DT = 600.0


@pytest.fixture
def x64() -> Iterator[None]:
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", False)


def _model(kappa: float = 0.0) -> LinearSWM2DTermModel:
    return LinearSWM2DTermModel.create(
        nx=16,
        ny=16,
        Lx=1e6,
        Ly=1e6,
        f0=1e-4,
        beta=2e-11,
        H0=100.0,
        lateral_viscosity=0.0,
        bottom_drag=kappa,
    )


def _bump(model) -> LinearSW2DState:
    g = model.grid
    x = jnp.arange(g.Nx) * g.dx
    y = jnp.arange(g.Ny) * g.dy
    X, Y = jnp.meshgrid(x, y)
    s = g.Nx * g.dx / 8
    h = jnp.exp(
        -0.5 * (((X - g.Nx * g.dx / 2) / s) ** 2 + ((Y - g.Ny * g.dy / 2) / s) ** 2)
    )
    return model.apply_boundary_conditions(
        LinearSW2DState(h=h, u=jnp.zeros_like(h), v=jnp.zeros_like(h))
    )


_step = eqx.filter_jit(lambda m, s, t: m.step(s, DT, t0=t))


def _loop(model, s0, n, every=1, t0=0.0):
    out, s, t = [], s0, t0
    for k in range(1, n + 1):
        s = _step(model, s, t)
        t += DT
        if k % every == 0:
            out.append(s)
    return jax.tree.map(lambda *xs: jnp.stack(xs), *out)


def _assert_close(a, b, rtol=1e-6, atol=1e-7):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_allclose(x, y, rtol=rtol, atol=atol)


# ----------------------------------------------------------------------
# rollout
# ----------------------------------------------------------------------


def test_rollout_matches_a_python_loop():
    model = _model()
    s0 = _bump(model)
    traj = rollout(model, s0, dt=DT, n_steps=6, save_every=2)
    assert traj.h.shape == (3, model.grid.Ny, model.grid.Nx)
    _assert_close(traj, _loop(model, s0, 6, every=2))


def test_rollout_can_include_the_initial_state():
    model = _model()
    s0 = _bump(model)
    traj = rollout(model, s0, dt=DT, n_steps=2, include_initial=True)
    assert traj.h.shape[0] == 3
    np.testing.assert_array_equal(traj.h[0], s0.h)


def test_rollout_carries_the_clock_for_time_dependent_terms():
    model = _model()
    forcing = TermFn(
        lambda t, s, a: LinearSW2DState(
            h=jnp.sin(t / 3600.0) * 1e-5 * jnp.ones_like(s.h),
            u=jnp.zeros_like(s.u),
            v=jnp.zeros_like(s.v),
        )
    )
    forced = eqx.tree_at(lambda m: m.terms, model, model.terms + forcing)
    s0 = _bump(forced)
    traj = rollout(forced, s0, dt=DT, n_steps=4, t0=1800.0)
    _assert_close(traj, _loop(forced, s0, 4, t0=1800.0))
    # restarting every step from t0 would give a different answer
    frozen = jax.tree.map(lambda *xs: jnp.stack(xs), *_restart_from(forced, s0, 4))
    assert not np.allclose(traj.h, frozen.h)


def _restart_from(model, s0, n):
    s, out = s0, []
    for _ in range(n):
        s = _step(model, s, 1800.0)
        out.append(s)
    return out


def test_scan_traces_the_step_a_fixed_number_of_times():
    model = _model()
    s0 = _bump(model)
    calls = []

    def counting(t, s, a):
        calls.append(1)
        return jax.tree.map(jnp.zeros_like, s)

    m = eqx.tree_at(lambda m: m.terms, model, model.terms + TermFn(counting))

    def traces(n):
        calls.clear()
        jax.jit(lambda s: rollout(m, s, dt=DT, n_steps=n))(s0)
        return len(calls)

    assert traces(4) == traces(32)


# ----------------------------------------------------------------------
# rollout_loss
# ----------------------------------------------------------------------


def test_rollout_loss_is_zero_on_its_own_trajectory_and_positive_otherwise():
    model = _model()
    s0 = _bump(model)
    ref = rollout(model, s0, dt=DT, n_steps=6, save_every=3)
    assert float(rollout_loss(model, s0, ref, dt=DT, stride=3)) == 0.0
    other = eqx.tree_at(lambda m: m.terms.terms[0].g, model, jnp.asarray(5.0))
    assert float(rollout_loss(other, s0, ref, dt=DT, stride=3)) > 0.0


def test_rollout_loss_is_the_mean_snapshot_loss():
    model = _model()
    s0 = _bump(model)
    ref = jax.tree.map(lambda x: 0.5 * x, rollout(model, s0, dt=DT, n_steps=4))
    pred = rollout(model, s0, dt=DT, n_steps=4)
    want = np.mean(
        [
            float(
                state_mse(
                    jax.tree.map(lambda x, i=i: x[i], pred),
                    jax.tree.map(lambda x, i=i: x[i], ref),
                )
            )
            for i in range(4)
        ]
    )
    np.testing.assert_allclose(rollout_loss(model, s0, ref, dt=DT), want, rtol=1e-6)


def test_rollout_loss_rejects_ragged_references():
    model = _model()
    s0 = _bump(model)
    ref = LinearSW2DState(
        h=jnp.zeros((2, 18, 18)), u=jnp.zeros((3, 18, 18)), v=jnp.zeros((2, 18, 18))
    )
    with pytest.raises(ValueError, match="different lengths"):
        rollout_loss(model, s0, ref, dt=DT)


@pytest.mark.slow  # two float64 compiles of a differentiated rollout
@pytest.mark.parametrize("checkpoint", [True, False])
def test_rollout_loss_gradient_matches_finite_differences(x64, checkpoint):
    model = _model()
    s0 = _bump(model)
    truth = eqx.tree_at(lambda m: m.terms.terms[0].g, model, jnp.asarray(9.0))
    ref = rollout(truth, s0, dt=DT, n_steps=8, save_every=2)

    def loss(m):
        return rollout_loss(m, s0, ref, dt=DT, stride=2, checkpoint=checkpoint)

    def with_g(g):
        return eqx.tree_at(lambda m: m.terms.terms[0].g, model, jnp.asarray(g))

    ad = eqx.filter_grad(loss)(model).terms.terms[0].g
    eps = 1e-4
    fd = (loss(with_g(9.81 + eps)) - loss(with_g(9.81 - eps))) / (2 * eps)
    np.testing.assert_allclose(ad, fd, rtol=1e-4)


# ----------------------------------------------------------------------
# End to end: train a learned closure inside the term tree
# ----------------------------------------------------------------------


class Drag(JaxOperator):
    """(u, v) -> (-a u, -a v), ``a`` learnable."""

    a: jax.Array = eqx.field(converter=jnp.asarray)

    def _apply(self, u, v):
        return -self.a * u, -self.a * v


@pytest.mark.slow  # 30 gradient evaluations of a 12-step rollout
def test_a_learned_closure_trains_through_a_rollout(x64):
    truth = _model(kappa=2e-4)
    s0 = _bump(truth)
    ref = rollout(truth, s0, dt=DT, n_steps=12, save_every=3)
    physics = [
        t
        for t in truth.terms.terms
        if isinstance(t, LinearSWMGravityWave | LinearSWMCoriolis)
    ]
    closure = OperatorTerm(op=Drag(a=0.0), inputs=("u", "v"), outputs=("u", "v"))
    hybrid = LinearSWM2DTermModel(
        terms=physics[0] + physics[1] + closure, grid=truth.grid
    )

    def loss(m):
        return rollout_loss(m, s0, ref, dt=DT, stride=3)

    loss_grad = eqx.filter_jit(eqx.filter_value_and_grad(loss))
    a = hybrid.terms.terms[2].op.a
    first = None
    for _ in range(30):
        value, grads = loss_grad(hybrid)
        first = value if first is None else first
        g = grads.terms.terms[2].op.a
        # Newton-like scalar step on a quadratic-ish loss: a ← a - L / L'
        a = a - 0.5 * value / g
        hybrid = eqx.tree_at(lambda m: m.terms.terms[2].op.a, hybrid, a)
    assert float(loss(hybrid)) < 1e-6 * float(first)
    np.testing.assert_allclose(a, 2e-4, rtol=1e-2)
