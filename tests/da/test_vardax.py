"""Strong-constraint 4DVar driving a somax model through vardax.

End-to-end check that the ``somax.da`` vardax bridge works: a somax model,
wrapped as a flat-vector ``ForwardModel`` via :class:`SomaxForwardModel`, can
be assimilated by ``vardax.StrongFourDVar`` with `gaussx`-built background and
observation covariances. The analysis must improve on the background.

The hard assertion uses the *non-advective* (linear) Lorenz-96 system: strong
4DVar over the fully chaotic model is genuinely fragile inside the outer
minimiser (a long rollout from a poor ``x_0`` can blow up), so the chaotic
case is reserved for the docs tutorial under a fixed, vetted configuration.
Requires the ``da`` dependency group; skipped otherwise.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest


pytest.importorskip("vardax")

import gaussx
import lineax as lx
from vardax import Batch1D, MaskedIdentity, StrongFourDVar

from somax._src.models.lorenz96 import L96State, Lorenz96
from somax.da import SomaxForwardModel, state_to_vector


def _structured_background(key, n: int) -> lx.AbstractLinearOperator:
    """Build a `gaussx` diag + low-rank background covariance B.

    Returns a dense PSD-tagged `lineax` operator: gaussx constructs the
    structured ``LowRankUpdate``; we materialise it so vardax's internal
    ``lx.CG`` solves (which need ``linearise``) accept it.
    """
    psd = lx.positive_semidefinite_tag
    u = jax.random.normal(key, (n, 2)) * 0.3
    base = lx.DiagonalLinearOperator(0.5 * jnp.ones(n))
    b_struct = gaussx.LowRankUpdate(base, u, jnp.ones(2), tags=psd)
    return lx.MatrixLinearOperator(b_struct.as_matrix(), psd)


def _twin_4dvar(advection: bool, n_steps: int, bg_std: float):
    """Run a strong-4DVar twin experiment; return (background, analysis, truth)."""
    key = jax.random.PRNGKey(0)
    K = 12
    dt = 0.05
    obs_var = 0.04

    model = Lorenz96.create(F=8.0, advection=advection)
    step = jax.jit(lambda s: model.step(s, dt))

    # Spin the truth onto the model's attractor / equilibrium.
    truth = L96State(x=8.0 * jnp.ones(K).at[0].add(0.01))
    for _ in range(200):
        truth = step(truth)
    x0_true, _ = state_to_vector(truth)

    forward = SomaxForwardModel(model=model, template=truth, dt=dt)

    # Truth trajectory over the window (T + 1 timesteps) + noisy observations.
    traj = [x0_true]
    state = truth
    for _ in range(n_steps):
        state = step(state)
        vec, _ = state_to_vector(state)
        traj.append(vec)
    traj = jnp.stack(traj)  # (T+1, K)

    noise = jnp.sqrt(obs_var) * jax.random.normal(key, traj.shape)
    batch = Batch1D(
        input=(traj + noise)[None],
        mask=jnp.ones_like(traj)[None],
        target=traj[None],
    )

    # Background = truth perturbed; structured B from gaussx, diagonal R.
    background = x0_true + bg_std * jax.random.normal(jax.random.PRNGKey(1), (K,))
    psd = lx.positive_semidefinite_tag
    B = _structured_background(jax.random.PRNGKey(2), K)
    R = lx.MatrixLinearOperator(jnp.diag(obs_var * jnp.ones(K)), psd)

    strong = StrongFourDVar(
        forward=forward,
        obs_op=MaskedIdentity(),
        prior_mean=background,
        prior_cov_op=B,
        obs_cov_op=R,
    )
    analysis = strong(batch)[0]
    return background, analysis, x0_true


@pytest.mark.slow
def test_strong_4dvar_improves_on_background():
    """4DVar analysis is closer to truth than the background (linear L96)."""
    background, analysis, truth = _twin_4dvar(advection=False, n_steps=5, bg_std=1.0)
    bg_rmse = jnp.sqrt(jnp.mean((background - truth) ** 2))
    an_rmse = jnp.sqrt(jnp.mean((analysis - truth) ** 2))

    assert jnp.all(jnp.isfinite(analysis))
    assert an_rmse < bg_rmse


def test_somax_forward_model_matches_step():
    """The flat-vector ForwardModel adapter equals a direct pytree ``step``."""
    model = Lorenz96.create(F=8.0)
    state = L96State(x=jnp.linspace(-2.0, 2.0, 12))
    forward = SomaxForwardModel(model=model, template=state, dt=0.05)

    vec, _ = state_to_vector(state)
    got = forward.step(vec, 0.05)
    expected, _ = state_to_vector(model.step(state, 0.05))
    assert jnp.allclose(got, expected, atol=1e-6)


def test_somax_forward_model_is_forward_model():
    """The adapter satisfies the pipekit_cycle ForwardModel surface."""
    from pipekit_cycle import ForwardModel

    model = Lorenz96.create(F=8.0)
    state = L96State(x=jnp.zeros(12))
    forward = SomaxForwardModel(model=model, template=state, dt=0.05)

    assert isinstance(forward, ForwardModel)
    assert forward.dt == 0.05
    assert forward.state_signature is None
