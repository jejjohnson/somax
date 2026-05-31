"""ETKF twin experiment driving a somax Lorenz-96 model through filterax.

End-to-end check that the ``somax.da`` filterax adapters work: a perturbed
ensemble assimilating sparse, noisy observations of a truth run should track
the truth — the analysis is closer to truth than the un-corrected forecast,
and well below the observation-noise floor. Requires the ``da`` dependency
group; skipped otherwise.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest


pytest.importorskip("filterax")

import lineax as lx
from filterax import ETKF

from somax._src.models.lorenz96 import L96State, Lorenz96
from somax.da import (
    SomaxDynamics,
    SubsampleObs,
    make_ensemble,
    state_to_vector,
)


def _twin_experiment(seed: int = 0):
    """Run a Lorenz-96 ETKF twin experiment; return analysis/forecast/truth.

    A 40-variable Lorenz-96 truth is spun onto the attractor, then observed
    at every other grid point with Gaussian noise (variance ``obs_var``). A
    perturbed 40-member ensemble assimilates those observations through the
    somax -> filterax adapters.
    """
    key = jax.random.PRNGKey(seed)
    K = 40
    dt = 0.05
    n_windows = 40
    n_members = 40
    obs_var = 0.5

    model = Lorenz96.create(F=8.0)
    step = jax.jit(lambda s: model.step(s, dt))

    # Spin the truth onto the Lorenz-96 attractor.
    truth = L96State(x=8.0 * jnp.ones(K).at[0].add(0.01))
    for _ in range(300):
        truth = step(truth)
    truth0 = truth

    # Truth trajectory + sparse, noisy observations (every other grid point).
    obs_idx = jnp.arange(0, K, 2)
    obs_op = SubsampleObs(indices=obs_idx)
    R = lx.DiagonalLinearOperator(obs_var * jnp.ones(obs_idx.size))

    truths, observations = [], []
    state = truth0
    for w in range(1, n_windows + 1):
        state = step(state)
        vec, _ = state_to_vector(state)
        truths.append(vec)
        key, sub = jax.random.split(key)
        noise = jnp.sqrt(obs_var) * jax.random.normal(sub, (obs_idx.size,))
        observations.append((vec[obs_idx] + noise, w * dt))
    truths = jnp.stack(truths)  # (T, N_x)

    # Perturbed ensemble + ETKF over the observation windows.
    key, sub = jax.random.split(key)
    ensemble0 = make_ensemble(truth0, sub, size=n_members, std=1.0)
    etkf = ETKF(dynamics=SomaxDynamics(model=model, template=truth0), obs_op=obs_op)
    result = etkf.assimilate(ensemble0, observations, R, t0=0.0)

    analysis_mean = result.analysis_history.mean(axis=1)  # (T, N_x)
    forecast_mean = result.forecast_history.mean(axis=1)  # (T, N_x)
    return analysis_mean, forecast_mean, truths, obs_var


@pytest.mark.slow
def test_etkf_analysis_beats_forecast():
    """Assimilating obs pulls the ensemble closer to truth than the forecast."""
    analysis_mean, forecast_mean, truths, obs_var = _twin_experiment()

    analysis_rmse = jnp.sqrt(jnp.mean((analysis_mean - truths) ** 2))
    forecast_rmse = jnp.sqrt(jnp.mean((forecast_mean - truths) ** 2))

    # The analysis must be finite, beat the (un-corrected) forecast, and sit
    # well below the observation-noise floor — i.e. the filter genuinely
    # constrains the state rather than just echoing the noisy observations.
    assert jnp.isfinite(analysis_rmse)
    assert analysis_rmse < forecast_rmse
    assert analysis_rmse < jnp.sqrt(obs_var)


def test_somax_dynamics_matches_model_step():
    """The flat-vector dynamics adapter equals a direct pytree ``step``."""
    model = Lorenz96.create(F=8.0)
    state = L96State(x=jnp.linspace(-2.0, 2.0, 12))
    dyn = SomaxDynamics(model=model, template=state)

    vec, _ = state_to_vector(state)
    got = dyn(vec, jnp.asarray(0.0), jnp.asarray(0.05))
    expected, _ = state_to_vector(model.step(state, 0.05))
    assert jnp.allclose(got, expected, atol=1e-6)


def test_subsample_obs_selects_indices():
    """The observation operator selects exactly the requested components."""
    obs_op = SubsampleObs(indices=jnp.array([0, 2, 4]))
    x = jnp.arange(6.0)
    assert jnp.array_equal(obs_op(x), jnp.array([0.0, 2.0, 4.0]))
