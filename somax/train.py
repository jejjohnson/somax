"""Public surface for training somax models through jit-native rollouts.

- :func:`rollout` — step a model ``n_steps`` times in one ``jax.lax.scan``
  (via ``pipekit_jax.Rollout``), returning the recorded trajectory.
- :func:`rollout_loss` — L(θ) = (1/T) ∑ₜ l(s_{tk}(θ), sʳᵉᶠₜ) against a
  reference trajectory; differentiate it with ``eqx.filter_grad``.
- :func:`state_mse` — the default per-snapshot loss.
- :class:`ModelStep` — a model step as a pipekit operator, for building
  your own ``ScanCycle`` / ``Rollout``.

Use these inside ``jit`` / ``grad``. Host-side runs that inspect the state
between chunks (the ``somax-sim`` runner) use ``pipekit_cycle.Cycle``.
"""

from __future__ import annotations

from somax._src.train.rollout import ModelStep, rollout, rollout_loss, state_mse


__all__ = [
    "ModelStep",
    "rollout",
    "rollout_loss",
    "state_mse",
]
