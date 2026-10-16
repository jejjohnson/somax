"""Jit-native rollouts and the rollout loss for training somax models.

Two ways to step a somax model many times, for two different jobs::

    host side (CLI runs, monitors between chunks)     in jit / grad (training, DA)
    ──────────────────────────────────────────         ────────────────────────────
    pipekit_cycle.Cycle                                pipekit_jax.Rollout / ScanCycle
    Python `for` loop: inspect state between          one `jax.lax.scan`: compile time
    steps, log, checkpoint, abort                      and program size flat in n_steps

Under ``jit`` a ``Cycle`` unrolls: the traced program, compile time and
memory grow linearly with the number of steps. That is fine for the CLI
(each chunk is jitted separately; the loop itself stays on the host) and
wrong for anything differentiated end to end. `rollout` and `rollout_loss`
are the jit-native path.

The carrier is ``(t, state)``: every step integrates from the current time
``t`` (``model.step(state, dt, t0=t)``), so time-dependent forcing is
evaluated at the right time — unlike a bare ``ForwardModel`` rollout, which
restarts each substep from the same ``t0``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree
from pipekit_jax import JaxOperator, Rollout


class ModelStep(JaxOperator):
    """One model step as a carrier-only operator: ``(t, s) ↦ (t + Δt, step(s))``.

    The model is a field, so every leaf of the model (term coefficients,
    physical constants, the weights of an ``OperatorTerm`` closure) is a
    leaf of the step and reachable by ``eqx.filter_grad``.

    Args:
        model: A somax model (anything with ``step(state, dt, t0=...)``).
        dt: Step size.
    """

    # The model holds grids and finitevolX operators: no JSON form.
    forbid_in_yaml: ClassVar[bool] = True

    model: Any
    dt: float = eqx.field(static=True)

    def _apply(self, carrier: tuple[Array, PyTree]) -> tuple[Array, PyTree]:
        t, state = carrier
        return t + self.dt, self.model.step(state, self.dt, t0=t)

    def get_config(self) -> dict[str, Any]:
        """Debug config: the model class and the step size."""
        return {"model_class": type(self.model).__name__, "dt": self.dt}


def rollout(
    model: Any,
    state0: PyTree,
    *,
    dt: float,
    n_steps: int,
    save_every: int = 1,
    include_initial: bool = False,
    t0: float = 0.0,
    checkpoint: bool = False,
) -> PyTree:
    """Step ``model`` ``n_steps`` times in one ``jax.lax.scan``.

    ``sₖ = stepᵏ(s₀)``, recorded every ``save_every`` steps.

    Args:
        model: A somax model (``step(state, dt, t0=...)``).
        state0: Initial state.
        dt: Step size.
        n_steps: Number of steps; a multiple of ``save_every``.
        save_every: Record every ``save_every``-th state.
        include_initial: Also record ``s₀`` first.
        t0: Start time.
        checkpoint: Rematerialise each ``save_every`` block on the backward
            pass (memory O(n_steps / save_every + save_every) instead of
            O(n_steps)).

    Returns:
        The states, stacked along a new leading axis:
        ``[n_steps // save_every (+1), ...]`` per leaf.
    """
    scan = Rollout(
        step=ModelStep(model=model, dt=dt),
        n_steps=n_steps,
        save_every=save_every,
        include_initial=include_initial,
        checkpoint=checkpoint,
    )
    t_start = jnp.asarray(t0, dtype=jnp.result_type(float))
    _, trajectory = scan((t_start, state0))
    return trajectory


def state_mse(pred: PyTree, ref: PyTree) -> Array:
    """l(s, sʳᵉᶠ) = ∑_fields mean((s - sʳᵉᶠ)²) over the array leaves."""
    errs = jax.tree.map(lambda p, r: jnp.mean((p - r) ** 2), pred, ref)
    return sum(jax.tree.leaves(errs))


def rollout_loss(
    model: Any,
    state0: PyTree,
    ref_traj: PyTree,
    *,
    dt: float,
    stride: int = 1,
    loss: Callable[[PyTree, PyTree], Array] = state_mse,
    t0: float = 0.0,
    checkpoint: bool = True,
) -> Array:
    """Mean per-snapshot loss of a rollout against a reference trajectory.

    L(θ) = (1/T) ∑ₜ l(s_{tk}(θ), sʳᵉᶠₜ),  t = 1 … T,  k = ``stride``

    where s_{tk} = stepᵗᵏ(s₀) and ``ref_traj`` holds sʳᵉᶠ₁ … sʳᵉᶠ_T stacked
    on a leading axis (s₀ excluded). Differentiate it with
    ``eqx.filter_grad(lambda m: rollout_loss(m, ...))(model)``.

    Observe the whole window, not only its end: a final-snapshot loss of a
    wave model has many local minima (a wave a full period ahead or behind
    matches as well), while a trajectory loss pins the phase.

    Args:
        model: A somax model (``step(state, dt, t0=...)``).
        state0: Initial state s₀.
        ref_traj: Reference states ``[T, ...]`` per leaf, one every
            ``stride`` steps after s₀.
        dt: Step size.
        stride: Steps between reference snapshots (k).
        loss: Per-snapshot loss l(state, ref_state); default `state_mse`.
        t0: Start time.
        checkpoint: Rematerialise each block on the backward pass.

    Returns:
        The scalar loss L.

    Raises:
        ValueError: if ``ref_traj``'s leaves disagree on their length.
    """
    lengths = {leaf.shape[0] for leaf in jax.tree.leaves(ref_traj)}
    if len(lengths) != 1:
        raise ValueError(f"ref_traj leaves have different lengths: {sorted(lengths)}")
    (n_snapshots,) = lengths
    pred = rollout(
        model,
        state0,
        dt=dt,
        n_steps=stride * n_snapshots,
        save_every=stride,
        t0=t0,
        checkpoint=checkpoint,
    )
    return jnp.mean(jax.vmap(loss)(pred, ref_traj))
