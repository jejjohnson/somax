"""Lorenz '96 periodic 1D chaotic system."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PyTree

from somax._src.core.model import SomaxModel
from somax._src.core.types import Diagnostics, Params, State


class L96Params(Params):
    """Differentiable parameters for the Lorenz '96 system.

    Args:
        F: External forcing amplitude (default 8).
    """

    F: Array


class L96State(State):
    """State vector for the Lorenz '96 system.

    Args:
        x: State variables with shape ``(N,)`` or ``(batch, N)``.
    """

    x: Array

    @classmethod
    def init_state(
        cls,
        ndim: int = 10,
        noise: float = 0.01,
        batchsize: int = 1,
        F: float = 8.0,
        key: jax.Array | None = None,
    ) -> L96State:
        """Create an initial state near the F-equilibrium.

        Args:
            ndim: Number of variables.
            noise: Perturbation amplitude on the first variable.
            batchsize: Number of ensemble members.
            F: Forcing value (used to set equilibrium).
            key: PRNG key for reproducibility.

        Returns:
            An ``L96State`` instance.
        """
        if key is None:
            key = jrandom.PRNGKey(123)
        if batchsize > 1:
            x0 = F * jnp.ones(shape=(batchsize, ndim))
            perturb = noise * jrandom.normal(key, shape=(batchsize,))
        else:
            x0 = F * jnp.ones(shape=(ndim,))
            perturb = noise * jrandom.normal(key, shape=())

        x0 = x0.at[..., 0].set(x0[..., 0] + perturb)
        return cls(x=x0)


class L96Diagnostics(Diagnostics):
    """On-demand diagnostics for the Lorenz '96 system.

    Args:
        energy: Total energy 0.5 * sum(x^2).
        mean: Spatial mean of x.
    """

    energy: Array
    mean: Array


class Lorenz96(SomaxModel):
    """Lorenz '96 periodic 1D chaotic system.

    N coupled ODEs with periodic boundary conditions::

        dX_k/dt = (X_{k+1} - X_{k-2}) * X_{k-1} - X_k + F

    Args:
        params: Differentiable parameters (F).
        advection: Whether to include the nonlinear advection term.
    """

    params: L96Params
    advection: bool = eqx.field(default=True, static=True)

    def vector_field(
        self, t: float, state: PyTree, args: PyTree | None = None
    ) -> L96State:
        """Compute the Lorenz '96 tendency."""
        x_dot = rhs_lorenz_96(x=state.x, F=self.params.F, advection=self.advection)
        return L96State(x=x_dot)

    def apply_boundary_conditions(self, state: PyTree) -> PyTree:
        """Identity — periodicity is implicit via jnp.roll."""
        return state

    def diagnose(self, state: PyTree) -> L96Diagnostics:
        """Compute energy and mean diagnostics."""
        return L96Diagnostics(
            energy=0.5 * jnp.sum(state.x**2),
            mean=jnp.mean(state.x),
        )

    @staticmethod
    def create(F: float = 8.0, advection: bool = True) -> Lorenz96:
        """Convenience factory with standard defaults.

        Args:
            F: Forcing amplitude.
            advection: Include nonlinear advection term.

        Returns:
            A ``Lorenz96`` model instance.
        """
        return Lorenz96(params=L96Params(F=jnp.array(F)), advection=advection)


def rhs_lorenz_96(x: Array, F: float = 8, advection: bool = True) -> Array:
    """Pure-function RHS for the Lorenz '96 system."""
    x_plus_1 = jnp.roll(x, -1)
    x_minus_2 = jnp.roll(x, 2)
    x_minus_1 = jnp.roll(x, 1)

    if advection:
        return (x_plus_1 - x_minus_2) * x_minus_1 - x + F
    else:
        return -x + F
