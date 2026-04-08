"""Lorenz '63 three-variable chaotic system."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PyTree

from somax._src.core.model import SomaxModel
from somax._src.core.types import Diagnostics, Params, State


class L63Params(Params):
    """Differentiable parameters for the Lorenz '63 system.

    Args:
        sigma: Prandtl number (default 10).
        rho: Rayleigh number (default 28).
        beta: Geometric factor (default 8/3).
    """

    sigma: Array
    rho: Array
    beta: Array


class L63State(State):
    """State vector for the Lorenz '63 system.

    Args:
        x: First variable.
        y: Second variable.
        z: Third variable.
    """

    x: Array
    y: Array
    z: Array

    @classmethod
    def init_state(
        cls,
        noise: float = 0.01,
        batchsize: int = 1,
        key: jax.Array | None = None,
    ) -> L63State:
        """Create an initial state near the (1, 1, 1) equilibrium.

        Args:
            noise: Perturbation amplitude.
            batchsize: Number of ensemble members.
            key: PRNG key for reproducibility.

        Returns:
            An ``L63State`` instance.
        """
        msg = f"batchsize not >= 1, {batchsize}"
        assert batchsize >= 1, msg

        if key is None:
            key = jrandom.PRNGKey(123)

        if batchsize > 1:
            x0, y0, z0 = jnp.array_split(jnp.ones((batchsize, 3)), 3, axis=-1)
            perturb = noise * jrandom.normal(key, shape=(batchsize, 1))
        else:
            x0, y0, z0 = jnp.array_split(jnp.ones((3,)), 3, axis=-1)
            perturb = noise * jrandom.normal(key, shape=())

        return cls(x=x0 + perturb, y=y0, z=z0)

    @property
    def array(self) -> Array:
        """Stack (x, y, z) into a single array."""
        return jnp.stack([self.x, self.y, self.z], axis=-1).squeeze()


class L63Diagnostics(Diagnostics):
    """On-demand diagnostics for the Lorenz '63 system.

    Args:
        energy: Kinetic energy 0.5 * (x^2 + y^2 + z^2).
    """

    energy: Array


class Lorenz63(SomaxModel):
    """Lorenz '63 three-variable chaotic system.

    The canonical low-dimensional chaotic attractor::

        dx/dt = sigma * (y - x)
        dy/dt = x * (rho - z) - y
        dz/dt = x * y - beta * z

    Args:
        params: Differentiable parameters (sigma, rho, beta).
    """

    params: L63Params

    def vector_field(
        self, t: float, state: PyTree, args: PyTree | None = None
    ) -> L63State:
        """Compute the Lorenz '63 tendency."""
        x_dot, y_dot, z_dot = rhs_lorenz_63(
            x=state.x,
            y=state.y,
            z=state.z,
            sigma=self.params.sigma,
            rho=self.params.rho,
            beta=self.params.beta,
        )
        return L63State(x=x_dot, y=y_dot, z=z_dot)

    def apply_boundary_conditions(self, state: PyTree) -> PyTree:
        """Identity — Lorenz '63 has no spatial boundaries."""
        return state

    def diagnose(self, state: PyTree) -> L63Diagnostics:
        """Compute energy diagnostic."""
        energy = 0.5 * (state.x**2 + state.y**2 + state.z**2)
        return L63Diagnostics(energy=energy)

    @staticmethod
    def create(
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8.0 / 3.0,
    ) -> Lorenz63:
        """Convenience factory with standard parameter defaults.

        Args:
            sigma: Prandtl number.
            rho: Rayleigh number.
            beta: Geometric factor.

        Returns:
            A ``Lorenz63`` model instance.
        """
        params = L63Params(
            sigma=jnp.array(sigma),
            rho=jnp.array(rho),
            beta=jnp.array(beta),
        )
        return Lorenz63(params=params)


def rhs_lorenz_63(
    x: Array,
    y: Array,
    z: Array,
    sigma: float = 10,
    rho: float = 28,
    beta: float = 2.667,
) -> tuple[Array, Array, Array]:
    """Pure-function RHS for the Lorenz '63 system."""
    x_dot = sigma * (y - x)
    y_dot = x * (rho - z) - y
    z_dot = x * y - beta * z
    return x_dot, y_dot, z_dot
