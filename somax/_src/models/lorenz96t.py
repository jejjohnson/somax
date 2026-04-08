"""Lorenz '96 two-tier (slow-fast) coupled system."""

from __future__ import annotations

from typing import Any

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PyTree

from somax._src.core.model import SomaxModel
from somax._src.core.types import Params, State


class L96TParams(Params):
    """Differentiable parameters for the two-tier Lorenz '96 system.

    Args:
        F: Forcing amplitude.
        h: Coupling coefficient.
        b: Ratio of amplitudes.
        c: Time-scale ratio.
    """

    F: Array
    h: Array
    b: Array
    c: Array


class L96TState(State):
    """State vector for the two-tier Lorenz '96 system.

    Args:
        x: Slow variables with shape ``(Dx,)``.
        y: Fast variables with shape ``(Dy * Dx,)``.
    """

    x: Array
    y: Array

    @classmethod
    def init_state(
        cls,
        ndims: tuple[int, ...] | int = (10, 20),
        noise: tuple[float, ...] | float = 0.01,
        batchsize: int = 1,
        b: float = 10.0,
        key: jax.Array | None = None,
    ) -> L96TState:
        """Create an initial state with random perturbations.

        Args:
            ndims: Tuple of (Dx, Dy) dimensions.
            noise: Perturbation amplitude(s).
            batchsize: Number of ensemble members.
            b: Amplitude ratio (scales slow variables).
            key: PRNG key for reproducibility.

        Returns:
            An ``L96TState`` instance.
        """
        if key is None:
            key = jrandom.PRNGKey(123)
        noise_t: tuple[float, ...] = check_dims(value=noise, ndim=2, name="noise")
        ndims_t: tuple[int, ...] = check_dims(value=ndims, ndim=2, name="ndims")

        keyx, keyy = jrandom.split(key=key, num=2)
        if batchsize > 1:
            x0 = (
                b * noise_t[0] * jrandom.normal(key=keyx, shape=(batchsize, ndims_t[0]))
            )
            y0 = noise_t[1] * jrandom.normal(
                key=keyy, shape=(batchsize, ndims_t[1] * ndims_t[0])
            )
        else:
            x0 = b * noise_t[0] * jrandom.normal(key=keyx, shape=(ndims_t[0],))
            y0 = noise_t[1] * jrandom.normal(key=keyy, shape=(ndims_t[1] * ndims_t[0],))

        return cls(x=x0, y=y0)


class Lorenz96t(SomaxModel):
    """Lorenz '96 two-tier (slow-fast) coupled system.

    Slow variables X couple to fast variables Y::

        dX_k/dt = (X_{k+1} - X_{k-2}) * X_{k-1} - X_k + F - (hc/b) * sum_j(Y_{j,k})
        dY_{j,k}/dt = cb * (Y_{j+1} - Y_{j-2}) * Y_{j-1} - cY + (hc/b) * X_k

    Args:
        params: Differentiable parameters (F, h, b, c).
        advection: Whether to include nonlinear advection terms.
    """

    params: L96TParams
    advection: bool = eqx.field(default=True, static=True)

    def vector_field(
        self, t: float, state: PyTree, args: PyTree | None = None
    ) -> L96TState:
        """Compute the two-tier Lorenz '96 tendency."""
        x_dot, y_dot = rhs_lorenz_96t(
            x=state.x,
            y=state.y,
            F=self.params.F,
            h=self.params.h,
            c=self.params.c,
            b=self.params.b,
            advection=self.advection,
            return_coupling=False,
        )
        return L96TState(x=x_dot, y=y_dot)

    def apply_boundary_conditions(self, state: PyTree) -> PyTree:
        """Identity — periodicity is implicit via jnp.roll."""
        return state

    @staticmethod
    def create(
        F: float = 18.0,
        h: float = 1.0,
        b: float = 10.0,
        c: float = 10.0,
        advection: bool = True,
    ) -> Lorenz96t:
        """Convenience factory with standard defaults.

        Args:
            F: Forcing amplitude.
            h: Coupling coefficient.
            b: Amplitude ratio.
            c: Time-scale ratio.
            advection: Include nonlinear advection.

        Returns:
            A ``Lorenz96t`` model instance.
        """
        params = L96TParams(
            F=jnp.array(F),
            h=jnp.array(h),
            b=jnp.array(b),
            c=jnp.array(c),
        )
        return Lorenz96t(params=params, advection=advection)


def rhs_lorenz_96t(
    x: Array,
    y: Array,
    F: Array | float = 18.0,
    h: Array | float = 1.0,
    b: Array | float = 10.0,
    c: Array | float = 10.0,
    advection: bool = True,
    return_coupling: bool = True,
) -> tuple[Array, Array] | tuple[Array, Array, Array]:
    """Pure-function RHS for the two-tier Lorenz '96 system."""
    x_dims = x.shape[0]
    xy_dims = y.shape[0]
    y_dims = xy_dims // x_dims

    assert xy_dims == x_dims * y_dims, "X n Y have incompatible dims"

    hcb = (h * c) / b

    # Slow variable tendency
    x_minus_1 = jnp.roll(x, 1)
    x_plus_1 = jnp.roll(x, -1)
    x_minus_2 = jnp.roll(x, 2)

    x_advection = x_minus_1 * (x_plus_1 - x_minus_2)

    y_summed = einops.rearrange(y, "(Dy Dx) -> Dy Dx", Dy=y_dims, Dx=x_dims)
    y_summed = einops.reduce(y_summed, "Dy Dx -> Dx", reduction="sum")

    if advection:
        x_rhs = x_advection - x + F - hcb * y_summed
    else:
        x_rhs = -x + F - hcb * y_summed

    # Fast variable tendency — reshape to (Dy, Dx), roll along j-axis only
    y_2d = einops.rearrange(y, "(Dy Dx) -> Dy Dx", Dy=y_dims, Dx=x_dims)
    y_j_plus_1 = jnp.roll(y_2d, -1, axis=0)
    y_j_minus_1 = jnp.roll(y_2d, 1, axis=0)
    y_j_minus_2 = jnp.roll(y_2d, 2, axis=0)

    y_advection_2d = (y_j_plus_1 - y_j_minus_2) * y_j_minus_1
    x_broadcast = einops.repeat(x, "Dx -> Dy Dx", Dy=y_dims)

    y_rhs_2d = -b * c * y_advection_2d - c * y_2d + hcb * x_broadcast
    y_rhs = einops.rearrange(y_rhs_2d, "Dy Dx -> (Dy Dx)")

    if return_coupling:
        return x_rhs, y_rhs, -hcb * y_summed
    else:
        return x_rhs, y_rhs


def check_dims(value: Any, ndim: int, name: str) -> tuple:
    """Normalize scalar or tuple inputs to a tuple of length ``ndim``.

    Only Python scalars and tuples are accepted. JAX arrays are not
    supported because shape parameters must be concrete (not traced).
    """
    if isinstance(value, (int, float)):
        return (value,) * ndim
    elif isinstance(value, tuple):
        assert len(value) == ndim, f"{name} must be a tuple of length {ndim}"
        return tuple(value)
    msg = f"Expected int, float, or tuple for {name}, got {type(value).__name__}"
    raise TypeError(msg)
