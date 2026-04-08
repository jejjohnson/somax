"""Forcing protocol for somax models."""

from __future__ import annotations

import abc

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array


class ForcingProtocol(eqx.Module):
    """Base class for forcing terms.

    Forcing objects are callable modules that return a forcing field
    given a time and grid. They compose with somax models via the
    ``forcing`` attribute.
    """

    @abc.abstractmethod
    def __call__(self, t: float, grid: eqx.Module) -> Array:
        """Evaluate forcing at time ``t`` on ``grid``."""
        ...


class ConstantForcing(ForcingProtocol):
    """Time-independent forcing field."""

    field: Array

    def __call__(self, t: float, grid: eqx.Module) -> Array:
        return self.field


class NoForcing(ForcingProtocol):
    """Zero forcing (free evolution)."""

    def __call__(self, t: float, grid: eqx.Module) -> Array:
        return jnp.float32(0.0)


class SeasonalWindForcing(ForcingProtocol):
    """Sinusoidal wind forcing with learnable amplitude.

    Produces ``tau0 * cos(omega * t + phase)`` where ``tau0`` is a
    differentiable amplitude field visible to ``jax.grad``.

    Args:
        tau0: Learnable amplitude array (visible to ``jax.grad``).
        omega: Angular frequency (static, e.g. ``2 * pi / T``).
        phase: Phase offset in radians (static).
    """

    tau0: Array
    omega: float = eqx.field(static=True)
    phase: float = eqx.field(static=True, default=0.0)

    def __call__(self, t: float, grid: eqx.Module) -> Array:
        """Evaluate seasonal forcing at time ``t``."""
        return self.tau0 * jnp.cos(self.omega * t + self.phase)


class InterpolatedForcing(ForcingProtocol):
    """Data-driven forcing via diffrax path interpolation.

    Wraps a ``diffrax.AbstractPath`` (e.g. ``LinearInterpolation``,
    ``CubicInterpolation``) to provide forcing from tabulated data.

    Args:
        path: A diffrax interpolation path built from data.
    """

    path: dfx.AbstractPath

    def __call__(self, t: float, grid: eqx.Module) -> Array:
        """Evaluate interpolated forcing at time ``t``."""
        return self.path.evaluate(t)

    @staticmethod
    def from_data(
        ts: Array,
        values: Array,
        method: str = "linear",
    ) -> InterpolatedForcing:
        """Build an ``InterpolatedForcing`` from tabulated data.

        Args:
            ts: Time coordinates with shape ``(T,)``.
            values: Forcing values with shape ``(T, ...)``.
            method: Interpolation method, ``"linear"`` (default) or
                ``"cubic"``.

        Returns:
            An ``InterpolatedForcing`` instance.
        """
        if method == "linear":
            path = dfx.LinearInterpolation(ts=ts, ys=values)
        elif method == "cubic":
            coeffs = dfx.backward_hermite_coefficients(ts=ts, ys=values)
            path = dfx.CubicInterpolation(ts=ts, coeffs=coeffs)
        else:
            msg = f"Unknown method {method!r}, expected 'linear' or 'cubic'"
            raise ValueError(msg)
        return InterpolatedForcing(path=path)
