"""Forcing protocol for somax models."""

from __future__ import annotations

import abc

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
