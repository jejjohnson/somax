"""Core type definitions for somax models."""

from __future__ import annotations

import equinox as eqx


class State(eqx.Module):
    """Base class for model state vectors.

    All model states should subclass this to enable interoperability
    with the somax model contract and JAX transformations.
    """


class Params(eqx.Module):
    """Base class for differentiable model parameters.

    Fields on Params subclasses are visible to ``jax.grad`` by default.
    Use ``eqx.field(static=True)`` for non-differentiable parameters.
    """


class PhysConsts(eqx.Module):
    """Base class for frozen physical constants.

    All fields should be marked ``static=True`` so they are invisible
    to ``jax.grad`` and treated as compile-time constants.
    """


class Diagnostics(eqx.Module):
    """Base class for on-demand diagnostic quantities.

    Computed from a model state via ``model.diagnose(state)``.
    """
