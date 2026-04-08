"""Tests for the forcing protocol."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import pytest

from somax.core import ConstantForcing, ForcingProtocol, NoForcing


class DummyGrid(eqx.Module):
    """Minimal grid for testing."""

    nx: int = 10


def test_constant_forcing():
    field = jnp.ones(5) * 3.0
    forcing = ConstantForcing(field=field)
    result = forcing(t=0.0, grid=DummyGrid())
    assert jnp.array_equal(result, field)


def test_constant_forcing_ignores_time():
    field = jnp.array([1.0, 2.0])
    forcing = ConstantForcing(field=field)
    r1 = forcing(t=0.0, grid=DummyGrid())
    r2 = forcing(t=100.0, grid=DummyGrid())
    assert jnp.array_equal(r1, r2)


def test_no_forcing():
    forcing = NoForcing()
    result = forcing(t=0.0, grid=DummyGrid())
    assert result == 0.0


def test_forcing_protocol_is_abstract():
    with pytest.raises(TypeError):
        ForcingProtocol()
