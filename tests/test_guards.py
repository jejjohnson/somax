"""Unit tests for somax.guards in-JIT fail-fast tripwires.

Cheap, no integration — these run in the fast PR lane.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import pytest

from somax.guards import guard_ceiling, guard_finite, guard_positive


class TestGuardFinite:
    def test_passes_finite_unchanged(self) -> None:
        x = jnp.arange(5.0)
        out = guard_finite(x, where="t")
        assert jnp.array_equal(out, x)

    def test_raises_on_nan_under_jit(self) -> None:
        f = eqx.filter_jit(lambda x: guard_finite(x, where="t"))
        bad = jnp.array([1.0, jnp.nan, 3.0])
        with pytest.raises(eqx.EquinoxRuntimeError):
            f(bad)

    def test_raises_on_inf_under_jit(self) -> None:
        f = eqx.filter_jit(lambda x: guard_finite(x, where="t"))
        with pytest.raises(eqx.EquinoxRuntimeError):
            f(jnp.array([jnp.inf]))


class TestGuardPositive:
    def test_passes_positive_unchanged(self) -> None:
        x = jnp.array([0.1, 1.0, 2.0])
        assert jnp.array_equal(guard_positive(x, where="h"), x)

    def test_raises_on_zero_under_jit(self) -> None:
        f = eqx.filter_jit(lambda x: guard_positive(x, where="h"))
        with pytest.raises(eqx.EquinoxRuntimeError):
            f(jnp.array([1.0, 0.0, 2.0]))

    def test_raises_on_negative_under_jit(self) -> None:
        f = eqx.filter_jit(lambda x: guard_positive(x, where="h"))
        with pytest.raises(eqx.EquinoxRuntimeError):
            f(jnp.array([-0.5, 1.0]))

    def test_custom_floor(self) -> None:
        f = eqx.filter_jit(lambda x: guard_positive(x, where="h", floor=1.0))
        # 0.5 <= floor 1.0 -> raises
        with pytest.raises(eqx.EquinoxRuntimeError):
            f(jnp.array([0.5, 2.0]))


class TestGuardCeiling:
    def test_passes_below_ceiling(self) -> None:
        x = jnp.array([1.0, -2.0, 3.0])
        assert jnp.array_equal(guard_ceiling(x, where="u", ceil=10.0), x)

    def test_raises_above_ceiling_under_jit(self) -> None:
        f = eqx.filter_jit(lambda x: guard_ceiling(x, where="u", ceil=5.0))
        with pytest.raises(eqx.EquinoxRuntimeError):
            f(jnp.array([1.0, -9.0]))
