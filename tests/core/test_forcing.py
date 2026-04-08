"""Tests for the forcing protocol."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from somax.core import (
    ConstantForcing,
    ForcingProtocol,
    InterpolatedForcing,
    NoForcing,
    SeasonalWindForcing,
)


class DummyGrid(eqx.Module):
    """Minimal grid for testing."""

    nx: int = 10


# ---------------------------------------------------------------------------
# Existing forcing tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# SeasonalWindForcing
# ---------------------------------------------------------------------------


class TestSeasonalWindForcing:
    def test_amplitude_at_t0(self):
        """At t=0 with phase=0, cos(0)=1, so result = tau0."""
        tau0 = jnp.array([1.0, 2.0, 3.0])
        forcing = SeasonalWindForcing(tau0=tau0, omega=2.0 * jnp.pi)
        result = forcing(t=0.0, grid=DummyGrid())
        assert jnp.allclose(result, tau0)

    def test_varies_with_time(self):
        """Result changes between t=0 and t=T/4."""
        tau0 = jnp.array(5.0)
        omega = 2.0 * jnp.pi  # period = 1
        forcing = SeasonalWindForcing(tau0=tau0, omega=omega)
        r0 = forcing(t=0.0, grid=DummyGrid())
        r_quarter = forcing(t=0.25, grid=DummyGrid())
        # cos(0) = 1, cos(pi/2) = 0
        assert jnp.isclose(r0, 5.0)
        assert jnp.isclose(r_quarter, 0.0, atol=1e-6)

    def test_phase_offset(self):
        """Phase shifts the cosine."""
        tau0 = jnp.array(1.0)
        forcing = SeasonalWindForcing(tau0=tau0, omega=2.0 * jnp.pi, phase=jnp.pi / 2)
        # cos(0 + pi/2) = 0
        result = forcing(t=0.0, grid=DummyGrid())
        assert jnp.isclose(result, 0.0, atol=1e-6)

    def test_grad_through_tau0(self):
        """jax.grad flows through the learnable tau0 amplitude."""

        def loss(forcing):
            return jnp.sum(forcing(t=0.0, grid=DummyGrid()) ** 2)

        tau0 = jnp.array([1.0, 2.0])
        forcing = SeasonalWindForcing(tau0=tau0, omega=1.0)
        grads = jax.grad(loss)(forcing)
        # d/d(tau0) sum((tau0 * cos(0))^2) = d/d(tau0) sum(tau0^2) = 2*tau0
        assert jnp.allclose(grads.tau0, 2.0 * tau0)

    def test_is_forcing_protocol(self):
        forcing = SeasonalWindForcing(tau0=jnp.array(1.0), omega=1.0)
        assert isinstance(forcing, ForcingProtocol)


# ---------------------------------------------------------------------------
# InterpolatedForcing
# ---------------------------------------------------------------------------


class TestInterpolatedForcing:
    def test_evaluates_at_known_points(self):
        """Interpolation matches exact values at data points."""
        ts = jnp.array([0.0, 1.0, 2.0])
        values = jnp.array([0.0, 10.0, 20.0])
        forcing = InterpolatedForcing.from_data(ts, values)
        assert jnp.isclose(forcing(t=0.0, grid=DummyGrid()), 0.0)
        assert jnp.isclose(forcing(t=1.0, grid=DummyGrid()), 10.0)
        assert jnp.isclose(forcing(t=2.0, grid=DummyGrid()), 20.0)

    def test_interpolates_between_points(self):
        """Linear interpolation at midpoint."""
        ts = jnp.array([0.0, 2.0])
        values = jnp.array([0.0, 10.0])
        forcing = InterpolatedForcing.from_data(ts, values)
        result = forcing(t=1.0, grid=DummyGrid())
        assert jnp.isclose(result, 5.0)

    def test_multidimensional_values(self):
        """Interpolation works with array-valued data."""
        ts = jnp.array([0.0, 1.0])
        values = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        forcing = InterpolatedForcing.from_data(ts, values)
        result = forcing(t=0.5, grid=DummyGrid())
        assert jnp.allclose(result, jnp.array([2.0, 3.0]))

    def test_is_forcing_protocol(self):
        ts = jnp.array([0.0, 1.0])
        values = jnp.array([0.0, 1.0])
        forcing = InterpolatedForcing.from_data(ts, values)
        assert isinstance(forcing, ForcingProtocol)

    def test_cubic_at_known_points(self):
        """Cubic interpolation matches exact values at knot points."""
        ts = jnp.array([0.0, 1.0, 2.0, 3.0])
        values = jnp.array([0.0, 1.0, 4.0, 9.0])
        forcing = InterpolatedForcing.from_data(ts, values, method="cubic")
        for i, t in enumerate(ts):
            result = forcing(t=float(t), grid=DummyGrid())
            assert jnp.isclose(result, values[i], atol=1e-5), (
                f"Cubic mismatch at t={t}: got {result}, expected {values[i]}"
            )

    def test_cubic_midpoint_smoother_than_linear(self):
        """Cubic interpolation of x^2 at midpoint is closer to true value."""
        ts = jnp.array([0.0, 1.0, 2.0, 3.0])
        values = ts**2  # [0, 1, 4, 9]
        cubic = InterpolatedForcing.from_data(ts, values, method="cubic")
        linear = InterpolatedForcing.from_data(ts, values, method="linear")
        # True value at t=1.5 is 2.25
        cubic_val = cubic(t=1.5, grid=DummyGrid())
        linear_val = linear(t=1.5, grid=DummyGrid())
        true_val = 1.5**2
        assert abs(float(cubic_val) - true_val) <= abs(float(linear_val) - true_val)

    def test_cubic_multidimensional(self):
        """Cubic interpolation works with array-valued data."""
        ts = jnp.array([0.0, 1.0, 2.0, 3.0])
        values = jnp.stack([ts, ts**2], axis=-1)  # (4, 2)
        forcing = InterpolatedForcing.from_data(ts, values, method="cubic")
        result = forcing(t=1.0, grid=DummyGrid())
        assert jnp.allclose(result, jnp.array([1.0, 1.0]), atol=1e-5)

    def test_invalid_method_raises(self):
        ts = jnp.array([0.0, 1.0])
        values = jnp.array([0.0, 1.0])
        with pytest.raises(ValueError, match="Unknown method"):
            InterpolatedForcing.from_data(ts, values, method="spline")
