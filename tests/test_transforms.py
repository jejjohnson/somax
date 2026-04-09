"""Tests for StratificationProfile and ModalTransform."""

from __future__ import annotations

import jax.numpy as jnp

from somax.core import ModalTransform, StratificationProfile


class TestStratificationProfile:
    def test_from_N2_constant_shapes(self):
        strat = StratificationProfile.from_N2_constant(
            N2=1e-5, depth=4000.0, n_layers=4
        )
        assert strat.H.shape == (4,)
        assert strat.g_prime.shape == (4,)
        assert strat.rho is not None
        assert strat.rho.shape == (4,)
        assert strat.nl == 4

    def test_from_N2_constant_uniform_layers(self):
        strat = StratificationProfile.from_N2_constant(
            N2=1e-5, depth=4000.0, n_layers=4
        )
        # All layers should have equal thickness
        assert jnp.allclose(strat.H, 1000.0)
        assert float(strat.total_depth) == 4000.0

    def test_from_N2_constant_top_gravity(self):
        strat = StratificationProfile.from_N2_constant(
            N2=1e-5, depth=4000.0, n_layers=4, g=9.81
        )
        # Top interface should be full gravity
        assert jnp.allclose(strat.g_prime[0], 9.81, atol=1e-4)

    def test_from_N2_constant_internal_gravity(self):
        N2 = 1e-5
        depth = 4000.0
        n_layers = 4
        strat = StratificationProfile.from_N2_constant(
            N2=N2, depth=depth, n_layers=n_layers
        )
        H_val = depth / n_layers
        expected_g_prime = N2 * H_val
        assert jnp.allclose(strat.g_prime[1:], expected_g_prime)

    def test_from_N2_constant_density_monotonic(self):
        strat = StratificationProfile.from_N2_constant(
            N2=1e-5, depth=4000.0, n_layers=4
        )
        # Density should increase with depth (stable stratification)
        assert jnp.all(jnp.diff(strat.rho) > 0)

    def test_from_N2_exponential_shapes(self):
        strat = StratificationProfile.from_N2_exponential(
            N2_surface=1e-4, scale_depth=500.0, depth=4000.0, n_layers=4
        )
        assert strat.H.shape == (4,)
        assert strat.g_prime.shape == (4,)
        assert strat.nl == 4

    def test_from_N2_exponential_decreasing_g_prime(self):
        strat = StratificationProfile.from_N2_exponential(
            N2_surface=1e-4, scale_depth=500.0, depth=4000.0, n_layers=4
        )
        # g_prime should decrease with depth (stronger stratification near surface)
        assert jnp.all(jnp.diff(strat.g_prime[1:]) < 0)

    def test_from_layers(self):
        strat = StratificationProfile.from_layers(
            H=[400.0, 1100.0, 2500.0],
            g_prime=[9.81, 0.025, 0.0125],
        )
        assert strat.nl == 3
        assert float(strat.total_depth) == 4000.0
        assert strat.rho is None

    def test_from_layers_with_rho(self):
        strat = StratificationProfile.from_layers(
            H=[400.0, 1100.0, 2500.0],
            g_prime=[9.81, 0.025, 0.0125],
            rho=[1025.0, 1027.5, 1028.0],
        )
        assert strat.rho is not None
        assert strat.rho.shape == (3,)

    def test_single_layer(self):
        strat = StratificationProfile.from_N2_constant(
            N2=1e-5, depth=1000.0, n_layers=1
        )
        assert strat.nl == 1
        assert strat.H.shape == (1,)


class TestModalTransform:
    def test_from_physics_shapes(self):
        mt = ModalTransform.from_physics(
            H=(500.0, 500.0), g_prime=(0.02, 0.02), f0=1e-4
        )
        assert mt.Cl2m.shape == (2, 2)
        assert mt.Cm2l.shape == (2, 2)
        assert mt.eigenvalues.shape == (2,)
        assert mt.rossby_radii.shape == (2,)

    def test_roundtrip_identity(self):
        mt = ModalTransform.from_physics(
            H=(400.0, 1100.0, 2500.0),
            g_prime=(9.81, 0.025, 0.0125),
            f0=1e-4,
        )
        x = jnp.array([1.0, 2.0, 3.0])
        roundtrip = mt.to_layer(mt.to_modal(x))
        assert jnp.allclose(roundtrip, x, atol=1e-5)

    def test_roundtrip_2d_fields(self):
        mt = ModalTransform.from_physics(
            H=(500.0, 500.0), g_prime=(0.02, 0.02), f0=1e-4
        )
        x = jnp.ones((2, 10, 10))
        roundtrip = mt.to_layer(mt.to_modal(x))
        assert jnp.allclose(roundtrip, x, atol=1e-5)

    def test_rossby_radii_finite(self):
        """Rossby radii should be finite and positive for baroclinic modes."""
        mt = ModalTransform.from_physics(
            H=(500.0, 500.0), g_prime=(0.02, 0.02), f0=1e-4
        )
        finite_radii = mt.rossby_radii[jnp.isfinite(mt.rossby_radii)]
        assert finite_radii.size > 0
        assert jnp.all(finite_radii > 0)

    def test_eigenvalues_real(self):
        mt = ModalTransform.from_physics(
            H=(400.0, 1100.0, 2500.0),
            g_prime=(9.81, 0.025, 0.0125),
            f0=1e-4,
        )
        assert jnp.all(jnp.isfinite(mt.eigenvalues))

    def test_from_stratification(self):
        strat = StratificationProfile.from_layers(
            H=[400.0, 1100.0, 2500.0],
            g_prime=[9.81, 0.025, 0.0125],
        )
        mt = ModalTransform.from_stratification(strat, f0=1e-4)
        assert mt.Cl2m.shape == (3, 3)
        x = jnp.array([1.0, 2.0, 3.0])
        roundtrip = mt.to_layer(mt.to_modal(x))
        assert jnp.allclose(roundtrip, x, atol=1e-5)

    def test_single_layer(self):
        mt = ModalTransform.from_physics(H=(1000.0,), g_prime=(9.81,), f0=1e-4)
        assert mt.Cl2m.shape == (1, 1)
        x = jnp.array([5.0])
        assert jnp.allclose(mt.to_layer(mt.to_modal(x)), x)
