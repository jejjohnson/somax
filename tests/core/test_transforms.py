"""Tests for StratificationProfile and ModalTransform."""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from finitevolx import build_coupling_matrix

from somax.core import ModalTransform, StratificationProfile


# MQGeometry 3-layer double-gyre stratification (Thiry et al. 2024) — unequal
# layer thicknesses, which is what made the coupling matrix non-symmetric and
# exposed the modal-decomposition bugs.
_MQG_H = (400.0, 1100.0, 2600.0)
_MQG_GP = (9.81, 0.025, 0.0125)
_MQG_F0 = 9.375e-5


# ---------------------------------------------------------------------------
# StratificationProfile
# ---------------------------------------------------------------------------


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
        assert jnp.allclose(strat.H, 1000.0)
        assert float(strat.total_depth) == 4000.0

    def test_from_N2_constant_top_gravity(self):
        strat = StratificationProfile.from_N2_constant(
            N2=1e-5, depth=4000.0, n_layers=4, g=9.81
        )
        assert jnp.allclose(strat.g_prime[0], 9.81, atol=1e-4)

    def test_from_N2_constant_internal_gravity(self):
        N2, depth, n_layers = 1e-5, 4000.0, 4
        strat = StratificationProfile.from_N2_constant(
            N2=N2, depth=depth, n_layers=n_layers
        )
        expected_g_prime = N2 * (depth / n_layers)
        assert jnp.allclose(strat.g_prime[1:], expected_g_prime)

    def test_from_N2_constant_density_monotonic(self):
        strat = StratificationProfile.from_N2_constant(
            N2=1e-5, depth=4000.0, n_layers=4
        )
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


# ---------------------------------------------------------------------------
# ModalTransform
# ---------------------------------------------------------------------------


class TestModalTransform:
    @pytest.fixture
    def two_layer_transform(self):
        """Standard 2-layer QG transform."""
        return ModalTransform.from_physics(
            H=(500.0, 4500.0),
            g_prime=(9.81, 0.025),
            f0=1e-4,
        )

    def test_shapes(self, two_layer_transform):
        t = two_layer_transform
        assert t.Cl2m.shape == (2, 2)
        assert t.Cm2l.shape == (2, 2)
        assert t.eigenvalues.shape == (2,)
        assert t.rossby_radii.shape == (2,)

    def test_rossby_radii_positive(self, two_layer_transform):
        assert jnp.all(two_layer_transform.rossby_radii > 0)

    def test_roundtrip(self, two_layer_transform):
        """to_modal then to_layer should recover the original field."""
        t = two_layer_transform
        x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        reconstructed = t.to_layer(t.to_modal(x))
        assert jnp.allclose(reconstructed, x, atol=1e-5)

    def test_inverse_roundtrip(self, two_layer_transform):
        """to_layer then to_modal should also be a roundtrip."""
        t = two_layer_transform
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        reconstructed = t.to_modal(t.to_layer(x))
        assert jnp.allclose(reconstructed, x, atol=1e-5)

    def test_projection_matrices_are_inverse(self, two_layer_transform):
        """Cl2m @ Cm2l should be identity."""
        t = two_layer_transform
        I = t.Cl2m @ t.Cm2l
        assert jnp.allclose(I, jnp.eye(2), atol=1e-5)

    def test_eigenvalues_sorted(self, two_layer_transform):
        """jnp.linalg.eigh returns sorted eigenvalues."""
        ev = two_layer_transform.eigenvalues
        assert jnp.all(ev[:-1] <= ev[1:])

    def test_three_layer(self):
        t = ModalTransform.from_physics(
            H=(300.0, 700.0, 4000.0),
            g_prime=(9.81, 0.025, 0.0125),
            f0=1e-4,
        )
        assert t.Cl2m.shape == (3, 3)
        x = jnp.ones((3, 5, 5))
        assert jnp.allclose(t.to_layer(t.to_modal(x)), x, atol=1e-5)

    def test_spatial_dimensions_preserved(self, two_layer_transform):
        """Transform works with 2D spatial fields (nl, Ny, Nx)."""
        t = two_layer_transform
        x = jnp.ones((2, 10, 15))
        modal = t.to_modal(x)
        assert modal.shape == (2, 10, 15)
        layer = t.to_layer(modal)
        assert layer.shape == (2, 10, 15)

    def test_rossby_radii_finite(self):
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

    def test_eigenvalues_non_negative(self):
        """Regression (bug 1): the layer coupling matrix is positive
        semi-definite, so every modal eigenvalue must be >= 0. A negative
        gravest eigenvalue (from a symmetric eigensolver's round-off on the
        non-symmetric A) flips the sign of the gravest-wavenumber denominators
        in the Helmholtz PV inversion and diverges multilayer QG within ~2 weeks.
        """
        mt = ModalTransform.from_physics(H=_MQG_H, g_prime=_MQG_GP, f0=_MQG_F0)
        assert jnp.all(mt.eigenvalues >= 0.0)

    def test_transform_diagonalizes_coupling_matrix(self):
        """Regression (bug 2): the modal transform must diagonalize the coupling
        matrix A. For unequal layer thicknesses A is NOT symmetric, so a
        symmetric eigensolver (``eigh``) returns eigenvectors that do not
        diagonalize A — the modal PV inversion then silently solves the wrong
        coupled elliptic problem (``Cl2m @ A @ Cm2l`` had off-diagonals as large
        as the eigenvalues themselves).
        """
        mt = ModalTransform.from_physics(H=_MQG_H, g_prime=_MQG_GP, f0=_MQG_F0)
        A = build_coupling_matrix(jnp.asarray(_MQG_H), jnp.asarray(_MQG_GP))
        D = mt.Cl2m @ A @ mt.Cm2l
        off_diagonal = D - jnp.diag(jnp.diagonal(D))
        assert jnp.max(jnp.abs(off_diagonal)) < 1e-6
        assert jnp.allclose(
            jnp.sort(jnp.diagonal(D)), jnp.sort(mt.eigenvalues), atol=1e-6
        )

    def test_eigenvalues_match_general_eigensolver(self):
        """Regression (bug 2): eigenvalues must match a *general* (non-symmetric)
        eigensolver on A, not the symmetric ``eigh`` which decomposes the wrong
        matrix for unequal layer thicknesses.
        """
        mt = ModalTransform.from_physics(H=_MQG_H, g_prime=_MQG_GP, f0=_MQG_F0)
        A = build_coupling_matrix(jnp.asarray(_MQG_H), jnp.asarray(_MQG_GP))
        ev_general = jnp.sort(jnp.real(jnp.linalg.eigvals(A)))
        assert jnp.allclose(jnp.sort(mt.eigenvalues), ev_general, atol=1e-6)

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
