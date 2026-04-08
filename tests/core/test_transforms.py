"""Tests for ModalTransform (precomputed layer-to-mode transforms)."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from somax._src.core.transforms import ModalTransform, _build_coupling_matrix


# ---------------------------------------------------------------------------
# Coupling matrix
# ---------------------------------------------------------------------------


class TestBuildCouplingMatrix:
    def test_shape(self):
        A = _build_coupling_matrix(H=(500.0, 500.0), g_prime=(9.81, 0.02))
        assert A.shape == (2, 2)

    def test_symmetric(self):
        """Coupling matrix should be symmetric for uniform layer depths."""
        A = _build_coupling_matrix(H=(500.0, 500.0, 500.0), g_prime=(9.81, 0.02, 0.01))
        assert jnp.allclose(A, A.T, atol=1e-12)

    def test_single_layer(self):
        """Single-layer case: A is a 1x1 matrix."""
        A = _build_coupling_matrix(H=(1000.0,), g_prime=(9.81,))
        assert A.shape == (1, 1)
        expected = -1.0 / (1000.0 * 9.81)
        assert jnp.isclose(A[0, 0], expected)


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
        x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
        reconstructed = t.to_layer(t.to_modal(x))
        assert jnp.allclose(reconstructed, x, atol=1e-10)

    def test_inverse_roundtrip(self, two_layer_transform):
        """to_layer then to_modal should also be a roundtrip."""
        t = two_layer_transform
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        reconstructed = t.to_modal(t.to_layer(x))
        assert jnp.allclose(reconstructed, x, atol=1e-10)

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
        """Verify 3-layer construction and roundtrip."""
        t = ModalTransform.from_physics(
            H=(300.0, 700.0, 4000.0),
            g_prime=(9.81, 0.025, 0.0125),
            f0=1e-4,
        )
        assert t.Cl2m.shape == (3, 3)
        x = jnp.ones((3, 5, 5))
        assert jnp.allclose(t.to_layer(t.to_modal(x)), x, atol=1e-10)

    def test_spatial_dimensions_preserved(self, two_layer_transform):
        """Transform works with 2D spatial fields (nl, Ny, Nx)."""
        t = two_layer_transform
        x = jnp.ones((2, 10, 15))
        modal = t.to_modal(x)
        assert modal.shape == (2, 10, 15)
        layer = t.to_layer(modal)
        assert layer.shape == (2, 10, 15)
