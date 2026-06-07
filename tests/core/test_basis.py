"""Tests for the reduced-order basis forcing (somax._src.core.basis).

The headline test is the QG-parity check: a one-column ``BasisForcing`` lifted
by ``ForcingTerm`` reproduces the hand-written ``dq = dq + tau0 * wind_forcing``
tendency bit for bit, validating the design's central "the existing forcing is
the one-column special case" claim without any geonnax dependency.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from somax._src.models.qg.barotropic import BarotropicQGState
from somax.core import (
    BasisForcing,
    ConstantInTime,
    ForcingTerm,
    FourierInTime,
    SeasonalWindForcing,
    SpatialBasis,
    TransformedForcing,
    add_to,
    control_filter,
)


def _one_column_forcing(pattern, coeff):
    """A one-column BasisForcing: field(t) = coeff * pattern (constant in time)."""
    phi = pattern.reshape(-1)[:, None]
    spatial = SpatialBasis.from_array(phi)
    return BasisForcing(
        coeffs=jnp.array([coeff]),
        spatial=spatial,
        temporal=ConstantInTime(m=1),
        grid_shape=pattern.shape,
    )


class TestSpatialBasis:
    def test_synthesize_shape_and_value(self):
        phi = jnp.arange(12.0).reshape(6, 2)
        sb = SpatialBasis.from_array(phi)
        coeffs = jnp.array([1.0, -2.0])
        np.testing.assert_allclose(sb.synthesize(coeffs), phi @ coeffs, atol=1e-6)

    def test_default_std_is_ones(self):
        sb = SpatialBasis.from_array(jnp.ones((4, 3)))
        np.testing.assert_array_equal(sb.prior_std(), jnp.ones(3))

    def test_analyze_synthesize_roundtrip_orthonormal(self):
        # Orthonormal columns => analyze is the exact inverse of synthesize.
        q, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(0), (8, 3)))
        sb = SpatialBasis.from_array(q)
        coeffs = jnp.array([0.5, -1.0, 2.0])
        np.testing.assert_allclose(sb.analyze(sb.synthesize(coeffs)), coeffs, atol=1e-5)


class TestBasisForcing:
    def test_call_reshapes_to_grid(self):
        bf = _one_column_forcing(jnp.ones((5, 7)), coeff=3.0)
        out = bf(0.0)
        assert out.shape == (5, 7)
        np.testing.assert_allclose(out, 3.0, atol=1e-6)

    def test_whiten_sets_scaled_coeffs(self):
        phi = jnp.ones((6, 3))
        std = jnp.array([2.0, 0.5, 4.0])
        bf = BasisForcing(
            coeffs=jnp.zeros(3),
            spatial=SpatialBasis.from_array(phi, std=std),
            temporal=ConstantInTime(m=3),
            grid_shape=(2, 3),
        )
        u = jnp.array([1.0, -2.0, 0.5])
        whitened = bf.whiten(u)
        np.testing.assert_allclose(whitened.coeffs, std * u, atol=1e-6)
        # round-trip: dividing by std recovers u
        np.testing.assert_allclose(whitened.coeffs / std, u, atol=1e-6)

    def test_regularization_value(self):
        std = jnp.array([2.0, 0.5])
        coeffs = jnp.array([4.0, 1.0])
        bf = BasisForcing(
            coeffs=coeffs,
            spatial=SpatialBasis.from_array(jnp.ones((4, 2)), std=std),
            temporal=ConstantInTime(m=2),
            grid_shape=(2, 2),
        )
        expected = 0.5 * jnp.sum((coeffs / std) ** 2)
        np.testing.assert_allclose(bf.regularization(), expected, atol=1e-6)


class TestSeasonalWindParity:
    def test_one_mode_fourier_reproduces_seasonal_wind(self):
        # SeasonalWindForcing(tau0, omega, phase) = tau0 * cos(omega t + phase).
        tau0 = jax.random.normal(jax.random.PRNGKey(1), (4, 5))
        omega, phase = 2.0 * jnp.pi / 10.0, 0.3
        swf = SeasonalWindForcing(tau0=tau0, omega=omega, phase=phase)

        bf = BasisForcing(
            coeffs=jnp.array([1.0]),
            spatial=SpatialBasis.from_array(tau0.reshape(-1)[:, None]),
            temporal=FourierInTime(freqs=jnp.array([omega]), phases=jnp.array([phase])),
            grid_shape=tau0.shape,
        )
        for t in (0.0, 1.7, 5.0, 9.3):
            np.testing.assert_allclose(bf(t), swf(t, grid=None), atol=1e-6)


class TestForcingTermQGParity:
    def test_reproduces_qg_wind_tendency_bitwise(self):
        # QG barotropic adds  dq = dq + tau0 * wind_forcing  (qg/barotropic.py:146).
        Ny, Nx = 8, 8
        tau0 = 0.7
        wind_forcing = jax.random.normal(jax.random.PRNGKey(2), (Ny, Nx))

        # one-column BasisForcing reproduces  tau0 * wind_forcing
        forcing = _one_column_forcing(wind_forcing, coeff=tau0)
        term = ForcingTerm(forcing, place=add_to("q"))

        state = BarotropicQGState(q=jnp.zeros((Ny, Nx)))
        tendency = term(0.0, state)

        # hand-written QG line, exactly:
        expected_dq = jnp.zeros((Ny, Nx)) + tau0 * wind_forcing
        np.testing.assert_array_equal(tendency.q, expected_dq)

    def test_zeros_other_components(self):
        # ForcingTerm writes only the targeted component; the rest stays zero.
        state = BarotropicQGState(q=jnp.ones((4, 4)))
        forcing = _one_column_forcing(jnp.ones((4, 4)), coeff=1.0)
        term = ForcingTerm(forcing, place=add_to("q"))
        tendency = term(0.0, state)
        # only q is touched and it equals the field (not state.q + field, since
        # the term builds its own zeros tendency)
        np.testing.assert_allclose(tendency.q, jnp.ones((4, 4)), atol=1e-6)

    def test_layer_placement(self):
        # add_to("q", layer=0) targets a single layer of a stacked component.
        class LayeredState(eqx.Module):
            q: jnp.ndarray

        state = LayeredState(q=jnp.zeros((3, 4, 4)))
        forcing = _one_column_forcing(jnp.ones((4, 4)), coeff=2.0)
        term = ForcingTerm(forcing, place=add_to("q", layer=0))
        tendency = term(0.0, state)
        np.testing.assert_allclose(tendency.q[0], 2.0, atol=1e-6)
        assert bool((tendency.q[1:] == 0).all())


class TestControlScoping:
    def test_control_filter_selects_only_coeffs(self):
        bf = _one_column_forcing(jnp.ones((4, 4)), coeff=1.0)
        diff, _static = eqx.partition(bf, control_filter(bf))
        # coeffs survive in the differentiable partition; the dictionary does not
        assert diff.coeffs is not None
        assert diff.spatial.Phi is None
        assert diff.spatial.std is None

    def test_grad_only_flows_to_coeffs(self):
        bf = _one_column_forcing(
            jax.random.normal(jax.random.PRNGKey(3), (4, 4)), coeff=1.0
        )
        filt = control_filter(bf)
        diff, static = eqx.partition(bf, filt)

        def loss(diff):
            forcing = eqx.combine(diff, static)
            return jnp.sum(forcing(0.5) ** 2)

        grads = jax.grad(loss)(diff)
        assert grads.coeffs is not None
        assert bool(jnp.any(grads.coeffs != 0.0))
        assert grads.spatial.Phi is None  # never differentiated


class TestTransformedForcing:
    def test_inverse_applied(self):
        base = _one_column_forcing(jnp.ones((3, 3)), coeff=2.0)  # field = 2.0
        tf = TransformedForcing(base=base, inverse=lambda z: 10.0**z)
        np.testing.assert_allclose(tf(0.0), 10.0**2.0, atol=1e-4)
