"""Tests for the spherical barotropic QG model."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from finitevolx import SphericalDivergence2D

from somax._src.models.spherical import SphericalQG, SphericalQGState


RADIUS = 6.371e6
OMEGA_EARTH = 7.292e-5
INTERIOR = (slice(1, -1), slice(1, -1))


def build(nx=48, ny=24, **kw):
    defaults = {
        "lon_range": (0.0, 360.0),
        "lat_range": (-80.0, 80.0),
        "radius": RADIUS,
        "omega": OMEGA_EARTH,
        "wind_profile": "none",
    }
    return SphericalQG.create(nx=nx, ny=ny, **{**defaults, **kw})


def smooth_streamfunction(model, amplitude=1.0e7):
    """A smooth field vanishing on the polar rows, as the solve assumes."""
    lat = np.asarray(model.grid.lat_T)
    lon = np.asarray(model.grid.lon_T)
    field = amplitude * np.sin(2.0 * lon) * np.cos(lat) ** 2
    return model._expand(jnp.asarray(field[1:-1, 1:-1]))


def zonal_streamfunction(model, amplitude=1.0e7):
    """Zonally symmetric: depends on latitude only."""
    lat = np.asarray(model.grid.lat_T)
    field = amplitude * np.cos(np.pi * lat / (2.0 * np.deg2rad(80.0)))
    return model._expand(jnp.asarray(field[1:-1, 1:-1]))


class TestPVInversion:
    def test_recovers_a_known_streamfunction(self):
        model = build()
        psi_true = smooth_streamfunction(model)
        recovered = model.invert_pv(model.laplacian(psi_true))
        error = float(
            jnp.abs(recovered[INTERIOR] - psi_true[INTERIOR]).max()
            / jnp.abs(psi_true[INTERIOR]).max()
        )
        assert error < 1e-5

    def test_round_trips_through_the_laplacian(self):
        model = build()
        rng = np.random.RandomState(0)
        q = model._expand(
            jnp.asarray(1e-5 * rng.randn(model.grid.Ny - 2, model.grid.Nx - 2))
        )
        residual = model.laplacian(model.invert_pv(q)) - q
        relative = float(jnp.abs(residual[INTERIOR]).max() / jnp.abs(q[INTERIOR]).max())
        assert relative < 1e-4

    def test_zero_vorticity_gives_zero_streamfunction(self):
        """The normalisation divides by the residual norm; zero must be safe."""
        model = build()
        zeros = jnp.zeros((model.grid.Ny, model.grid.Nx))
        psi = model.invert_pv(zeros)
        assert np.isfinite(np.asarray(psi)).all()
        np.testing.assert_array_equal(np.asarray(psi), 0.0)

    def test_is_linear(self):
        model = build()
        psi_a = smooth_streamfunction(model)
        psi_b = zonal_streamfunction(model)
        q_a, q_b = model.laplacian(psi_a), model.laplacian(psi_b)
        combined = model.invert_pv(q_a + 3.0 * q_b)
        separate = model.invert_pv(q_a) + 3.0 * model.invert_pv(q_b)
        # Measured against the field magnitude, not pointwise: each call
        # is a separate CG solve converged to cg_tol, so the two routes
        # differ at roughly that level and pointwise ratios blow up
        # wherever the streamfunction passes through zero.
        magnitude = float(jnp.abs(separate[INTERIOR]).max())
        difference = float(jnp.abs(combined[INTERIOR] - separate[INTERIOR]).max())
        assert difference / magnitude < 1e-3

    def test_streamfunction_vanishes_on_the_polar_rows(self):
        """The Dirichlet condition that makes the problem well posed."""
        model = build()
        psi = model.invert_pv(model.laplacian(smooth_streamfunction(model)))
        np.testing.assert_array_equal(np.asarray(psi)[0, :], 0.0)
        np.testing.assert_array_equal(np.asarray(psi)[-1, :], 0.0)

    def test_scales_with_the_planet_radius(self):
        """psi ~ q R^2, so a bigger planet gives a bigger streamfunction."""
        small = build(radius=1.0e6)
        large = build(radius=4.0e6)
        rng = np.random.RandomState(1)
        q = jnp.asarray(1e-5 * rng.randn(small.grid.Ny, small.grid.Nx))
        q = small._expand(q[INTERIOR])
        ratio = float(
            jnp.abs(large.invert_pv(q)).max() / jnp.abs(small.invert_pv(q)).max()
        )
        assert ratio == pytest.approx(16.0, rel=0.05)


class TestVelocities:
    def test_are_non_divergent(self):
        """A streamfunction flow has no divergence, up to discretisation."""
        model = build()
        psi = smooth_streamfunction(model)
        u, v = model.velocities(psi)
        divergence = SphericalDivergence2D(grid=model.grid)(u, v)
        inner = (slice(2, -2), slice(2, -2))
        speed = float(jnp.maximum(jnp.abs(u).max(), jnp.abs(v).max()))
        scale = speed / float(model.grid.R * model.grid.dlat)
        assert float(jnp.abs(divergence[inner]).max()) < 0.2 * scale

    def test_a_zonal_streamfunction_gives_no_meridional_flow(self):
        model = build()
        _, v = model.velocities(zonal_streamfunction(model))
        assert float(jnp.abs(v[INTERIOR]).max()) == pytest.approx(0.0, abs=1e-9)

    def test_sign_convention_is_the_usual_one(self):
        """u = -dpsi/dy: a poleward-increasing psi gives westward flow."""
        model = build()
        lat = np.asarray(model.grid.lat_T)
        psi = model._expand(jnp.asarray((1.0e7 * lat)[1:-1, 1:-1]))
        u, _ = model.velocities(psi)
        assert float(jnp.mean(u[2:-2, 2:-2])) < 0.0


class TestCoriolisAndBeta:
    def test_f_is_the_full_latitudinal_sine(self):
        model = build()
        np.testing.assert_allclose(
            model.f_field, 2.0 * OMEGA_EARTH * jnp.sin(model.grid.lat_T), rtol=1e-6
        )

    def test_a_zonally_symmetric_state_is_steady(self):
        """No meridional flow and no zonal structure means no advection.

        This is what confirms beta enters through advection of absolute
        vorticity rather than through a hard-coded constant.
        """
        model = build()
        psi = zonal_streamfunction(model)
        state = model.apply_boundary_conditions(
            SphericalQGState(q=model.laplacian(psi))
        )
        tendency = model.vector_field(0.0, state)
        inner = (slice(2, -2), slice(2, -2))
        vorticity_scale = float(jnp.abs(state.q[inner]).max())
        rate = vorticity_scale * 2.0 * OMEGA_EARTH
        assert float(jnp.abs(tendency.q[inner]).max()) < 0.05 * rate

    def test_a_zonally_varying_state_is_not_steady(self):
        """Otherwise the steadiness test above would be vacuous."""
        model = build()
        state = model.apply_boundary_conditions(
            SphericalQGState(q=model.laplacian(smooth_streamfunction(model)))
        )
        tendency = model.vector_field(0.0, state)
        inner = (slice(2, -2), slice(2, -2))
        assert float(jnp.abs(tendency.q[inner]).max()) > 0.0


class TestBoundaryConditions:
    def test_longitude_wraps(self):
        model = build()
        rng = np.random.RandomState(2)
        state = model.apply_boundary_conditions(
            SphericalQGState(q=jnp.asarray(rng.randn(model.grid.Ny, model.grid.Nx)))
        )
        np.testing.assert_allclose(state.q[:, 0], state.q[:, -2])
        np.testing.assert_allclose(state.q[:, -1], state.q[:, 1])

    def test_vorticity_is_zero_on_the_polar_rows(self):
        model = build()
        rng = np.random.RandomState(3)
        state = model.apply_boundary_conditions(
            SphericalQGState(q=jnp.asarray(rng.randn(model.grid.Ny, model.grid.Nx)))
        )
        np.testing.assert_array_equal(np.asarray(state.q)[0, :], 0.0)
        np.testing.assert_array_equal(np.asarray(state.q)[-1, :], 0.0)


class TestDiagnostics:
    def test_invariants_are_advertised(self):
        model = build()
        state = SphericalQGState(q=model.laplacian(smooth_streamfunction(model)))
        assert set(model.diagnose(state).invariants()) == {
            "kinetic_energy",
            "enstrophy",
            "circulation",
        }

    def test_energy_and_enstrophy_are_positive(self):
        model = build()
        state = SphericalQGState(q=model.laplacian(smooth_streamfunction(model)))
        diagnostics = model.diagnose(state)
        assert float(diagnostics.kinetic_energy) > 0.0
        assert float(diagnostics.enstrophy) > 0.0

    def test_a_state_at_rest_has_no_energy(self):
        model = build()
        zeros = jnp.zeros((model.grid.Ny, model.grid.Nx))
        diagnostics = model.diagnose(SphericalQGState(q=zeros))
        assert float(diagnostics.kinetic_energy) == pytest.approx(0.0)
        assert float(diagnostics.enstrophy) == pytest.approx(0.0)

    def test_energy_scales_quadratically_with_amplitude(self):
        model = build()
        psi = smooth_streamfunction(model)
        single = float(
            model.diagnose(SphericalQGState(q=model.laplacian(psi))).kinetic_energy
        )
        double = float(
            model.diagnose(
                SphericalQGState(q=model.laplacian(2.0 * psi))
            ).kinetic_energy
        )
        assert double / single == pytest.approx(4.0, rel=1e-3)


class TestIntegration:
    def test_stays_finite(self):
        model = build(nx=32, ny=16, lateral_viscosity=1e4)
        state = model.apply_boundary_conditions(
            SphericalQGState(q=model.laplacian(smooth_streamfunction(model, 1e6)))
        )
        out = model.integrate(state, t0=0.0, t1=6 * 3600.0, dt=1800.0).ys
        assert np.isfinite(np.asarray(out.q)).all()

    def test_the_state_actually_evolves(self):
        model = build(nx=32, ny=16)
        state = model.apply_boundary_conditions(
            SphericalQGState(q=model.laplacian(smooth_streamfunction(model, 1e6)))
        )
        final = jax.tree_util.tree_map(
            lambda leaf: leaf[-1],
            model.integrate(state, t0=0.0, t1=6 * 3600.0, dt=1800.0).ys,
        )
        change = float(jnp.abs(final.q - state.q).max())
        assert change > 1e-3 * float(jnp.abs(state.q).max())


class TestTransforms:
    def test_jit(self):
        model = build(nx=32, ny=16)
        state = SphericalQGState(q=model.laplacian(smooth_streamfunction(model)))
        jitted = eqx.filter_jit(lambda m, s: m.vector_field(0.0, s))
        got = jitted(model, state).q
        expected = model.vector_field(0.0, state).q
        # Relative to the tendency's own magnitude: the RHS runs a CG
        # solve, and jit may reorder the reductions inside it, so the
        # iterate that trips the tolerance can differ slightly.
        difference = float(jnp.abs(got - expected).max())
        assert difference / float(jnp.abs(expected).max()) < 1e-3

    def test_grad_reaches_the_parameters(self):
        model = build(nx=32, ny=16, lateral_viscosity=1e4)
        state = SphericalQGState(q=model.laplacian(smooth_streamfunction(model)))

        def loss(m):
            return jnp.sum(m.vector_field(0.0, state).q ** 2)

        grads = eqx.filter_grad(loss)(model)
        assert np.isfinite(float(grads.params.lateral_viscosity))

    def test_grid_is_a_pytree_not_a_static_field(self):
        model = build()
        leaves = jax.tree_util.tree_leaves(model.grid)
        arrays = [leaf for leaf in leaves if hasattr(leaf, "shape") and leaf.shape]
        assert arrays


class TestGeometryRestrictionsAreStated:
    """``psi = 0`` on both polar rows is a wall, not a pole.

    A zonal solid-body flow has ``psi`` proportional to ``sin(phi)``,
    which takes different values at the two poles and cannot be
    represented at all under that condition — so a full-sphere range
    would silently solve a different problem.
    """

    def test_a_full_sphere_latitude_range_is_rejected(self):
        with pytest.raises(ValueError, match="reaches a pole"):
            build(lat_range=(-90.0, 90.0))

    def test_a_regional_longitude_span_is_rejected(self):
        with pytest.raises(ValueError, match="periodic"):
            build(lon_range=(0.0, 120.0))

    def test_a_band_is_still_accepted(self):
        assert build(lat_range=(-85.0, 85.0)) is not None


class TestPvInversionAssertionAcceptsThisModel:
    """The preflight check must recognise the spherical spelling.

    The Cartesian model keeps its inversion private and its Laplacian
    on ``diff``; this one exposes ``invert_pv`` and a separate
    ``laplacian`` operator, and the check used to reject it before
    testing anything.
    """

    def test_the_model_exposes_the_pair_the_check_looks_for(self):
        model = build()
        assert callable(model.invert_pv)
        assert callable(model.laplacian)

    def test_the_round_trip_the_check_performs_closes(self):
        model = build()
        psi = smooth_streamfunction(model)
        q = model.laplacian(psi)
        residual = model.laplacian(model.invert_pv(q)) - q
        relative = float(
            jnp.linalg.norm(residual[INTERIOR]) / jnp.linalg.norm(q[INTERIOR])
        )
        assert relative < 1e-4


class TestConstrainedRowsStayConstrained:
    """``q = 0`` on the polar rows must survive a step.

    The integrator projects only the RHS input, so a nonzero tendency
    there would accumulate and break the Dirichlet condition the PV
    inversion assumes.
    """

    def state(self, model):
        return model.apply_boundary_conditions(
            SphericalQGState(q=model.laplacian(smooth_streamfunction(model)))
        )

    def test_the_tendency_vanishes_on_the_polar_rows(self):
        model = build()
        tendency = model.vector_field(0.0, self.state(model))
        np.testing.assert_array_equal(np.asarray(tendency.q)[0, :], 0.0)
        np.testing.assert_array_equal(np.asarray(tendency.q)[-1, :], 0.0)

    def test_the_interior_tendency_is_not_zero(self):
        model = build()
        tendency = model.vector_field(0.0, self.state(model))
        assert float(jnp.abs(tendency.q[2:-2, 2:-2]).max()) > 0.0

    def test_the_condition_survives_an_integration(self):
        model = build(nx=32, ny=16, lateral_viscosity=1e4)
        out = model.integrate(self.state(model), t0=0.0, t1=6 * 3600.0, dt=1800.0).ys
        final = np.asarray(out.q[-1])
        np.testing.assert_allclose(final[0, :], 0.0, atol=1e-12)
        np.testing.assert_allclose(final[-1, :], 0.0, atol=1e-12)
