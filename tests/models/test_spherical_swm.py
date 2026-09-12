"""Tests for the spherical shallow water model."""

from __future__ import annotations

import itertools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from somax._src.models.spherical import SphericalSWM, SphericalSWMState


RADIUS = 6.371e6
OMEGA_EARTH = 7.292e-5
GRAVITY = 9.81
DEPTH = 3000.0
JET_SPEED = 20.0


def build(nx=48, ny=24, **kw):
    defaults = {
        "lon_range": (0.0, 360.0),
        "lat_range": (-80.0, 80.0),
        "radius": RADIUS,
        "omega": OMEGA_EARTH,
        "g": GRAVITY,
        "H0": DEPTH,
        "wind_profile": "none",
    }
    return SphericalSWM.create(nx=nx, ny=ny, **{**defaults, **kw})


def at_rest(model):
    shape = (model.grid.Ny, model.grid.Nx)
    return SphericalSWMState(
        h=jnp.full(shape, DEPTH), u=jnp.zeros(shape), v=jnp.zeros(shape)
    )


def solid_body_rotation(model, speed=JET_SPEED):
    """Williamson test 2: solid-body rotation in geostrophic balance.

    ``u = U cos(phi)``, ``v = 0`` and
    ``g h = g H0 - (R Omega U + U**2/2) sin^2(phi)``.
    """
    lat = np.asarray(model.grid.lat_T)
    height = (
        DEPTH
        - (RADIUS * OMEGA_EARTH * speed + 0.5 * speed**2) * np.sin(lat) ** 2 / GRAVITY
    )
    return model.apply_boundary_conditions(
        SphericalSWMState(
            h=jnp.asarray(height),
            u=jnp.asarray(speed * np.cos(lat)),
            v=jnp.zeros_like(jnp.asarray(lat)),
        )
    )


class TestCoriolis:
    def test_is_the_full_latitudinal_sine(self):
        model = build()
        lat_corner = model.grid.lat_T + 0.5 * model.grid.dlat
        expected = 2.0 * OMEGA_EARTH * jnp.sin(lat_corner)
        np.testing.assert_allclose(model.f_field, expected, rtol=1e-6)

    def test_changes_sign_across_the_equator(self):
        model = build()
        f = np.asarray(model.f_field)[:, 0]
        assert f.min() < 0.0 < f.max()

    def test_is_not_a_beta_plane(self):
        """A beta plane is linear in y; the sphere is not."""
        model = build(lat_range=(-80.0, 80.0))
        lat = np.asarray(model.grid.lat_T)[1:-1, 0] + 0.5 * model.grid.dlat
        f = np.asarray(model.f_field)[1:-1, 0]
        linear = np.polyfit(lat, f, 1)
        residual = np.abs(f - np.polyval(linear, lat)).max()
        assert residual > 0.1 * np.abs(f).max()

    def test_implied_beta_is_largest_at_the_equator(self):
        """beta = 2 Omega cos(phi)/R falls to zero at the poles."""
        model = build()
        lat = np.asarray(model.grid.lat_T)[1:-1, 0] + 0.5 * model.grid.dlat
        f = np.asarray(model.f_field)[1:-1, 0]
        beta = np.gradient(f, lat * RADIUS)
        equator = int(np.argmin(np.abs(lat)))
        assert beta[equator] == pytest.approx(beta.max(), rel=0.05)
        assert beta[0] < beta[equator]
        assert beta[-1] < beta[equator]

    def test_consts_carry_no_f0_or_beta(self):
        """On a sphere they are not free constants; the geometry sets them."""
        fields = set(build().consts.__dataclass_fields__)
        assert "f0" not in fields
        assert "beta" not in fields
        assert {"omega", "radius"} <= fields


class TestRestState:
    def test_uniform_rest_state_is_exactly_steady(self):
        model = build()
        tendency = model.vector_field(0.0, at_rest(model))
        for field in ("h", "u", "v"):
            np.testing.assert_array_equal(np.asarray(getattr(tendency, field)), 0.0)

    def test_stays_at_rest_when_integrated(self):
        model = build()
        state = at_rest(model)
        out = model.integrate(state, t0=0.0, t1=6 * 3600.0, dt=600.0).ys
        np.testing.assert_allclose(out.h[-1], state.h, rtol=1e-6)
        np.testing.assert_allclose(np.asarray(out.u[-1]), 0.0, atol=1e-12)


class TestSolidBodyRotation:
    """The model must rotate with the planet without spinning anything up."""

    def test_zonal_and_mass_tendencies_vanish(self):
        model = build()
        tendency = model.vector_field(0.0, solid_body_rotation(model))
        interior = (slice(2, -2), slice(2, -2))
        np.testing.assert_allclose(np.asarray(tendency.u)[interior], 0.0, atol=1e-12)
        np.testing.assert_allclose(np.asarray(tendency.h)[interior], 0.0, atol=1e-10)

    def test_meridional_residual_is_small_against_the_coriolis_term(self):
        model = build()
        tendency = model.vector_field(0.0, solid_body_rotation(model))
        interior = (slice(2, -2), slice(2, -2))
        residual = float(jnp.abs(tendency.v[interior]).max())
        coriolis = 2.0 * OMEGA_EARTH * JET_SPEED
        assert residual < 0.1 * coriolis

    def test_meridional_residual_converges_first_order(self):
        """It is discretisation error, not a bug — so it must shrink.

        ``u`` is specified at T-latitudes while the meridional balance
        is evaluated at V-points, half a cell away, which is a
        first-order offset on a C-grid.
        """
        residuals = []
        for ny in (16, 32, 64):
            model = build(nx=2 * ny, ny=ny)
            tendency = model.vector_field(0.0, solid_body_rotation(model))
            interior = (slice(2, -2), slice(2, -2))
            residuals.append(float(jnp.abs(tendency.v[interior]).max()))
        for coarse, fine in itertools.pairwise(residuals):
            assert coarse / fine == pytest.approx(2.0, rel=0.15)


class TestSphericalMetric:
    def test_zonal_cell_width_shrinks_polewards(self):
        # Computed here rather than via grid.dx_T: that helper arrives
        # with finitevolX #240, and somax still pins v0.0.41.
        model = build()
        grid = model.grid
        dx = np.asarray(grid.R * grid.cos_lat_T * grid.dlon)[1:-1, 0]
        lat = np.asarray(grid.lat_T)[1:-1, 0]
        assert dx[np.argmax(np.abs(lat))] < dx[np.argmin(np.abs(lat))]

    def test_a_zonal_gradient_is_stronger_at_high_latitude(self):
        """Converging meridians amplify d/dx for the same field."""
        model = build()
        lon = np.asarray(model.grid.lon_T)
        field = jnp.asarray(np.sin(lon))
        gradient = np.abs(np.asarray(model.diff.diff_lon_T_to_U(field)))
        lat = np.asarray(model.grid.lat_T)[:, 0]
        equator = int(np.argmin(np.abs(lat)))
        pole = int(np.argmax(np.abs(lat[1:-1]))) + 1
        assert gradient[pole].max() > gradient[equator].max()


class TestBoundaryConditions:
    def test_longitude_wraps(self):
        model = build()
        rng = np.random.RandomState(0)
        shape = (model.grid.Ny, model.grid.Nx)
        state = model.apply_boundary_conditions(
            SphericalSWMState(
                h=jnp.asarray(DEPTH + rng.randn(*shape)),
                u=jnp.asarray(rng.randn(*shape)),
                v=jnp.asarray(rng.randn(*shape)),
            )
        )
        np.testing.assert_allclose(state.h[:, 0], state.h[:, -2])
        np.testing.assert_allclose(state.h[:, -1], state.h[:, 1])

    def test_no_meridional_flow_through_the_polar_edges(self):
        model = build()
        rng = np.random.RandomState(1)
        shape = (model.grid.Ny, model.grid.Nx)
        state = model.apply_boundary_conditions(
            SphericalSWMState(
                h=jnp.full(shape, DEPTH),
                u=jnp.zeros(shape),
                v=jnp.asarray(rng.randn(*shape)),
            )
        )
        np.testing.assert_array_equal(np.asarray(state.v)[0, :], 0.0)
        np.testing.assert_array_equal(np.asarray(state.v)[-1, :], 0.0)


class TestDiagnostics:
    def test_invariants_are_advertised(self):
        model = build()
        invariants = model.diagnose(at_rest(model)).invariants()
        assert set(invariants) == {
            "mass",
            "total_energy",
            "potential_enstrophy",
            "casimir_q3",
        }

    def test_mass_uses_the_spherical_cell_area(self):
        """A uniform layer's mass is its depth times the spherical area."""
        model = build()
        from finitevolx import spherical_area_weights

        area = float(jnp.sum(spherical_area_weights(model.grid)[1:-1, 1:-1]))
        mass = float(model.diagnose(at_rest(model)).mass)
        assert mass == pytest.approx(DEPTH * area, rel=1e-5)

    def test_the_tendency_conserves_mass_for_solid_body_rotation(self):
        """For this state the discrete mass tendency is identically zero."""
        model = build()
        tendency = model.vector_field(0.0, solid_body_rotation(model))
        np.testing.assert_array_equal(np.asarray(tendency.h)[1:-1, 1:-1], 0.0)

    def test_mass_drift_is_bounded(self):
        """Not machine precision: see finitevolX#247.

        The spherical flux divergence does not telescope exactly
        against ``spherical_area_weights``, so mass drifts at the 1e-3
        level. The drift is bounded, and pinning it here makes a
        regression that worsens it visible.
        """
        model = build()
        state = solid_body_rotation(model)
        before = float(model.diagnose(state).mass)
        final = jax.tree_util.tree_map(
            lambda leaf: leaf[-1],
            model.integrate(state, t0=0.0, t1=6 * 3600.0, dt=600.0).ys,
        )
        after = float(model.diagnose(final).mass)
        assert abs(after - before) / before < 5e-3

    def test_mass_drift_does_not_depend_on_the_time_step(self):
        """Which is what rules out time-integration error as the cause."""
        model = build()
        state = solid_body_rotation(model)
        before = float(model.diagnose(state).mass)
        drifts = []
        for dt in (600.0, 300.0):
            final = jax.tree_util.tree_map(
                lambda leaf: leaf[-1],
                model.integrate(state, t0=0.0, t1=6 * 3600.0, dt=dt).ys,
            )
            drifts.append(abs(float(model.diagnose(final).mass) - before) / before)
        assert drifts[0] == pytest.approx(drifts[1], rel=0.05)

    def test_energy_is_positive_for_a_moving_state(self):
        model = build()
        energy = float(model.diagnose(solid_body_rotation(model)).energy)
        assert energy > 0.0


class TestTransforms:
    def test_integration_stays_finite(self):
        model = build(lateral_viscosity=1e4)
        out = model.integrate(
            solid_body_rotation(model), t0=0.0, t1=12 * 3600.0, dt=600.0
        ).ys
        assert np.isfinite(np.asarray(out.h)).all()
        assert np.isfinite(np.asarray(out.u)).all()
        assert np.isfinite(np.asarray(out.v)).all()

    def test_jit(self):
        model = build()
        state = solid_body_rotation(model)
        jitted = eqx.filter_jit(lambda m, s: m.vector_field(0.0, s))
        np.testing.assert_allclose(
            jitted(model, state).u, model.vector_field(0.0, state).u, rtol=1e-6
        )

    def test_grad_reaches_the_parameters(self):
        model = build(lateral_viscosity=1e4)
        state = solid_body_rotation(model)

        def loss(m):
            return jnp.sum(m.vector_field(0.0, state).u ** 2)

        grads = eqx.filter_grad(loss)(model)
        assert np.isfinite(float(grads.params.lateral_viscosity))

    def test_grid_is_a_pytree_not_a_static_field(self):
        """A spherical grid carries arrays, so static would be wrong.

        Marking it static hands equinox JAX arrays as hashable statics,
        which it warns about. The grid's own scalars (Nx, dlon, R) stay
        leaves too; what matters is that the cos(lat) arrays are there.
        """
        model = build()
        leaves = jax.tree_util.tree_leaves(model.grid)
        arrays = [leaf for leaf in leaves if hasattr(leaf, "shape") and leaf.shape]
        assert arrays, "spherical grid should expose its metric arrays as leaves"


class TestTotalEnergyIsDimensionallyConsistent:
    """``kinetic_energy`` is *specific* KE, so it needs weighting by h.

    Adding ``0.5(u^2+v^2)`` straight to ``0.5 g h^2`` mixes units and
    undercounts the kinetic part by a factor of the layer thickness.
    """

    def test_energy_scales_with_the_depth_of_a_moving_layer(self):
        """Doubling h at fixed velocity doubles the kinetic part.

        Measured at negligible gravity: at Earth's g the potential term
        is ~1e18 and the kinetic one ~1e14, so subtracting to isolate
        the kinetic part loses it to float32 cancellation.
        """
        model = build(g=1.0e-9)
        shape = (model.grid.Ny, model.grid.Nx)

        def energy(depth):
            state = SphericalSWMState(
                h=jnp.full(shape, depth),
                u=jnp.full(shape, 1.0),
                v=jnp.zeros(shape),
            )
            return float(model.diagnose(state).energy)

        assert energy(2.0 * DEPTH) == pytest.approx(2.0 * energy(DEPTH), rel=1e-3)

    def test_the_kinetic_part_is_not_independent_of_depth(self):
        """The bug this replaces: an unweighted KE would not move."""
        model = build(g=1.0e-9)
        shape = (model.grid.Ny, model.grid.Nx)

        def energy(depth):
            state = SphericalSWMState(
                h=jnp.full(shape, depth),
                u=jnp.full(shape, 1.0),
                v=jnp.zeros(shape),
            )
            return float(model.diagnose(state).energy)

        assert energy(2.0 * DEPTH) != pytest.approx(energy(DEPTH), rel=1e-2)

    def test_the_kinetic_part_matches_the_hand_computed_integral(self):
        from finitevolx import spherical_area_weights

        model = build()
        shape = (model.grid.Ny, model.grid.Nx)
        state = SphericalSWMState(
            h=jnp.full(shape, DEPTH),
            u=jnp.full(shape, 2.0),
            v=jnp.zeros(shape),
        )
        area = float(jnp.sum(spherical_area_weights(model.grid)[1:-1, 1:-1]))
        expected = DEPTH * 0.5 * 2.0**2 * area + 0.5 * GRAVITY * DEPTH**2 * area
        assert float(model.diagnose(state).energy) == pytest.approx(expected, rel=1e-3)

    def test_a_state_at_rest_is_purely_potential(self):
        from finitevolx import spherical_area_weights

        model = build()
        area = float(jnp.sum(spherical_area_weights(model.grid)[1:-1, 1:-1]))
        energy = float(model.diagnose(at_rest(model)).energy)
        assert energy == pytest.approx(0.5 * GRAVITY * DEPTH**2 * area, rel=1e-4)


class TestGeometryRestrictionsAreStated:
    """The boundary conditions describe one shape of domain only."""

    def test_a_regional_longitude_span_is_rejected(self):
        """The zonal wrap would join two unrelated boundaries."""
        with pytest.raises(ValueError, match="periodic"):
            build(lon_range=(0.0, 90.0))

    def test_a_full_sphere_latitude_range_is_rejected(self):
        """A solid wall is the wrong condition at a true pole."""
        with pytest.raises(ValueError, match="reaches a pole"):
            build(lat_range=(-90.0, 90.0))

    def test_either_pole_alone_is_enough_to_reject(self):
        for lat_range in ((-90.0, 80.0), (-80.0, 90.0)):
            with pytest.raises(ValueError, match="reaches a pole"):
                build(lat_range=lat_range)

    def test_a_band_is_still_accepted(self):
        assert build(lat_range=(-85.0, 85.0)) is not None

    def test_the_global_span_is_still_accepted(self):
        assert build(lon_range=(0.0, 360.0)) is not None

    def test_an_offset_global_span_is_accepted(self):
        """It is the 360-degree width that matters, not the origin."""
        assert build(lon_range=(-180.0, 180.0)) is not None


class TestConstrainedRowsStayConstrained:
    """The integrator projects the RHS input, not its output.

    ``apply_boundary_conditions`` zeroes the wall-normal ``v`` before
    each evaluation, but neither the returned nor the saved state is
    projected — so a nonzero tendency on those rows accumulates there
    and the snapshots show flow through a wall the model calls closed.
    """

    def sheared(self, model):
        """Zonal flow at the walls, which drives a Coriolis dv/dt."""
        lat = np.asarray(model.grid.lat_T)
        shape = lat.shape
        return model.apply_boundary_conditions(
            SphericalSWMState(
                h=jnp.full(shape, DEPTH),
                u=jnp.full(shape, JET_SPEED),
                v=jnp.zeros(shape),
            )
        )

    def test_the_meridional_tendency_vanishes_on_the_walls(self):
        model = build()
        tendency = model.vector_field(0.0, self.sheared(model))
        for row in (0, -2, -1):
            np.testing.assert_array_equal(np.asarray(tendency.v)[row, :], 0.0)

    def test_the_interior_tendency_is_not_zero(self):
        """Otherwise the test above would be vacuous."""
        model = build()
        tendency = model.vector_field(0.0, self.sheared(model))
        assert float(jnp.abs(tendency.v[2:-2, :]).max()) > 0.0

    def test_the_wall_stays_closed_through_an_integration(self):
        model = build(nx=32, ny=16)
        out = model.integrate(self.sheared(model), t0=0.0, t1=6 * 3600.0, dt=600.0).ys
        final = np.asarray(out.v[-1])
        for row in (0, -2, -1):
            np.testing.assert_allclose(final[row, :], 0.0, atol=1e-12)

    def test_the_other_tendencies_are_untouched(self):
        """Only ``v`` is constrained; ``h`` and ``u`` are free there."""
        model = build()
        tendency = model.vector_field(0.0, self.sheared(model))
        assert np.isfinite(np.asarray(tendency.h)).all()
        assert np.isfinite(np.asarray(tendency.u)).all()


def _land_mask(model, rows=slice(8, 12), cols=slice(5, 9)):
    from finitevolx import Mask2D

    wet = np.ones((model.grid.Ny, model.grid.Nx), dtype=bool)
    wet[rows, cols] = False
    return Mask2D.from_mask(jnp.asarray(wet))


class TestWindSourcesRespectTheMask:
    """Every other operator here is masked; the source was not.

    Nothing projects land cells back out of the returned state, so an
    unmasked source gives dry U/V cells a velocity that then persists
    into the snapshots.
    """

    def model(self):
        bare = build()
        mask = _land_mask(bare)
        return build(mask=mask, wind_amplitude=1.0, wind_profile="zonal"), mask

    def test_no_tendency_on_dry_zonal_cells(self):
        model, mask = self.model()
        du = np.asarray(model.vector_field(0.0, at_rest(model)).u)
        dry = ~np.asarray(mask.u).astype(bool)
        np.testing.assert_allclose(du[dry], 0.0, atol=0.0)

    def test_no_tendency_on_dry_meridional_cells(self):
        model, mask = self.model()
        dv = np.asarray(model.vector_field(0.0, at_rest(model)).v)
        dry = ~np.asarray(mask.v).astype(bool)
        np.testing.assert_allclose(dv[dry], 0.0, atol=0.0)

    def test_the_wet_cells_are_still_forced(self):
        """Otherwise masking everything would pass the tests above."""
        model, mask = self.model()
        du = np.asarray(model.vector_field(0.0, at_rest(model)).u)
        wet = np.asarray(mask.u).astype(bool)
        assert float(np.abs(du[wet]).max()) > 0.0

    def test_an_unmasked_model_is_unaffected(self):
        model = build(wind_amplitude=1.0, wind_profile="zonal")
        du = np.asarray(model.vector_field(0.0, at_rest(model)).u)
        assert float(np.abs(du).max()) > 0.0


class TestCornerInvariantsUseCornerAreas:
    """PV lives at X-points, half a latitude step north of the T-cells.

    Their dual-cell area goes as ``cos(lat_T + dlat/2)``, so reusing
    the T-cell weights biases enstrophy and the cubic Casimir, and
    increasingly so towards the poles.
    """

    def test_the_corner_weights_differ_from_the_t_weights(self):
        from finitevolx import spherical_area_weights

        from somax._src.models.spherical.swm import _corner_area_weights

        model = build()
        t_area = np.asarray(spherical_area_weights(model.grid))
        x_area = np.asarray(_corner_area_weights(model.grid))
        assert not np.allclose(t_area, x_area)

    def test_the_bias_grows_towards_the_poles(self):
        from finitevolx import spherical_area_weights

        from somax._src.models.spherical.swm import _corner_area_weights

        model = build()
        ratio = np.asarray(_corner_area_weights(model.grid)) / np.asarray(
            spherical_area_weights(model.grid)
        )
        interior = ratio[1:-1, 1:-1]
        mid = interior.shape[0] // 2
        assert abs(interior[1, 0] - 1.0) > abs(interior[mid, 0] - 1.0)

    def test_the_corner_weights_are_non_negative(self):
        from somax._src.models.spherical.swm import _corner_area_weights

        assert float(np.asarray(_corner_area_weights(build().grid)).min()) >= 0.0

    def test_enstrophy_is_still_positive_and_finite(self):
        model = build()
        rng = np.random.RandomState(0)
        shape = (model.grid.Ny, model.grid.Nx)
        state = SphericalSWMState(
            h=jnp.full(shape, DEPTH),
            u=jnp.asarray(0.1 * rng.randn(*shape)),
            v=jnp.asarray(0.1 * rng.randn(*shape)),
        )
        enstrophy = float(model.diagnose(state).enstrophy)
        assert np.isfinite(enstrophy) and enstrophy > 0.0


class TestSphericalFactoriesPreserveConstrainedParameters:
    """``jnp.asarray`` cannot convert a paramax wrapper."""

    def test_a_positive_viscosity_is_accepted(self):
        import paramax

        from somax._src.core.types import positive

        model = build(lateral_viscosity=positive(100.0))
        assert isinstance(model.params.lateral_viscosity, paramax.Parameterize)
        assert float(paramax.unwrap(model.params).lateral_viscosity) == pytest.approx(
            100.0, rel=1e-5
        )

    def test_a_frozen_drag_is_accepted(self):
        import paramax

        from somax._src.core.types import frozen

        model = build(bottom_drag=frozen(1e-7))
        assert isinstance(model.params.bottom_drag, paramax.NonTrainable)

    def test_a_constrained_wind_amplitude_is_accepted(self):
        from somax._src.core.types import positive

        model = build(wind_amplitude=positive(0.1), wind_profile="zonal")
        tendency = model.vector_field(0.0, at_rest(model))
        assert bool(jnp.all(jnp.isfinite(tendency.u)))

    def test_plain_numbers_still_work(self):
        model = build(lateral_viscosity=100.0)
        assert float(model.params.lateral_viscosity) == pytest.approx(100.0)
