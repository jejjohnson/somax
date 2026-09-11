"""Tests for the spherical ``from_nondimensional`` factories (#177)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from somax._src.cli._assertions import (
    AssertionFailedError,
    check_equatorial_deformation_radius,
)
from somax._src.models.spherical import (
    SphericalQG,
    SphericalQGState,
    SphericalSWM,
    SphericalSWMState,
)


RADIUS = 6.371e6
OMEGA_EARTH = 7.292e-5
GRAVITY = 9.81
DEPTH = 3000.0

#: The Coriolis stand-in on a sphere: f(phi) = f0 sin(phi) with f0 = 2 Omega.
F0_HAT = 2.0

GRID = {
    "nx": 48,
    "ny": 24,
    "lon_range": (0.0, 360.0),
    "lat_range": (-80.0, 80.0),
    "wind_profile": "zonal",
}
INTERIOR = (slice(2, -2), slice(2, -2))


def relative(got, expected) -> float:
    return float(
        jnp.abs(jnp.asarray(got) - jnp.asarray(expected)).max()
        / jnp.abs(jnp.asarray(expected)).max()
    )


class TestScaleSet:
    def test_is_planetary(self):
        _, scales = SphericalSWM.from_nondimensional(rossby=0.05, lamb=10.0, **GRID)
        assert scales.kind == "planetary"

    def test_unit_radius_rotation_and_depth(self):
        model, scales = SphericalSWM.from_nondimensional(rossby=0.05, lamb=10.0, **GRID)
        assert (scales.L, scales.H, scales.T) == (1.0, 1.0, 1.0)
        assert model.consts.radius == 1.0
        assert model.consts.omega == 1.0
        assert model.consts.H0 == 1.0

    def test_f0_is_twice_omega(self):
        """Not one: on a sphere f = 2 Omega sin(phi)."""
        _, scales = SphericalSWM.from_nondimensional(rossby=0.05, lamb=10.0, **GRID)
        assert scales.f0 == F0_HAT

    def test_rossby_round_trips(self):
        _, scales = SphericalSWM.from_nondimensional(rossby=0.07, lamb=10.0, **GRID)
        assert scales.rossby == pytest.approx(0.07)

    def test_velocity_scale_is_two_rossby(self):
        """U = 2 Omega a Ro, and Omega = a = 1 here."""
        _, scales = SphericalSWM.from_nondimensional(rossby=0.05, lamb=10.0, **GRID)
        assert float(scales.U) == pytest.approx(0.1)

    def test_the_qg_factory_agrees_on_the_scale_set(self):
        _, swm = SphericalSWM.from_nondimensional(rossby=0.05, lamb=10.0, **GRID)
        _, qg = SphericalQG.from_nondimensional(rossby=0.05, **GRID)
        assert (qg.kind, qg.L, qg.T, qg.f0, qg.U) == (
            swm.kind,
            swm.L,
            swm.T,
            swm.f0,
            swm.U,
        )


class TestStratificationInputs:
    def test_lamb_is_the_inverse_burger(self):
        _, scales = SphericalSWM.from_nondimensional(rossby=0.05, lamb=10.0, **GRID)
        assert scales.burger == pytest.approx(0.1)
        assert scales.lamb == pytest.approx(10.0)

    def test_burger_sets_gravity(self):
        """Bu = gH/(f0 a)^2 with H = a = 1 and f0 = 2, so g = 4 Bu."""
        model, _ = SphericalSWM.from_nondimensional(rossby=0.05, burger=0.25, **GRID)
        assert model.consts.gravity == pytest.approx(1.0)

    def test_froude_and_burger_agree_through_rossby(self):
        rossby, burger = 0.05, 0.1
        froude = rossby / np.sqrt(burger)
        by_burger, _ = SphericalSWM.from_nondimensional(
            rossby=rossby, burger=burger, **GRID
        )
        by_froude, _ = SphericalSWM.from_nondimensional(
            rossby=rossby, froude=froude, **GRID
        )
        assert by_froude.consts.gravity == pytest.approx(by_burger.consts.gravity)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"burger": 0.1, "lamb": 10.0},
            {"burger": 0.1, "froude": 0.15},
            {"burger": 0.1, "froude": 0.15, "lamb": 10.0},
        ],
    )
    def test_exactly_one_stratification_input_is_required(self, kwargs):
        with pytest.raises(ValueError, match="exactly one of burger, froude or lamb"):
            SphericalSWM.from_nondimensional(rossby=0.05, **kwargs, **GRID)

    def test_the_qg_factory_takes_no_stratification(self):
        """Barotropic QG is rigid-lid, so a Burger number is meaningless."""
        with pytest.raises(TypeError):
            SphericalQG.from_nondimensional(rossby=0.05, burger=0.1, **GRID)


class TestCoefficientMapping:
    def test_ekman_sets_the_bottom_drag(self):
        """Ek = kappa/f0 with f0 = 2, so kappa = 2 Ek."""
        model, _ = SphericalSWM.from_nondimensional(
            rossby=0.05, lamb=10.0, ekman=0.01, **GRID
        )
        assert float(model.params.bottom_drag) == pytest.approx(0.02)

    def test_lateral_ekman_sets_the_viscosity(self):
        model, _ = SphericalSWM.from_nondimensional(
            rossby=0.05, lamb=10.0, ekman_lateral=1e-4, **GRID
        )
        assert float(model.params.lateral_viscosity) == pytest.approx(2e-4)

    def test_wind_carries_the_velocity_scale(self):
        """tau_hat = tau0/(f0 U), so tau0 = tau_hat f0 U = 4 Ro tau_hat."""
        model, _ = SphericalSWM.from_nondimensional(
            rossby=0.05, lamb=10.0, wind_hat=0.5, **GRID
        )
        assert float(model.params.wind_amplitude) == pytest.approx(0.5 * 4 * 0.05)

    def test_qg_coefficients_use_the_same_factor(self):
        model, _ = SphericalQG.from_nondimensional(
            rossby=0.05, ekman=0.01, ekman_lateral=1e-4, **GRID
        )
        assert float(model.params.bottom_drag) == pytest.approx(0.02)
        assert float(model.params.lateral_viscosity) == pytest.approx(2e-4)

    def test_zero_forcing_is_the_default(self):
        model, _ = SphericalSWM.from_nondimensional(rossby=0.05, lamb=10.0, **GRID)
        assert float(model.params.bottom_drag) == 0.0
        assert float(model.params.lateral_viscosity) == 0.0
        assert float(model.params.wind_amplitude) == 0.0


class TestValidation:
    @pytest.mark.parametrize("rossby", [0.0, -0.1, float("nan"), float("inf")])
    def test_rossby_must_be_positive_and_finite(self, rossby):
        with pytest.raises(ValueError, match="rossby"):
            SphericalSWM.from_nondimensional(rossby=rossby, lamb=10.0, **GRID)

    @pytest.mark.parametrize("name", ["ekman", "ekman_lateral", "wind_hat"])
    def test_dissipation_and_forcing_must_be_non_negative(self, name):
        with pytest.raises(ValueError, match=name):
            SphericalSWM.from_nondimensional(
                rossby=0.05, lamb=10.0, **{name: -1.0}, **GRID
            )

    def test_lamb_must_be_positive(self):
        with pytest.raises(ValueError, match="lamb"):
            SphericalSWM.from_nondimensional(rossby=0.05, lamb=-1.0, **GRID)

    @pytest.mark.parametrize("name", ["ekman", "ekman_lateral", "wind_hat"])
    def test_the_qg_factory_validates_too(self, name):
        with pytest.raises(ValueError, match=name):
            SphericalQG.from_nondimensional(rossby=0.05, **{name: -1.0}, **GRID)


class TestEquivalenceWithTheDimensionalModel:
    """The point of the exercise: the same physics, different units.

    Length unit ``a``, time unit ``1/Omega``, thickness unit ``H0`` —
    so the velocity unit is ``a Omega``, *not* the velocity scale ``U``
    (which is ``2 Ro`` of it). Conflating the two is the easy mistake,
    and these tests would catch it: they fail by exactly ``1/(2 Ro)``.
    """

    rossby = 0.05

    def dimensional_and_nondimensional(self, **overrides):
        f0 = 2.0 * OMEGA_EARTH
        physical = {
            "lateral_viscosity": 1.0e4,
            "bottom_drag": 1.0e-7,
            "wind_amplitude": 1.0e-8,
            **overrides,
        }
        dimensional = SphericalSWM.create(
            radius=RADIUS, omega=OMEGA_EARTH, g=GRAVITY, H0=DEPTH, **physical, **GRID
        )
        velocity = f0 * RADIUS * self.rossby
        nondimensional, scales = SphericalSWM.from_nondimensional(
            rossby=self.rossby,
            burger=GRAVITY * DEPTH / (f0 * RADIUS) ** 2,
            ekman=physical["bottom_drag"] / f0,
            ekman_lateral=physical["lateral_viscosity"] / (f0 * RADIUS**2),
            wind_hat=physical["wind_amplitude"] / (f0 * velocity),
            check_resolution=False,
            **GRID,
        )
        return dimensional, nondimensional, scales

    def states(self, model):
        rng = np.random.RandomState(0)
        shape = (model.grid.Ny, model.grid.Nx)
        velocity_unit = RADIUS * OMEGA_EARTH
        h = DEPTH * (1.0 + 0.01 * rng.randn(*shape))
        u = velocity_unit * 0.01 * rng.randn(*shape)
        v = velocity_unit * 0.01 * rng.randn(*shape)
        dimensional = SphericalSWMState(
            h=jnp.asarray(h), u=jnp.asarray(u), v=jnp.asarray(v)
        )
        nondimensional = SphericalSWMState(
            h=jnp.asarray(h / DEPTH),
            u=jnp.asarray(u / velocity_unit),
            v=jnp.asarray(v / velocity_unit),
        )
        return dimensional, nondimensional

    def test_burger_matches_the_dimensional_configuration(self):
        _, _, scales = self.dimensional_and_nondimensional()
        f0 = 2.0 * OMEGA_EARTH
        assert scales.burger == pytest.approx(GRAVITY * DEPTH / (f0 * RADIUS) ** 2)

    @pytest.mark.parametrize("field", ["h", "u", "v"])
    def test_tendencies_agree_after_rescaling(self, field):
        dimensional, nondimensional, _ = self.dimensional_and_nondimensional()
        state_d, state_n = self.states(dimensional)
        state_d = dimensional.apply_boundary_conditions(state_d)
        state_n = nondimensional.apply_boundary_conditions(state_n)

        unit = DEPTH if field == "h" else RADIUS * OMEGA_EARTH
        rescaled = getattr(dimensional.vector_field(0.0, state_d), field) / (
            unit * OMEGA_EARTH
        )
        got = getattr(nondimensional.vector_field(0.0, state_n), field)
        assert relative(rescaled[INTERIOR], got[INTERIOR]) < 1e-4

    def test_the_coriolis_field_rescales(self):
        dimensional, nondimensional, _ = self.dimensional_and_nondimensional()
        assert (
            relative(dimensional.f_field / OMEGA_EARTH, nondimensional.f_field) < 1e-6
        )


class TestQGEquivalence:
    def test_vorticity_tendency_agrees_after_rescaling(self):
        f0 = 2.0 * OMEGA_EARTH
        rossby, nu, kappa = 0.05, 1.0e4, 1.0e-7
        dimensional = SphericalQG.create(
            radius=RADIUS,
            omega=OMEGA_EARTH,
            lateral_viscosity=nu,
            bottom_drag=kappa,
            **GRID,
        )
        nondimensional, _ = SphericalQG.from_nondimensional(
            rossby=rossby,
            ekman=kappa / f0,
            ekman_lateral=nu / (f0 * RADIUS**2),
            **GRID,
        )
        lat = np.asarray(dimensional.grid.lat_T)
        lon = np.asarray(dimensional.grid.lon_T)
        q = 1.0e-6 * np.sin(2.0 * lon) * np.cos(lat) ** 2

        state_d = dimensional.apply_boundary_conditions(
            SphericalQGState(q=jnp.asarray(q))
        )
        state_n = nondimensional.apply_boundary_conditions(
            SphericalQGState(q=jnp.asarray(q / OMEGA_EARTH))
        )
        rescaled = dimensional.vector_field(0.0, state_d).q / OMEGA_EARTH**2
        got = nondimensional.vector_field(0.0, state_n).q
        assert relative(rescaled[INTERIOR], got[INTERIOR]) < 1e-3


class TestEquatorialDeformationRadius:
    """``L_eq/a = Bu**(1/4)``: the equatorial waveguide's trapping width."""

    def test_passes_on_a_well_resolved_sphere(self):
        model, _ = SphericalSWM.from_nondimensional(
            nx=128, ny=64, rossby=0.05, lamb=10.0
        )
        check_equatorial_deformation_radius(None, model)

    def test_fails_on_a_coarse_grid(self):
        with pytest.raises(AssertionFailedError, match="L_eq/dx"):
            SphericalSWM.from_nondimensional(nx=8, ny=4, rossby=0.05, lamb=1e4)

    def test_the_guard_can_be_switched_off(self):
        model, _ = SphericalSWM.from_nondimensional(
            nx=8, ny=4, rossby=0.05, lamb=1e4, check_resolution=False
        )
        assert model.grid.Nx == 10

    def test_radius_is_the_quarter_power_of_the_burger_number(self):
        """Which is what makes the guard read the same at any scale."""
        model, scales = SphericalSWM.from_nondimensional(
            nx=128, ny=64, rossby=0.05, burger=0.0625
        )
        expected = scales.burger**0.25
        c = np.sqrt(model.consts.gravity * model.consts.H0)
        beta_eq = 2.0 * model.consts.omega / model.consts.radius
        assert np.sqrt(c / beta_eq) == pytest.approx(expected, rel=1e-6)

    def test_rejects_the_rigid_lid_qg_model(self):
        model, _ = SphericalQG.from_nondimensional(rossby=0.05, **GRID)
        with pytest.raises(AssertionFailedError, match="rigid-lid"):
            check_equatorial_deformation_radius(None, model)

    def test_rejects_a_model_without_a_grid(self):
        with pytest.raises(AssertionFailedError, match="lacks consts/grid"):
            check_equatorial_deformation_radius(None, object())

    def test_warns_when_marginally_resolved(self, caplog):
        """A warning, not a failure: it is usable but not comfortable."""
        model, _ = SphericalSWM.from_nondimensional(
            nx=32, ny=16, rossby=0.05, lamb=8.0, check_resolution=False
        )
        c = np.sqrt(model.consts.gravity * model.consts.H0)
        beta_eq = 2.0 * model.consts.omega / model.consts.radius
        ratio = np.sqrt(c / beta_eq) / (2.0 * np.pi / 32.0)
        assert 2.0 < ratio < 4.0
        check_equatorial_deformation_radius(None, model)


def spherical_bundle(nx: int = 32, ny: int = 16):
    """A minimal global-sphere bundle.

    Built by hand rather than through a scenario: every spherical
    scenario is still a stub (#79), so there is nothing in the registry
    to ask for lon/lat bounds yet.
    """
    from somax._src.cli.scenarios._types import (
        Constants,
        ForcingFields,
        Geometry,
        InitialConditionSpec,
        ScenarioBundle,
    )

    return ScenarioBundle(
        name="test_sphere",
        geometry=Geometry(
            kind="sphere",
            nx=nx,
            ny=ny,
            lon_bounds=(0.0, 360.0),
            lat_bounds=(-80.0, 80.0),
        ),
        constants=Constants(f0=-1.2e-4, beta=1.0e-11),
        forcing=ForcingFields(),
        initial_condition=InitialConditionSpec(type="at_rest"),
    )


class TestRegistryWiring:
    """The ``scenario.nondim`` path through the CLI model registry."""

    def entry(self, name):
        from somax._src.cli.models_registry import MODELS

        return MODELS[name]

    @pytest.mark.parametrize("name", ["spherical_swm", "spherical_qg"])
    def test_the_entry_advertises_a_nondimensional_factory(self, name):
        assert self.entry(name).from_nondimensional is not None

    def test_swm_builds_from_a_nondim_block(self):
        built = self.entry("spherical_swm").from_nondimensional(
            spherical_bundle(), {"rossby": 0.05, "lamb": 8.0}
        )
        assert built.model.consts.radius == 1.0
        assert built.state0.h.shape == (built.model.grid.Ny, built.model.grid.Nx)

    def test_qg_builds_from_a_nondim_block(self):
        built = self.entry("spherical_qg").from_nondimensional(
            spherical_bundle(), {"rossby": 0.05, "ekman": 0.01}
        )
        assert built.model.consts.omega == 1.0
        assert float(built.model.params.bottom_drag) == pytest.approx(0.02)

    @pytest.mark.parametrize("name", ["spherical_swm", "spherical_qg"])
    def test_an_unknown_key_is_rejected(self, name):
        with pytest.raises(ValueError, match=r"unknown scenario\.nondim key"):
            self.entry(name).from_nondimensional(
                spherical_bundle(), {"rossby": 0.05, "lamb": 8.0, "nope": 1.0}
            )

    @pytest.mark.parametrize(
        ("name", "block"),
        [
            ("spherical_swm", {"lamb": 8.0}),
            ("spherical_qg", {"ekman": 0.01}),
        ],
    )
    def test_rossby_is_required(self, name, block):
        with pytest.raises(ValueError, match="requires 'rossby'"):
            self.entry(name).from_nondimensional(spherical_bundle(), block)

    def test_swm_requires_a_stratification_key(self):
        with pytest.raises(ValueError, match="burger', 'froude' or 'lamb'"):
            self.entry("spherical_swm").from_nondimensional(
                spherical_bundle(), {"rossby": 0.05}
            )

    @pytest.mark.parametrize("name", ["spherical_swm", "spherical_qg"])
    def test_a_cartesian_geometry_is_rejected(self, name):
        from somax._src.cli.scenarios import SCENARIOS

        bundle = SCENARIOS["double_gyre"].build(
            {
                "grid": {"nx": 8, "ny": 8, "Lx": 1.0e6, "Ly": 1.0e6},
                "initial_condition": {"type": "at_rest"},
            }
        )
        with pytest.raises(ValueError, match="lon_bounds / lat_bounds"):
            self.entry(name).from_nondimensional(bundle, {"rossby": 0.05, "lamb": 8.0})
