"""Tests for the characteristic-scale sets."""

from __future__ import annotations

import math

import pytest

from somax._src.core.scales import Scales


class TestAdvectiveSet:
    def test_time_scale_is_the_eddy_turnover(self):
        s = Scales.advective(L=1e6, U=0.1, f0=1e-4)
        assert pytest.approx(1e7) == s.T
        assert s.kind == "advective"

    def test_rossby_number(self):
        s = Scales.advective(L=1e6, U=0.1, f0=1e-4)
        assert s.rossby == pytest.approx(0.1 / (1e-4 * 1e6))

    def test_defaults_suit_the_non_rotating_pde_models(self):
        s = Scales.advective(L=2.0, U=0.5)
        assert s.f0 == 1.0
        assert s.H == 1.0
        assert pytest.approx(4.0) == s.T


class TestInertialSet:
    def test_time_scale_is_the_inertial_period(self):
        s = Scales.inertial(L=1e6, f0=1e-4, H=500.0, rossby=0.05)
        assert pytest.approx(1.0 / 1e-4) == s.T
        assert s.kind == "inertial"

    def test_velocity_follows_from_the_rossby_number(self):
        s = Scales.inertial(L=1e6, f0=1e-4, H=500.0, rossby=0.05)
        assert pytest.approx(1e-4 * 1e6 * 0.05) == s.U
        assert s.rossby == pytest.approx(0.05)

    def test_time_scale_differs_from_the_advective_one(self):
        """The families genuinely disagree, which is why T is stored."""
        s = Scales.inertial(L=1e6, f0=1e-4, H=500.0, rossby=0.05)
        assert pytest.approx(s.L / s.U) != s.T

    def test_froude_and_burger_are_consistent(self):
        s = Scales.inertial(L=1e6, f0=1e-4, H=500.0, rossby=0.05)
        # Bu = (Ro / Fr)**2 for the shallow-water scaling.
        assert s.burger == pytest.approx((s.rossby / s.froude) ** 2, rel=1e-12)


class TestPlanetarySet:
    def test_scales_follow_the_planet(self):
        s = Scales.planetary(a=6.371e6, Omega=7.292e-5, H=1000.0, rossby=0.02)
        assert pytest.approx(6.371e6) == s.L
        assert pytest.approx(1.0 / 7.292e-5) == s.T
        assert s.kind == "planetary"

    def test_rossby_is_the_spherical_convention(self):
        """``Ro = U / (2 Omega a)``, recovered from the generic property."""
        a, omega, ro = 6.371e6, 7.292e-5, 0.02
        s = Scales.planetary(a=a, Omega=omega, H=1000.0, rossby=ro)
        assert pytest.approx(2.0 * omega * a * ro) == s.U
        assert s.rossby == pytest.approx(ro)
        assert s.U / (2.0 * omega * a) == pytest.approx(ro)

    def test_f0_is_twice_omega(self):
        s = Scales.planetary(a=6.371e6, Omega=7.292e-5, H=1000.0, rossby=0.02)
        assert s.f0 == pytest.approx(2.0 * 7.292e-5)
        assert s.omega == pytest.approx(7.292e-5)

    def test_lamb_parameter_matches_the_textbook_form(self):
        a, omega, H = 6.371e6, 7.292e-5, 1000.0
        s = Scales.planetary(a=a, Omega=omega, H=H, rossby=0.02)
        expected = 4.0 * omega**2 * a**2 / (s.g * H)
        assert s.lamb == pytest.approx(expected, rel=1e-12)
        assert s.lamb == pytest.approx(1.0 / s.burger, rel=1e-12)


class TestDerivedScales:
    @pytest.fixture
    def scales(self):
        return Scales.advective(L=1e6, U=0.1, f0=1e-4, H=1000.0)

    def test_eta_is_the_geostrophic_height_scale(self, scales):
        assert scales.eta == pytest.approx(1e-4 * 0.1 * 1e6 / scales.g)

    def test_vorticity_and_streamfunction(self, scales):
        assert scales.vorticity == pytest.approx(0.1 / 1e6)
        assert scales.streamfunction == pytest.approx(0.1 * 1e6)

    def test_froude(self, scales):
        assert scales.froude == pytest.approx(0.1 / math.sqrt(scales.g * 1000.0))


class TestNondimensionalCounterpart:
    def test_advective_unit_scales_preserve_rossby(self):
        s = Scales.advective(L=1e6, U=0.1, f0=1e-4)
        nd = s.nondimensional()
        assert nd.L == 1.0
        assert nd.U == 1.0
        assert nd.T == 1.0
        assert nd.rossby == pytest.approx(s.rossby)
        assert nd.kind == "advective"

    def test_inertial_unit_scales_preserve_rossby(self):
        s = Scales.inertial(L=1e6, f0=1e-4, H=500.0, rossby=0.05)
        nd = s.nondimensional()
        assert nd.L == 1.0
        assert nd.f0 == 1.0
        assert nd.T == 1.0
        assert nd.rossby == pytest.approx(s.rossby)

    def test_planetary_unit_scales_preserve_rossby(self):
        s = Scales.planetary(a=6.371e6, Omega=7.292e-5, H=1000.0, rossby=0.02)
        nd = s.nondimensional()
        assert nd.L == 1.0
        assert nd.T == 1.0
        assert nd.rossby == pytest.approx(s.rossby)
        assert nd.kind == "planetary"


class TestStaticness:
    def test_fields_are_static_so_groups_drive_control_flow(self):
        """A traced scale would break resolution guards in a factory."""
        import jax

        s = Scales.advective(L=1e6, U=0.1, f0=1e-4)
        leaves = jax.tree_util.tree_leaves(s)
        assert leaves == []

    def test_dimensionless_groups_are_plain_floats(self):
        s = Scales.advective(L=1e6, U=0.1, f0=1e-4)
        assert isinstance(s.rossby, float)
        assert isinstance(s.burger, float)
        assert isinstance(s.T, float)


class TestValidation:
    @pytest.mark.parametrize(
        ("kwargs", "bad"),
        [
            ({"L": 0.0, "U": 1.0}, "L"),
            ({"L": 1.0, "U": -1.0}, "U"),
            ({"L": 1.0, "U": 1.0, "H": 0.0}, "H"),
            ({"L": math.inf, "U": 1.0}, "L"),
        ],
    )
    def test_advective_rejects_non_positive_scales(self, kwargs, bad):
        with pytest.raises(ValueError, match=bad):
            Scales.advective(**kwargs)

    def test_inertial_rejects_zero_coriolis(self):
        with pytest.raises(ValueError, match="f0"):
            Scales.inertial(L=1.0, f0=0.0, H=1.0, rossby=0.1)

    def test_planetary_rejects_zero_rotation(self):
        with pytest.raises(ValueError, match="Omega"):
            Scales.planetary(a=1.0, Omega=0.0, H=1.0, rossby=0.1)
