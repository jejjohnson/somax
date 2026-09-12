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


class TestNondimensionalPreservesTheGroups:
    """The unit set must describe the *same* physics, not just unit scales.

    Carrying the SI ``9.81`` across would leave every gravity-dependent
    group wrong, and a factory handed those scales would build a
    different problem.
    """

    def cases(self):
        return [
            Scales.advective(L=1.0e6, U=0.1, f0=1.0e-4, H=500.0),
            Scales.inertial(L=1.0e6, f0=1.0e-4, H=500.0, rossby=0.01),
            Scales.planetary(a=6.371e6, Omega=7.292e-5, H=3000.0, rossby=0.05),
        ]

    @pytest.mark.parametrize("group", ["rossby", "burger", "froude", "lamb"])
    def test_group_is_unchanged(self, group):
        for scales in self.cases():
            got = getattr(scales.nondimensional(), group)
            assert got == pytest.approx(getattr(scales, group), rel=1e-12)

    def test_the_height_scale_ratio_is_unchanged(self):
        """``eta/H`` is dimensionless, so it must survive too."""
        for scales in self.cases():
            unit = scales.nondimensional()
            assert unit.eta / unit.H == pytest.approx(scales.eta / scales.H, rel=1e-12)

    def test_gravity_is_not_carried_across(self):
        """The whole point: 9.81 is meaningless at unit scales."""
        scales = Scales.inertial(L=1.0e6, f0=1.0e-4, H=500.0, rossby=0.01)
        assert scales.nondimensional().g != scales.g

    def test_the_kind_survives(self):
        for scales in self.cases():
            assert scales.nondimensional().kind == scales.kind

    def test_it_is_idempotent(self):
        for scales in self.cases():
            once = scales.nondimensional()
            assert once.nondimensional().g == pytest.approx(once.g, rel=1e-12)


class TestGravityValidation:
    """``g`` is a physical input like any other; reject bad ones on entry.

    Otherwise the failure surfaces later inside ``eta`` (a divide by
    zero) or ``froude`` (a square root of a negative), far from the
    call that caused it.
    """

    @pytest.mark.parametrize("bad", [0.0, -9.81, float("nan"), float("inf")])
    def test_advective_rejects_it(self, bad):
        with pytest.raises(ValueError, match="g must be"):
            Scales.advective(L=1.0, U=1.0, g=bad)

    @pytest.mark.parametrize("bad", [0.0, -9.81, float("nan")])
    def test_inertial_rejects_it(self, bad):
        with pytest.raises(ValueError, match="g must be"):
            Scales.inertial(L=1.0, f0=1.0, H=1.0, rossby=0.1, g=bad)

    @pytest.mark.parametrize("bad", [0.0, -9.81, float("nan")])
    def test_planetary_rejects_it(self, bad):
        with pytest.raises(ValueError, match="g must be"):
            Scales.planetary(a=1.0, Omega=1.0, H=1.0, rossby=0.1, g=bad)


class TestAdvectiveCoriolisValidation:
    """``f0`` may be zero or negative here, but never non-finite.

    Zero is a non-rotating model and a negative value is the southern
    hemisphere, so :func:`_require_positive` would be too strict. NaN
    and infinity are still fatal: they make ``eta`` and every
    rotation-dependent group non-finite, and the object looks valid
    until something downstream divides by it.
    """

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_it_rejects_a_non_finite_f0(self, bad):
        with pytest.raises(ValueError, match="f0 must be"):
            Scales.advective(L=1.0, U=1.0, f0=bad)

    @pytest.mark.parametrize("allowed", [0.0, -1e-4])
    def test_it_still_accepts_zero_and_negative(self, allowed):
        assert Scales.advective(L=1.0, U=1.0, f0=allowed).f0 == allowed


class TestTheDiffusiveSetHasNoGroupsToPreserve:
    """Its rotational groups are bookkeeping, not physics.

    ``Scales.diffusive`` has no imposed velocity and no rotation: its
    ``U = kappa/L`` exists only to keep the derived properties
    well-defined. Preserving ``rossby`` or ``froude`` across
    ``nondimensional()`` would be preserving an artefact, so the
    contract is scoped to exclude them — and the one number the family
    does have, the nondimensional diffusivity, is 1 either way.
    """

    def scales(self):
        return Scales.diffusive(L=2.0, kappa=0.01)

    def test_the_nondimensional_diffusivity_is_one_before_and_after(self):
        """``kappa`` is implied by ``T = L**2 / kappa``."""
        scales = self.scales()
        for candidate in (scales, scales.nondimensional()):
            kappa = candidate.L**2 / candidate.T
            assert kappa * candidate.T / candidate.L**2 == pytest.approx(1.0)

    def test_the_kind_still_survives(self):
        assert self.scales().nondimensional().kind == "diffusive"

    def test_the_unit_set_really_is_at_unit_scales(self):
        unit = self.scales().nondimensional()
        assert pytest.approx(1.0) == unit.L
        assert pytest.approx(1.0) == unit.T

    def test_rossby_is_not_claimed_to_survive(self):
        """Documented, and pinned so the exclusion is deliberate."""
        scales = self.scales()
        assert scales.nondimensional().rossby != pytest.approx(scales.rossby)
