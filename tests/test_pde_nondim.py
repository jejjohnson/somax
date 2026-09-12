"""Tests for the pde1d / pde2d nondimensional factories."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from somax._src.core.scales import Scales
from somax._src.models.pde1d.burgers import Burgers1D, Burgers1DState
from somax._src.models.pde1d.diffusion import Diffusion1D, Diffusion1DState
from somax._src.models.pde1d.linear_convection import LinearConvection1D
from somax._src.models.pde1d.nonlinear_convection import NonlinearConvection1D
from somax._src.models.pde2d.burgers import Burgers2D
from somax._src.models.pde2d.diffusion import Diffusion2D
from somax._src.models.pde2d.linear_convection import LinearConvection2D
from somax._src.models.pde2d.navier_stokes import IncompressibleNS2D
from somax._src.models.pde2d.nonlinear_convection import NonlinearConvection2D


def relative_error(got, expected):
    got, expected = np.asarray(got), np.asarray(expected)
    return np.abs(got - expected).max() / np.abs(expected).max()


ONE_D = [
    ("Burgers1D", lambda: Burgers1D.from_nondimensional(nx=32, reynolds=100.0)),
    ("Diffusion1D", lambda: Diffusion1D.from_nondimensional(nx=32)),
    ("LinearConvection1D", lambda: LinearConvection1D.from_nondimensional(nx=32)),
    (
        "NonlinearConvection1D",
        lambda: NonlinearConvection1D.from_nondimensional(nx=32),
    ),
]

TWO_D = [
    (
        "Burgers2D",
        lambda: Burgers2D.from_nondimensional(nx=16, ny=16, reynolds=100.0),
    ),
    ("Diffusion2D", lambda: Diffusion2D.from_nondimensional(nx=16, ny=16)),
    (
        "IncompressibleNS2D",
        lambda: IncompressibleNS2D.from_nondimensional(nx=16, ny=16, reynolds=100.0),
    ),
    (
        "LinearConvection2D",
        lambda: LinearConvection2D.from_nondimensional(nx=16, ny=16),
    ),
    (
        "NonlinearConvection2D",
        lambda: NonlinearConvection2D.from_nondimensional(nx=16, ny=16),
    ),
]

ALL_MODELS = ONE_D + TWO_D


class TestCommonContract:
    @pytest.mark.parametrize(
        ("name", "build"), ALL_MODELS, ids=[n for n, _ in ALL_MODELS]
    )
    def test_returns_model_and_scales(self, name, build):
        model, scales = build()
        assert model is not None
        assert isinstance(scales, Scales)

    @pytest.mark.parametrize(
        ("name", "build"), ALL_MODELS, ids=[n for n, _ in ALL_MODELS]
    )
    def test_domain_is_unit_length(self, name, build):
        model, _ = build()
        assert model.grid.Lx == pytest.approx(1.0)

    @pytest.mark.parametrize(
        ("name", "build"), ALL_MODELS, ids=[n for n, _ in ALL_MODELS]
    )
    def test_time_scale_is_unity(self, name, build):
        """Every set is arranged so the model's own time unit is 1."""
        _, scales = build()
        assert pytest.approx(1.0) == scales.T

    @pytest.mark.parametrize(("name", "build"), TWO_D, ids=[n for n, _ in TWO_D])
    def test_aspect_only_stretches_y(self, name, build):
        model, _ = build()
        assert model.grid.Ly == pytest.approx(1.0)


class TestReynolds:
    @pytest.mark.parametrize("reynolds", [1.0, 100.0, 1e4])
    def test_burgers_1d_viscosity_is_the_inverse(self, reynolds):
        model, _ = Burgers1D.from_nondimensional(nx=32, reynolds=reynolds)
        assert float(model.params.nu) == pytest.approx(1.0 / reynolds, rel=1e-6)

    def test_burgers_2d_viscosity_is_the_inverse(self):
        model, _ = Burgers2D.from_nondimensional(nx=16, ny=16, reynolds=250.0)
        assert float(model.params.nu) == pytest.approx(1.0 / 250.0, rel=1e-6)

    def test_navier_stokes_viscosity_is_the_inverse(self):
        model, _ = IncompressibleNS2D.from_nondimensional(nx=16, ny=16, reynolds=500.0)
        assert float(model.params.nu) == pytest.approx(1.0 / 500.0, rel=1e-6)

    def test_advective_scale_set(self):
        _, scales = Burgers1D.from_nondimensional(nx=32, reynolds=100.0)
        assert scales.kind == "advective"
        assert scales.L == 1.0
        assert scales.U == 1.0

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_non_positive_reynolds_is_rejected(self, bad):
        with pytest.raises(ValueError, match="reynolds"):
            Burgers1D.from_nondimensional(nx=32, reynolds=bad)

    def test_negative_aspect_is_rejected(self):
        with pytest.raises(ValueError, match="aspect"):
            Burgers2D.from_nondimensional(nx=16, ny=16, reynolds=100.0, aspect=-1.0)


class TestDiffusiveSet:
    def test_nondimensional_diffusivity_is_one(self):
        """The diffusive scaling leaves no free number: kappa becomes 1."""
        model, scales = Diffusion1D.from_nondimensional(nx=32)
        assert float(model.params.nu) == pytest.approx(1.0)
        assert scales.kind == "diffusive"

    def test_two_dimensional_case_matches(self):
        model, scales = Diffusion2D.from_nondimensional(nx=16, ny=16)
        assert float(model.params.nu) == pytest.approx(1.0)
        assert scales.kind == "diffusive"

    def test_diffusive_scales_recover_the_time_scale(self):
        scales = Scales.diffusive(L=2.0, kappa=0.01)
        assert pytest.approx(2.0**2 / 0.01) == scales.T

    def test_diffusive_rejects_non_positive_inputs(self):
        with pytest.raises(ValueError, match="kappa"):
            Scales.diffusive(L=1.0, kappa=0.0)


class TestConvectionSpeed:
    def test_linear_convection_speed_is_unity(self):
        model, scales = LinearConvection1D.from_nondimensional(nx=32)
        assert float(model.params.c) == pytest.approx(1.0)
        assert scales.kind == "advective"

    def test_linear_convection_2d_speed_is_unity(self):
        """The *speed*, not each component — the default is diagonal."""
        model, _ = LinearConvection2D.from_nondimensional(nx=16, ny=16)
        speed = np.hypot(float(model.params.cx), float(model.params.cy))
        assert speed == pytest.approx(1.0)


class TestCflHelper:
    def test_advective_step(self):
        scales = Scales.advective(L=2.0, U=1.0)
        assert scales.dt_from_cfl(0.4, 100) == pytest.approx(0.4 / 100)

    def test_diffusive_step(self):
        scales = Scales.diffusive(L=2.0, kappa=0.01)
        assert scales.dt_from_cfl(0.4, 100) == pytest.approx(0.4 / 100**2 / 2)

    def test_mode_can_be_forced(self):
        scales = Scales.advective(L=1.0, U=1.0)
        got = scales.dt_from_cfl(0.5, 10, mode="diffusive", diffusivity=1.0)
        assert got == pytest.approx(0.5 * (1 / 10) ** 2 / 2)

    def test_unknown_mode_is_rejected(self):
        with pytest.raises(ValueError, match=r"advective.*diffusive"):
            Scales.advective(L=1.0, U=1.0).dt_from_cfl(0.5, 10, mode="nope")

    def test_rejects_non_positive_inputs(self):
        with pytest.raises(ValueError, match="cfl"):
            Scales.advective(L=1.0, U=1.0).dt_from_cfl(0.0, 10)

    def test_a_step_from_the_helper_keeps_burgers_finite(self):
        model, scales = Burgers1D.from_nondimensional(nx=64, reynolds=100.0)
        dt = scales.dt_from_cfl(0.4, 64)
        x = jnp.linspace(0.0, 1.0, model.grid.Nx)
        state = Burgers1DState(u=jnp.sin(2.0 * jnp.pi * x))
        out = model.integrate(state, t0=0.0, t1=20 * dt, dt=dt).ys
        assert np.isfinite(np.asarray(out.u)).all()


class TestDimensionalEquivalenceBurgers:
    """A nondimensional Burgers run reproduces its dimensional twin.

    With ``L`` and ``U`` chosen, ``nu = U L / Re``, ``u_dim = U u'`` and
    ``t_dim = (L/U) t'``.
    """

    L = 3.0
    U = 0.25
    RE = 200.0
    NX = 64

    def dimensional(self):
        return Burgers1D.create(
            nx=self.NX, Lx=self.L, nu=self.U * self.L / self.RE, periodic=True
        )

    def initial(self, model, amplitude):
        x = jnp.linspace(0.0, 1.0, model.grid.Nx)
        return Burgers1DState(u=amplitude * jnp.sin(2.0 * jnp.pi * x))

    def test_viscosity_round_trips(self):
        dim = self.dimensional()
        assert self.U * self.L / float(dim.params.nu) == pytest.approx(
            self.RE, rel=1e-5
        )

    def test_trajectories_agree_under_the_scales_mapping(self):
        dim = self.dimensional()
        nd, _ = Burgers1D.from_nondimensional(
            nx=self.NX, reynolds=self.RE, periodic=True
        )
        scales = Scales.advective(L=self.L, U=self.U)

        state_nd = self.initial(nd, 1.0)
        state_dim = self.initial(dim, self.U)

        t_nd, n = 0.2, 200
        sol_nd = nd.integrate(state_nd, t0=0.0, t1=t_nd, dt=t_nd / n).ys
        sol_dim = dim.integrate(
            state_dim, t0=0.0, t1=t_nd * scales.T, dt=t_nd * scales.T / n
        ).ys
        assert relative_error(np.asarray(sol_nd.u) * self.U, sol_dim.u) < 1e-4

    def test_the_state_actually_evolves(self):
        dim = self.dimensional()
        scales = Scales.advective(L=self.L, U=self.U)
        state = self.initial(dim, self.U)
        out = dim.integrate(
            state, t0=0.0, t1=0.2 * scales.T, dt=0.2 * scales.T / 200
        ).ys
        change = np.abs(np.asarray(out.u) - np.asarray(state.u)).max()
        assert change > 0.05 * np.abs(np.asarray(state.u)).max()


class TestDimensionalEquivalenceDiffusion:
    """A diffusive-set run reproduces its dimensional twin."""

    L = 3.0
    KAPPA = 0.02
    NX = 64

    def test_trajectories_agree_under_the_scales_mapping(self):
        dim = Diffusion1D.create(nx=self.NX, Lx=self.L, nu=self.KAPPA, periodic=True)
        nd, _ = Diffusion1D.from_nondimensional(nx=self.NX, periodic=True)
        scales = Scales.diffusive(L=self.L, kappa=self.KAPPA)

        x = jnp.linspace(0.0, 1.0, dim.grid.Nx)
        profile = jnp.sin(2.0 * jnp.pi * x)
        amplitude = 0.7
        state_nd = Diffusion1DState(u=profile)
        state_dim = Diffusion1DState(u=amplitude * profile)

        t_nd, n = 0.02, 400
        sol_nd = nd.integrate(state_nd, t0=0.0, t1=t_nd, dt=t_nd / n).ys
        sol_dim = dim.integrate(
            state_dim, t0=0.0, t1=t_nd * scales.T, dt=t_nd * scales.T / n
        ).ys
        # Diffusion is linear, so the amplitude is carried through.
        assert relative_error(np.asarray(sol_nd.u) * amplitude, sol_dim.u) < 1e-4

    def test_the_state_actually_decays(self):
        dim = Diffusion1D.create(nx=self.NX, Lx=self.L, nu=self.KAPPA, periodic=True)
        scales = Scales.diffusive(L=self.L, kappa=self.KAPPA)
        x = jnp.linspace(0.0, 1.0, dim.grid.Nx)
        state = Diffusion1DState(u=jnp.sin(2.0 * jnp.pi * x))
        out = dim.integrate(
            state, t0=0.0, t1=0.02 * scales.T, dt=0.02 * scales.T / 400
        ).ys
        assert (
            np.abs(np.asarray(out.u)).max() < 0.95 * np.abs(np.asarray(state.u)).max()
        )


class TestCflStepInTheCorrectDirection:
    """The conversion into the set's time unit was inverted.

    Converting a dimensional bound ``C dx / U`` into units of ``T``
    divides by ``T``, so the factor is ``(L/U)/T`` — the reciprocal of
    what was applied. Only the advective set, where that factor is
    exactly 1, was unaffected.
    """

    def test_the_inertial_step_grows_as_the_rossby_number_shrinks(self):
        """A slower flow may take longer steps, not shorter ones."""
        slow = Scales.inertial(L=1.0e6, f0=1.0e-4, H=500.0, rossby=0.001)
        fast = Scales.inertial(L=1.0e6, f0=1.0e-4, H=500.0, rossby=0.1)
        assert slow.dt_from_cfl(0.5, 100) > fast.dt_from_cfl(0.5, 100)

    def test_the_inertial_step_is_the_nondimensional_courant_bound(self):
        rossby = 0.01
        scales = Scales.inertial(L=1.0e6, f0=1.0e-4, H=500.0, rossby=rossby)
        # u_hat = U T / L = Ro for this set.
        assert scales.dt_from_cfl(0.5, 100) == pytest.approx(0.5 / 100 / rossby)

    def test_the_planetary_step_uses_twice_the_rossby_number(self):
        """``U = 2 Omega a Ro``, so ``u_hat`` is ``2 Ro``, not ``Ro``."""
        rossby = 0.05
        scales = Scales.planetary(a=6.371e6, Omega=7.292e-5, H=3000.0, rossby=rossby)
        assert scales.dt_from_cfl(0.5, 100) == pytest.approx(0.5 / 100 / (2 * rossby))

    def test_the_advective_set_is_unchanged(self):
        scales = Scales.advective(L=2.0, U=1.0)
        assert scales.dt_from_cfl(0.4, 100) == pytest.approx(0.4 / 100)

    def test_the_bound_is_advective_only(self):
        """It bounds advection, so a caller must still cover waves.

        A shallow-water run is limited by its gravity wave, whose
        nondimensional speed is ``sqrt(Bu)/Ro`` — much faster than the
        flow at small Rossby number. ``dt_from_cfl`` deliberately does
        not guess at that; this pins the ratio so the omission stays
        visible.
        """
        rossby, burger = 0.1, 1.0
        scales = Scales.inertial(L=1.0, f0=1.0, H=1.0, rossby=rossby, g=burger)
        advective = scales.dt_from_cfl(0.2, 32)

        # Nondimensional speeds: the flow is Ro, the gravity wave
        # sqrt(Bu)/Ro. The two bounds differ by their ratio.
        flow_speed = rossby
        wave_speed = np.sqrt(burger) / rossby
        wave_bound = 0.2 / 32 / wave_speed
        assert advective / wave_bound == pytest.approx(wave_speed / flow_speed)


class TestCflAccountsForEveryAxis:
    """A 2-D bound depends on both spacings, not just one.

    With a flat domain the ``y`` spacing can be far smaller than the
    ``x`` one, and a single-axis bound then returns a step orders of
    magnitude too large.
    """

    def test_a_flat_domain_gives_a_smaller_step(self):
        scales = Scales.diffusive(L=1.0, kappa=1.0)
        square = scales.dt_from_cfl(0.5, (64, 64))
        flat = scales.dt_from_cfl(0.5, (64, 64), extent=(1.0, 0.1))
        assert flat < square / 50

    def test_the_one_dimensional_bound_is_recovered(self):
        scales = Scales.diffusive(L=1.0, kappa=1.0)
        assert scales.dt_from_cfl(0.5, 100) == pytest.approx(0.5 * (1 / 100) ** 2 / 2)

    def test_two_equal_axes_halve_the_one_dimensional_bound(self):
        """``sum 1/dx_i**2`` doubles when a second identical axis appears."""
        scales = Scales.diffusive(L=1.0, kappa=1.0)
        assert scales.dt_from_cfl(0.5, (100, 100)) == pytest.approx(
            scales.dt_from_cfl(0.5, 100) / 2
        )

    def test_the_advective_bound_is_dominated_by_the_smallest_spacing(self):
        """But not equal to it — every axis contributes to the sum."""
        scales = Scales.advective(L=1.0, U=1.0)
        got = scales.dt_from_cfl(0.5, (10, 100))
        assert got == pytest.approx(0.5 / np.sqrt(10.0**2 + 100.0**2))
        assert got < 0.5 / 100

    def test_extent_must_match_the_axes(self):
        scales = Scales.advective(L=1.0, U=1.0)
        with pytest.raises(ValueError, match="names 1 axes"):
            scales.dt_from_cfl(0.5, (10, 10), extent=1.0)

    def test_a_non_positive_cell_count_is_rejected(self):
        scales = Scales.advective(L=1.0, U=1.0)
        with pytest.raises(ValueError, match=r"n_cells\[1\]"):
            scales.dt_from_cfl(0.5, (10, 0))


class TestDiffusiveBoundNeedsTheModelsDiffusivity:
    """``L * U`` is not the diffusivity the pde models carry.

    Their nondimensional diffusivity is ``1/Re``, and ``L * U`` is
    identically 1 for the scales they return — so the bound came out a
    factor of ``Re`` too large.
    """

    def test_it_refuses_to_guess_on_a_non_diffusive_set(self):
        scales = Scales.advective(L=1.0, U=1.0)
        with pytest.raises(ValueError, match="nondimensional diffusivity"):
            scales.dt_from_cfl(0.5, 64, mode="diffusive")

    def test_a_diffusive_set_still_knows_its_own(self):
        scales = Scales.diffusive(L=1.0, kappa=1.0)
        assert scales.dt_from_cfl(0.5, 64) > 0.0

    def test_the_reynolds_number_enters_the_bound(self):
        _, scales = Burgers1D.from_nondimensional(nx=64, reynolds=100.0)
        low = scales.dt_from_cfl(0.4, 64, mode="diffusive", diffusivity=1 / 10.0)
        high = scales.dt_from_cfl(0.4, 64, mode="diffusive", diffusivity=1 / 1000.0)
        assert high == pytest.approx(100.0 * low)

    def test_the_bound_matches_the_explicit_formula(self):
        _, scales = Burgers1D.from_nondimensional(nx=64, reynolds=100.0)
        got = scales.dt_from_cfl(0.4, 64, mode="diffusive", diffusivity=1 / 100.0)
        assert got == pytest.approx(0.4 * (1 / 64) ** 2 / (2 / 100.0))

    def test_a_low_reynolds_run_stays_finite_at_the_diffusive_bound(self):
        model, scales = Burgers1D.from_nondimensional(nx=64, reynolds=2.0)
        dt = scales.dt_from_cfl(0.4, 64, mode="diffusive", diffusivity=1 / 2.0)
        x = jnp.linspace(0.0, 1.0, model.grid.Nx)
        state = Burgers1DState(u=jnp.sin(2.0 * jnp.pi * x))
        out = model.integrate(state, t0=0.0, t1=20 * dt, dt=dt).ys
        assert np.isfinite(np.asarray(out.u)).all()


class TestConvectionKeepsItsDirection:
    """Scaling out the speed must not scale out the direction.

    ``(1, 0)``, ``(1, 1)`` and ``(-1, 2)`` are different problems on
    the same grid; only their common magnitude is a unit choice.
    """

    def test_a_purely_zonal_direction_is_representable(self):
        model, _ = LinearConvection2D.from_nondimensional(
            nx=16, ny=16, direction=(1.0, 0.0)
        )
        assert float(model.params.cx) == pytest.approx(1.0)
        assert float(model.params.cy) == pytest.approx(0.0)

    def test_an_unequal_ratio_is_preserved(self):
        model, _ = LinearConvection2D.from_nondimensional(
            nx=16, ny=16, direction=(3.0, 4.0)
        )
        assert float(model.params.cx) == pytest.approx(0.6)
        assert float(model.params.cy) == pytest.approx(0.8)

    def test_the_speed_is_still_one(self):
        model, _ = LinearConvection2D.from_nondimensional(
            nx=16, ny=16, direction=(-1.0, 2.0)
        )
        speed = np.hypot(float(model.params.cx), float(model.params.cy))
        assert speed == pytest.approx(1.0)

    def test_negative_propagation_is_representable(self):
        model, _ = LinearConvection2D.from_nondimensional(
            nx=16, ny=16, direction=(-1.0, 0.0)
        )
        assert float(model.params.cx) == pytest.approx(-1.0)

    def test_the_default_is_the_previous_behaviour(self):
        """Diagonal, now normalised so the speed really is the scale."""
        model, _ = LinearConvection2D.from_nondimensional(nx=16, ny=16)
        assert float(model.params.cx) == pytest.approx(float(model.params.cy))
        assert float(model.params.cx) == pytest.approx(1.0 / np.sqrt(2.0))

    def test_the_zero_vector_is_rejected(self):
        with pytest.raises(ValueError, match="zero vector"):
            LinearConvection2D.from_nondimensional(nx=8, ny=8, direction=(0.0, 0.0))

    def test_a_non_finite_direction_is_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            LinearConvection2D.from_nondimensional(
                nx=8, ny=8, direction=(float("nan"), 1.0)
            )

    def test_the_one_dimensional_sign_is_preserved(self):
        model, _ = LinearConvection1D.from_nondimensional(nx=32, direction=-2.0)
        assert float(model.params.c) == pytest.approx(-1.0)

    def test_the_one_dimensional_default_is_rightward(self):
        model, _ = LinearConvection1D.from_nondimensional(nx=32)
        assert float(model.params.c) == pytest.approx(1.0)

    def test_a_zero_one_dimensional_direction_is_rejected(self):
        with pytest.raises(ValueError, match="non-zero"):
            LinearConvection1D.from_nondimensional(nx=32, direction=0.0)


#: Every model whose factory advertises forwarded ``create`` kwargs.
FACTORY_CLASSES = [
    Diffusion1D,
    LinearConvection1D,
    Burgers1D,
    NonlinearConvection1D,
    Diffusion2D,
    LinearConvection2D,
    Burgers2D,
    NonlinearConvection2D,
    IncompressibleNS2D,
]


class TestDocumentedKwargsAreAccepted:
    """Every forwarded argument a factory advertises must exist.

    ``method`` was listed on four factories whose ``create()`` has no
    such parameter, so following the docstring raised ``TypeError``.
    """

    @pytest.mark.parametrize(
        "cls", FACTORY_CLASSES, ids=[c.__name__ for c in FACTORY_CLASSES]
    )
    def test_documented_kwargs_exist_on_create(self, cls):
        import inspect
        import re

        name = cls.__name__
        doc = cls.from_nondimensional.__doc__
        match = re.search(
            r"\*\*create_kw: Forwarded to :meth:`create` \(([^)]*)\)", doc, re.S
        )
        assert match, f"{name}: no forwarded-kwargs line to check"
        documented = set(re.findall(r"``(\w+)``", match.group(1)))
        accepted = set(inspect.signature(cls.create).parameters)
        assert documented <= accepted, (
            f"{name} advertises {sorted(documented - accepted)}, which "
            f"create() does not accept"
        )


class TestAdvectiveBoundAccountsForDirection:
    """Courant is ``dt * sum_i |c_i|/dx_i``, not ``dt * |c|/min(dx)``.

    On a square grid the default diagonal convection direction has
    ``cx = cy = 1/sqrt(2)``, so a step taken from the minimum spacing
    alone overshoots the requested Courant number by ``sqrt(2)``.
    """

    def scales(self):
        return Scales.advective(L=1.0, U=1.0)

    def courant(self, dt, spacings, components):
        return dt * sum(abs(c) / s for c, s in zip(components, spacings, strict=True))

    def test_the_default_diagonal_no_longer_overshoots(self):
        dt = self.scales().dt_from_cfl(0.5, (100, 100))
        unit = 1.0 / np.sqrt(2.0)
        got = self.courant(dt, (0.01, 0.01), (unit, unit))
        assert got == pytest.approx(0.5, rel=1e-6)

    def test_the_old_formula_really_did_overshoot(self):
        """Otherwise the test above would pass for the wrong reason."""
        unit = 1.0 / np.sqrt(2.0)
        naive_dt = 0.5 * 0.01  # C * min(dx) / |c|, with |c| = 1
        got = self.courant(naive_dt, (0.01, 0.01), (unit, unit))
        assert got == pytest.approx(0.5 * np.sqrt(2.0), rel=1e-6)

    def test_an_axis_aligned_direction_gets_the_exact_bound(self):
        dt = self.scales().dt_from_cfl(0.5, (100, 100), direction=(1.0, 0.0))
        assert dt == pytest.approx(0.5 / 100)
        assert self.courant(dt, (0.01, 0.01), (1.0, 0.0)) == pytest.approx(0.5)

    def test_a_known_direction_is_never_more_restrictive_than_the_default(self):
        scales = self.scales()
        default = scales.dt_from_cfl(0.5, (100, 100))
        for direction in ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (3.0, 4.0)):
            assert scales.dt_from_cfl(0.5, (100, 100), direction=direction) >= default

    def test_the_direction_magnitude_does_not_matter(self):
        scales = self.scales()
        one = scales.dt_from_cfl(0.5, (100, 100), direction=(1.0, 2.0))
        ten = scales.dt_from_cfl(0.5, (100, 100), direction=(10.0, 20.0))
        assert one == pytest.approx(ten)

    def test_a_negative_component_behaves_like_its_magnitude(self):
        scales = self.scales()
        assert scales.dt_from_cfl(
            0.5, (100, 100), direction=(-1.0, 0.0)
        ) == pytest.approx(scales.dt_from_cfl(0.5, (100, 100), direction=(1.0, 0.0)))

    def test_the_one_dimensional_bound_is_unchanged(self):
        assert self.scales().dt_from_cfl(0.5, 100) == pytest.approx(0.5 / 100)

    def test_a_mismatched_direction_is_rejected(self):
        with pytest.raises(ValueError, match="direction names"):
            self.scales().dt_from_cfl(0.5, (10, 10), direction=(1.0,))

    def test_the_zero_vector_is_rejected(self):
        with pytest.raises(ValueError, match="zero vector"):
            self.scales().dt_from_cfl(0.5, (10, 10), direction=(0.0, 0.0))

    def test_a_non_finite_direction_is_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            self.scales().dt_from_cfl(0.5, (10, 10), direction=(float("nan"), 1.0))

    def test_the_diffusive_bound_ignores_direction(self):
        """Diffusion is isotropic, so a direction says nothing about it."""
        scales = Scales.diffusive(L=1.0, kappa=1.0)
        assert scales.dt_from_cfl(0.5, (50, 50)) == pytest.approx(
            scales.dt_from_cfl(0.5, (50, 50), direction=(1.0, 0.0))
        )

    def test_the_convection_factory_direction_can_be_reused(self):
        """The factory and the step helper speak the same language."""
        model, scales = LinearConvection2D.from_nondimensional(
            nx=64, ny=64, direction=(1.0, 0.0)
        )
        dt = scales.dt_from_cfl(0.4, (64, 64), direction=(1.0, 0.0))
        assert dt == pytest.approx(0.4 / 64)
        assert float(model.params.cy) == pytest.approx(0.0)


class TestDirectionNormalisationSurvivesExtremeComponents:
    """The direction is documented as magnitude-invariant.

    ``sqrt(sum(c*c))`` breaks that at both ends: squaring components
    around ``1e200`` overflows, giving an infinite speed and a
    direction of all zeros, while components around ``1e-200``
    underflow to a zero speed and are rejected as if the vector were
    the zero vector. ``math.hypot`` does neither.

    (In CPython the overflowing square raises ``OverflowError`` rather
    than returning infinity — broken either way, just more loudly.)
    """

    scales = Scales.advective(L=1.0, U=1.0, f0=1.0, H=1.0)

    def reference(self):
        return self.scales.dt_from_cfl(0.4, (16, 16), direction=(3.0, 4.0))

    @pytest.mark.parametrize("magnitude", [1e-200, 1e-30, 1.0, 1e30, 1e200])
    def test_the_step_is_the_same_at_any_magnitude(self, magnitude):
        scaled = self.scales.dt_from_cfl(
            0.4, (16, 16), direction=(3.0 * magnitude, 4.0 * magnitude)
        )
        assert scaled == pytest.approx(self.reference(), rel=1e-12)

    def test_squaring_really_does_break(self):
        """Otherwise the parametrization above would prove nothing."""
        with pytest.raises(OverflowError):
            _ = (3.0e200) ** 2
        # And at the other end it silently underflows, which is worse:
        # the sum is zero and the vector looks like the zero vector.
        assert (3.0e-200) ** 2 == 0.0

    def test_the_zero_vector_is_still_rejected(self):
        with pytest.raises(ValueError, match="zero vector"):
            self.scales.dt_from_cfl(0.4, (16, 16), direction=(0.0, 0.0))
