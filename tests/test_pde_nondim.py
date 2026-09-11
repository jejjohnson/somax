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

    def test_linear_convection_2d_speeds_are_unity(self):
        model, _ = LinearConvection2D.from_nondimensional(nx=16, ny=16)
        assert float(model.params.cx) == pytest.approx(1.0)
        assert float(model.params.cy) == pytest.approx(1.0)


class TestCflHelper:
    def test_advective_step(self):
        scales = Scales.advective(L=2.0, U=1.0)
        assert scales.dt_from_cfl(0.4, 100) == pytest.approx(0.4 / 100)

    def test_diffusive_step(self):
        scales = Scales.diffusive(L=2.0, kappa=0.01)
        assert scales.dt_from_cfl(0.4, 100) == pytest.approx(0.4 / 100**2 / 2)

    def test_mode_can_be_forced(self):
        scales = Scales.advective(L=1.0, U=1.0)
        assert scales.dt_from_cfl(0.5, 10, mode="diffusive") == pytest.approx(
            0.5 * (1 / 10) ** 2 / 2
        )

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
