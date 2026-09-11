"""Tests for the barotropic QG nondimensional factory and its guards."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from somax._src.cli._assertions import (
    AssertionFailedError,
    check_munk_width,
    check_stommel_width,
)
from somax._src.core.scales import Scales
from somax._src.core.transforms import StateAffine
from somax._src.models.qg.barotropic import BarotropicQG, BarotropicQGState


class TestParameterMapping:
    """Each dimensionless input lands on the right ``create()`` kwarg."""

    @pytest.fixture
    def built(self):
        return BarotropicQG.from_nondimensional(
            nx=32,
            ny=32,
            rossby=0.02,
            beta_hat=50.0,
            delta_M=0.1,
            delta_S=0.1,
            check_resolution=False,
        )

    def test_rossby_sets_the_coriolis_parameter(self, built):
        model, _ = built
        assert float(model.consts.f0) == pytest.approx(1.0 / 0.02)

    def test_beta_hat_is_beta_directly(self, built):
        model, _ = built
        assert float(model.consts.beta) == pytest.approx(50.0)

    def test_delta_M_sets_the_viscosity(self, built):
        model, _ = built
        assert float(model.params.lateral_viscosity) == pytest.approx(
            0.1**3 * 50.0, rel=1e-6
        )

    def test_delta_S_sets_the_bottom_drag(self, built):
        model, _ = built
        assert float(model.params.bottom_drag) == pytest.approx(0.1 * 50.0, rel=1e-6)

    def test_domain_is_unit_length(self, built):
        model, _ = built
        assert model.grid.Lx == pytest.approx(1.0)
        assert model.grid.Ly == pytest.approx(1.0)

    def test_aspect_stretches_only_y(self):
        model, _ = BarotropicQG.from_nondimensional(
            nx=32,
            ny=16,
            rossby=0.02,
            beta_hat=50.0,
            delta_M=0.1,
            aspect=0.5,
            check_resolution=False,
        )
        assert model.grid.Lx == pytest.approx(1.0)
        assert model.grid.Ly == pytest.approx(0.5)

    def test_returns_the_advective_unit_scale_set(self, built):
        _, scales = built
        assert scales.kind == "advective"
        assert scales.L == 1.0
        assert scales.U == 1.0
        assert scales.T == 1.0
        assert scales.rossby == pytest.approx(0.02)

    def test_recovers_delta_M_from_the_built_model(self, built):
        """The round trip a resolution guard relies on."""
        model, _ = built
        nu = float(model.params.lateral_viscosity)
        beta = float(model.consts.beta)
        assert (nu / beta) ** (1 / 3) == pytest.approx(0.1, rel=1e-5)


class TestWindAmplitude:
    def test_defaults_to_the_sverdrup_consistent_amplitude(self):
        """tau_hat = beta_hat makes the Sverdrup interior velocity exactly U."""
        model, _ = BarotropicQG.from_nondimensional(
            nx=32,
            ny=32,
            rossby=0.02,
            beta_hat=50.0,
            delta_M=0.1,
            check_resolution=False,
        )
        assert float(model.params.wind_amplitude) == pytest.approx(50.0)

    def test_delta_I_sets_the_amplitude(self):
        model, _ = BarotropicQG.from_nondimensional(
            nx=32,
            ny=32,
            rossby=0.02,
            beta_hat=50.0,
            delta_M=0.1,
            delta_I=0.2,
            check_resolution=False,
        )
        assert float(model.params.wind_amplitude) == pytest.approx(
            0.2**2 * 50.0**2, rel=1e-5
        )

    def test_the_default_matches_the_consistent_delta_I(self):
        """delta_I = beta_hat**-0.5 is the value the default corresponds to."""
        beta_hat = 50.0
        kw = {
            "nx": 32,
            "ny": 32,
            "rossby": 0.02,
            "beta_hat": beta_hat,
            "delta_M": 0.1,
            "check_resolution": False,
        }
        default, _ = BarotropicQG.from_nondimensional(**kw)
        explicit, _ = BarotropicQG.from_nondimensional(**kw, delta_I=beta_hat**-0.5)
        assert float(explicit.params.wind_amplitude) == pytest.approx(
            float(default.params.wind_amplitude), rel=1e-5
        )


class TestValidation:
    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"rossby": 0.0}, "rossby"),
            ({"rossby": -1.0}, "rossby"),
            ({"beta_hat": 0.0}, "beta_hat"),
            ({"delta_M": -0.1}, "delta_M"),
            ({"delta_S": -0.1}, "delta_S"),
            ({"delta_I": 0.0}, "delta_I"),
            ({"aspect": 0.0}, "aspect"),
        ],
    )
    def test_rejects_out_of_range_inputs(self, kwargs, message):
        base = {
            "nx": 16,
            "ny": 16,
            "rossby": 0.02,
            "beta_hat": 50.0,
            "delta_M": 0.1,
            "check_resolution": False,
        }
        with pytest.raises(ValueError, match=message):
            BarotropicQG.from_nondimensional(**{**base, **kwargs})


class TestResolutionGuards:
    def test_unresolved_munk_layer_is_rejected(self):
        """delta_M * nx < 2 means the boundary current has nowhere to live."""
        with pytest.raises(AssertionFailedError, match="munk_width check FAILED"):
            BarotropicQG.from_nondimensional(
                nx=16,
                ny=16,
                rossby=0.02,
                beta_hat=50.0,
                delta_M=0.01,  # 0.16 cells wide
            )

    def test_resolved_munk_layer_passes(self):
        model, _ = BarotropicQG.from_nondimensional(
            nx=64, ny=64, rossby=0.02, beta_hat=50.0, delta_M=0.1
        )
        assert model is not None

    def test_unresolved_stommel_layer_is_rejected(self):
        with pytest.raises(AssertionFailedError, match="stommel_width check FAILED"):
            BarotropicQG.from_nondimensional(
                nx=32,
                ny=32,
                rossby=0.02,
                beta_hat=50.0,
                delta_M=0.15,
                delta_S=0.001,  # 0.03 cells wide
            )

    def test_stommel_guard_is_skipped_when_there_is_no_drag(self):
        """delta_S = 0 is a legitimate Munk-only run, not a failure."""
        model, _ = BarotropicQG.from_nondimensional(
            nx=64, ny=64, rossby=0.02, beta_hat=50.0, delta_M=0.1, delta_S=0.0
        )
        assert float(model.params.bottom_drag) == 0.0

    def test_check_resolution_false_allows_a_coarse_model(self):
        model, _ = BarotropicQG.from_nondimensional(
            nx=16,
            ny=16,
            rossby=0.02,
            beta_hat=50.0,
            delta_M=0.001,
            check_resolution=False,
        )
        assert model is not None

    def test_guard_message_names_a_usable_remedy(self):
        with pytest.raises(AssertionFailedError) as excinfo:
            BarotropicQG.from_nondimensional(
                nx=16, ny=16, rossby=0.02, beta_hat=50.0, delta_M=0.01
            )
        message = str(excinfo.value)
        assert "delta_M/dx" in message
        assert "nu >=" in message


class TestGuardsOnDimensionalModels:
    """The guards compare two lengths, so they work in SI units too."""

    def test_munk_guard_on_a_dimensional_model(self):
        # delta_M = (nu/beta)^(1/3) = (1e3/1.6e-11)^(1/3) ~ 3.97e4 m,
        # dx = 1e6/16 = 6.25e4 m -> 0.63 cells: unresolved.
        model = BarotropicQG.create(
            nx=16, ny=16, Lx=1e6, Ly=1e6, beta=1.6e-11, lateral_viscosity=1e3
        )
        with pytest.raises(AssertionFailedError, match="munk_width check FAILED"):
            check_munk_width(None, model)

    def test_munk_guard_passes_on_a_fine_dimensional_grid(self):
        model = BarotropicQG.create(
            nx=256, ny=256, Lx=1e6, Ly=1e6, beta=1.6e-11, lateral_viscosity=1e3
        )
        check_munk_width(None, model)

    def test_zero_viscosity_is_reported_not_silently_skipped(self):
        model = BarotropicQG.create(nx=64, ny=64, lateral_viscosity=0.0)
        with pytest.raises(AssertionFailedError, match="lateral_viscosity is zero"):
            check_munk_width(None, model)

    def test_zero_drag_is_reported_not_silently_skipped(self):
        model = BarotropicQG.create(nx=64, ny=64, bottom_drag=0.0)
        with pytest.raises(AssertionFailedError, match="bottom_drag is zero"):
            check_stommel_width(None, model)

    def test_f_plane_has_no_boundary_layer(self):
        model = BarotropicQG.create(nx=64, ny=64, beta=0.0, lateral_viscosity=1e3)
        with pytest.raises(AssertionFailedError, match=r"consts\.beta is zero"):
            check_munk_width(None, model)

    def test_guards_reject_a_model_without_the_needed_structure(self):
        class Bare:
            pass

        with pytest.raises(AssertionFailedError, match="lacks params/consts/grid"):
            check_munk_width(None, Bare())

    def test_guards_see_through_constrained_parameters(self):
        """A positive()-wrapped viscosity must still be checked."""
        import equinox as eqx

        from somax._src.core.types import positive

        model = BarotropicQG.create(
            nx=16, ny=16, Lx=1e6, Ly=1e6, beta=1.6e-11, lateral_viscosity=1e3
        )
        wrapped = eqx.tree_at(
            lambda m: m.params.lateral_viscosity, model, positive(1e3)
        )
        with pytest.raises(AssertionFailedError, match="munk_width check FAILED"):
            check_munk_width(None, wrapped)


class TestDimensionalEquivalence:
    """A nondimensional run reproduces its dimensional counterpart.

    This is the property the whole factory rests on: the RHS reads only
    ``grid.dx/dy``, ``consts`` and ``params``, so rebuilding at unit
    scales must integrate the same equation.

    With ``L``, ``U`` chosen and ``H = 1``:

        f0    = U / (Ro L)          beta = beta_hat U / L**2
        nu    = delta_M**3 L**3 beta kappa = delta_S beta L
        tau0  = tau_hat U**2 / L**2

    and the states map as ``q_dim = (U/L) q_nd`` with ``t_dim = (L/U) t_nd``.
    """

    L = 1.0e6
    U = 0.05
    ROSSBY = 0.02
    BETA_HAT = 20.0
    DELTA_M = 0.08
    DELTA_S = 0.05
    NX = NY = 48

    def dimensional_model(self):
        beta = self.BETA_HAT * self.U / self.L**2
        return BarotropicQG.create(
            nx=self.NX,
            ny=self.NY,
            Lx=self.L,
            Ly=self.L,
            f0=self.U / (self.ROSSBY * self.L),
            beta=beta,
            lateral_viscosity=self.DELTA_M**3 * self.L**3 * beta,
            bottom_drag=self.DELTA_S * beta * self.L,
            wind_amplitude=self.BETA_HAT * self.U**2 / self.L**2,
        )

    def nondimensional_model(self):
        return BarotropicQG.from_nondimensional(
            nx=self.NX,
            ny=self.NY,
            rossby=self.ROSSBY,
            beta_hat=self.BETA_HAT,
            delta_M=self.DELTA_M,
            delta_S=self.DELTA_S,
        )

    def initial_state(self, grid, amplitude):
        rng = np.random.RandomState(0)
        field = rng.randn(grid.Ny, grid.Nx)
        field -= field.mean()
        return BarotropicQGState(q=jnp.asarray(amplitude * field))

    def test_coefficients_round_trip(self):
        """The dimensional coefficients recover the dimensionless numbers."""
        dim = self.dimensional_model()
        beta = float(dim.consts.beta)
        nu = float(dim.params.lateral_viscosity)
        kappa = float(dim.params.bottom_drag)
        assert (nu / beta) ** (1 / 3) / self.L == pytest.approx(self.DELTA_M, rel=1e-4)
        assert kappa / (beta * self.L) == pytest.approx(self.DELTA_S, rel=1e-4)
        assert self.U / (float(dim.consts.f0) * self.L) == pytest.approx(
            self.ROSSBY, rel=1e-6
        )

    def test_trajectories_agree_under_the_scales_mapping(self):
        dim = self.dimensional_model()
        nd, _ = self.nondimensional_model()
        scales = Scales.advective(L=self.L, U=self.U, f0=float(dim.consts.f0))
        transform = StateAffine.from_scales(BarotropicQGState, scales)

        q_scale = scales.vorticity
        state_dim = self.initial_state(dim.grid, 0.05 * q_scale)
        state_nd = transform.forward(state_dim)

        t_nd = 0.4
        n_steps = 40
        sol_nd = nd.integrate(state_nd, t0=0.0, t1=t_nd, dt=t_nd / n_steps)
        sol_dim = dim.integrate(
            state_dim,
            t0=0.0,
            t1=t_nd * scales.T,
            dt=t_nd * scales.T / n_steps,
        )

        got = np.asarray(transform.inverse(sol_nd.ys).q)
        expected = np.asarray(sol_dim.ys.q)
        assert np.isfinite(got).all()
        rel = np.abs(got - expected).max() / np.abs(expected).max()
        assert rel < 2e-3, f"nondim and dimensional runs diverged: rel={rel:.3e}"

    def test_the_two_runs_are_not_trivially_equal(self):
        """Guard against the comparison passing because nothing happened."""
        dim = self.dimensional_model()
        scales = Scales.advective(L=self.L, U=self.U, f0=float(dim.consts.f0))
        state = self.initial_state(dim.grid, 0.05 * scales.vorticity)
        evolved = dim.integrate(
            state, t0=0.0, t1=0.4 * scales.T, dt=0.4 * scales.T / 40
        ).ys.q
        change = np.abs(np.asarray(evolved) - np.asarray(state.q)).max()
        assert change > 0.05 * np.abs(np.asarray(state.q)).max()


class TestAssertionRegistry:
    def test_new_guards_are_registered(self):
        from somax._src.cli._assertions import PREFLIGHT_ASSERTIONS

        assert PREFLIGHT_ASSERTIONS["munk_width"] is check_munk_width
        assert PREFLIGHT_ASSERTIONS["stommel_width"] is check_stommel_width
