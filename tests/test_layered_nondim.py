"""Tests for the layered models' nondimensional factories.

Shallow water uses the inertial scale set, the two baroclinic QG models
the advective one. Both take Burger numbers **per interface**, which
invert to the stratification one-to-one; the per-mode deformation radii
that follow from the eigenproblem are a different quantity and are
checked separately.
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp
import numpy as np
import pytest

from somax._src.cli._assertions import AssertionFailedError
from somax._src.core.scales import Scales
from somax._src.core.transforms import StateAffine
from somax._src.models.qg.baroclinic import BaroclinicQG, BaroclinicQGState
from somax._src.models.qg.reparameterized import ReparameterizedQG
from somax._src.models.swm.multilayer import (
    MultilayerShallowWater2D,
    MultilayerSW2DState,
)
from somax._src.models.swm.nonlinear_2d import (
    NonlinearShallowWater2D,
    NonlinearSW2DState,
)


def relative_error(got, expected):
    got, expected = np.asarray(got), np.asarray(expected)
    return np.abs(got - expected).max() / np.abs(expected).max()


class TestNonlinearShallowWaterMapping:
    @pytest.fixture
    def built(self):
        return NonlinearShallowWater2D.from_nondimensional(
            nx=16,
            ny=16,
            rossby=0.05,
            burger=2.0,
            beta_hat=0.5,
            ekman=1e-3,
            ekman_lateral=2e-3,
            wind_hat=0.25,
        )

    def test_unit_coriolis_and_domain(self, built):
        model, _ = built
        assert float(model.consts.f0) == pytest.approx(1.0)
        assert model.grid.Lx == pytest.approx(1.0)

    def test_burger_becomes_gravity(self, built):
        model, _ = built
        assert float(model.consts.gravity) == pytest.approx(2.0)

    def test_ekman_numbers_map_to_drag_and_viscosity(self, built):
        model, _ = built
        assert float(model.params.bottom_drag) == pytest.approx(1e-3)
        assert float(model.params.lateral_viscosity) == pytest.approx(2e-3)

    def test_beta_hat_maps_to_beta(self, built):
        model, _ = built
        assert float(model.consts.beta) == pytest.approx(0.5)

    def test_wind_carries_the_velocity_scale(self, built):
        """tau_hat = tau0/(f0 U), so the model kwarg is tau_hat * Ro."""
        model, _ = built
        assert float(model.params.wind_amplitude) == pytest.approx(0.25 * 0.05)

    def test_velocity_scale_is_rossby(self, built):
        _, scales = built
        assert scales.kind == "inertial"
        assert pytest.approx(0.05) == scales.U
        assert pytest.approx(1.0) == scales.T

    def test_burger_round_trips_through_the_scales(self, built):
        _, scales = built
        assert scales.burger == pytest.approx(2.0, rel=1e-9)

    def test_froude_is_equivalent_to_burger(self):
        """Bu = (Ro/Fr)**2, so the two routes must agree."""
        rossby, burger = 0.05, 2.0
        froude = rossby / np.sqrt(burger)
        by_burger, _ = NonlinearShallowWater2D.from_nondimensional(
            nx=16, ny=16, rossby=rossby, burger=burger
        )
        by_froude, _ = NonlinearShallowWater2D.from_nondimensional(
            nx=16, ny=16, rossby=rossby, froude=float(froude)
        )
        assert float(by_froude.consts.gravity) == pytest.approx(
            float(by_burger.consts.gravity), rel=1e-6
        )

    def test_supplying_both_burger_and_froude_is_rejected(self):
        with pytest.raises(ValueError, match="exactly one of burger or froude"):
            NonlinearShallowWater2D.from_nondimensional(
                nx=16, ny=16, rossby=0.05, burger=1.0, froude=0.05
            )

    def test_supplying_neither_is_rejected(self):
        with pytest.raises(ValueError, match="exactly one of burger or froude"):
            NonlinearShallowWater2D.from_nondimensional(nx=16, ny=16, rossby=0.05)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"rossby": 0.0}, "rossby"),
            ({"ekman": -1.0}, "ekman"),
            ({"aspect": -1.0}, "aspect"),
            ({"burger": 0.0}, "burger"),
        ],
    )
    def test_rejects_out_of_range_inputs(self, kwargs, message):
        base = {"nx": 16, "ny": 16, "rossby": 0.05, "burger": 1.0}
        with pytest.raises(ValueError, match=message):
            NonlinearShallowWater2D.from_nondimensional(**{**base, **kwargs})


class TestMultilayerMapping:
    @pytest.fixture
    def built(self):
        return MultilayerShallowWater2D.from_nondimensional(
            nx=16,
            ny=16,
            rossby=0.05,
            burger=[1.0, 0.1],
            thickness_ratio=[1.0, 9.0],
            ekman=1e-3,
        )

    def test_thickness_ratio_becomes_the_layer_depths(self, built):
        model, _ = built
        np.testing.assert_allclose(np.asarray(model.strat.H), [1.0, 9.0])

    def test_burger_inverts_to_reduced_gravity(self, built):
        """g'_k = Bu_k (f0 L)**2 / H_k, with f0 = L = 1."""
        model, _ = built
        np.testing.assert_allclose(
            np.asarray(model.strat.g_prime), [1.0 / 1.0, 0.1 / 9.0], rtol=1e-6
        )

    def test_recovering_burger_from_the_model(self, built):
        model, _ = built
        g_prime = np.asarray(model.strat.g_prime)
        H = np.asarray(model.strat.H)
        f0 = float(model.consts.f0)
        np.testing.assert_allclose(g_prime * H / f0**2, [1.0, 0.1], rtol=1e-6)

    def test_mismatched_sequence_lengths_are_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            MultilayerShallowWater2D.from_nondimensional(
                nx=16, ny=16, rossby=0.05, burger=[1.0, 0.1], thickness_ratio=[1.0]
            )

    def test_non_positive_burger_is_rejected(self):
        with pytest.raises(ValueError, match=r"burger\[1\]"):
            MultilayerShallowWater2D.from_nondimensional(
                nx=16,
                ny=16,
                rossby=0.05,
                burger=[1.0, 0.0],
                thickness_ratio=[1.0, 9.0],
            )


class TestBaroclinicQGMapping:
    KW: ClassVar[dict] = {
        "nx": 64,
        "ny": 64,
        "rossby": 0.02,
        "beta_hat": 20.0,
        "burger": [1.0, 0.02],
        "thickness_ratio": [1.0, 4.0],
        "delta_M": 0.06,
    }

    @pytest.fixture
    def built(self):
        return BaroclinicQG.from_nondimensional(**self.KW)

    def test_rossby_sets_the_coriolis_parameter(self, built):
        model, _ = built
        assert float(model.consts.f0) == pytest.approx(1.0 / 0.02)

    def test_delta_M_sets_the_viscosity(self, built):
        model, _ = built
        assert float(model.params.lateral_viscosity) == pytest.approx(
            0.06**3 * 20.0, rel=1e-5
        )

    def test_wind_defaults_to_the_sverdrup_value(self, built):
        model, _ = built
        assert float(model.params.wind_amplitude) == pytest.approx(20.0)

    def test_advective_unit_scales(self, built):
        _, scales = built
        assert scales.kind == "advective"
        assert scales.T == 1.0
        assert scales.rossby == pytest.approx(0.02)

    def test_burger_inverts_with_the_model_f0(self, built):
        model, _ = built
        f0 = float(model.consts.f0)
        g_prime = np.asarray(model.strat.g_prime)
        H = np.asarray(model.strat.H)
        np.testing.assert_allclose(g_prime * H / f0**2, [1.0, 0.02], rtol=1e-5)

    def test_modal_radii_are_in_units_of_L(self, built):
        """So they are directly comparable with dx, which is what the guard does."""
        model, _ = built
        radii = np.asarray(model.modal.rossby_radii)
        internal = radii[np.isfinite(radii)]
        assert internal.size == 1
        assert 0.0 < internal[0] < 1.0

    def test_modal_radius_is_not_the_interface_burger(self):
        """The two conventions genuinely differ; the docs say so, so test it."""
        model, _ = BaroclinicQG.from_nondimensional(**self.KW)
        radii = np.asarray(model.modal.rossby_radii)
        internal = float(radii[np.isfinite(radii)][0])
        # sqrt of the second interface Burger number, the naive guess.
        assert internal != pytest.approx(np.sqrt(0.02), rel=1e-2)

    def test_modal_radius_is_near_the_rigid_lid_two_layer_formula(self):
        """L_d**2 = g' H1 H2 / (f0**2 (H1 + H2)), up to the free surface.

        The eigenproblem includes the free-surface mode through
        ``g_prime[0]``, so the internal radius sits a few percent below
        the rigid-lid value rather than on it.
        """
        model, _ = BaroclinicQG.from_nondimensional(**self.KW)
        g_prime = np.asarray(model.strat.g_prime)
        H = np.asarray(model.strat.H)
        f0 = float(model.consts.f0)
        rigid_lid = np.sqrt(g_prime[1] * H[0] * H[1] / (f0**2 * (H[0] + H[1])))
        radii = np.asarray(model.modal.rossby_radii)
        internal = float(radii[np.isfinite(radii)][0])
        assert internal == pytest.approx(rigid_lid, rel=0.1)
        assert internal < rigid_lid

    def test_reparameterized_uses_the_same_mapping(self):
        bc, _ = BaroclinicQG.from_nondimensional(**self.KW)
        rp, _ = ReparameterizedQG.from_nondimensional(**self.KW)
        np.testing.assert_allclose(
            np.asarray(rp.strat.g_prime), np.asarray(bc.strat.g_prime), rtol=1e-6
        )
        assert float(rp.consts.f0) == pytest.approx(float(bc.consts.f0))


class TestLayeredResolutionGuards:
    def test_unresolved_deformation_radius_is_rejected(self):
        """A tiny Burger number means the internal radius is sub-grid."""
        with pytest.raises(
            AssertionFailedError, match="deformation_radius check FAILED"
        ):
            BaroclinicQG.from_nondimensional(
                nx=32,
                ny=32,
                rossby=0.02,
                beta_hat=20.0,
                burger=[1.0, 1e-6],
                thickness_ratio=[1.0, 4.0],
                delta_M=0.1,
            )

    def test_unresolved_munk_layer_is_rejected(self):
        with pytest.raises(AssertionFailedError, match="munk_width check FAILED"):
            BaroclinicQG.from_nondimensional(
                nx=32,
                ny=32,
                rossby=0.02,
                beta_hat=20.0,
                burger=[1.0, 0.05],
                thickness_ratio=[1.0, 4.0],
                delta_M=0.005,
            )

    def test_check_resolution_false_skips_the_guards(self):
        model, _ = BaroclinicQG.from_nondimensional(
            nx=16,
            ny=16,
            rossby=0.02,
            beta_hat=20.0,
            burger=[1.0, 1e-6],
            thickness_ratio=[1.0, 4.0],
            delta_M=0.001,
            check_resolution=False,
        )
        assert model is not None


class TestShallowWaterDimensionalEquivalence:
    """An inertial-set nondimensional run reproduces its dimensional twin.

    With ``L``, ``f0``, ``H`` chosen and ``U = f0 L Ro``::

        g    = Bu (f0 L)**2 / H       nu    = Ek_lat f0 L**2
        beta = beta_hat f0 / L        kappa = Ek f0

    and the model's own units are ``H`` for thickness and ``L f0`` for
    velocity.

    The regime is deliberately Ro = 0.2, Bu = 1: the Bernoulli term
    carries ``g*h`` about its mean, so at small Rossby number the mean
    thickness swamps the anomaly and float32 cancellation alone puts a
    few times 1e-4 between two algebraically identical runs. At Ro = 0.2
    the mean is only five times the anomaly and the comparison is clean.
    Thickness error is measured against the anomaly for the same reason:
    dividing by a mean of 500 would flatter any disagreement.
    """

    L = 1.0e6
    F0 = 1.0e-4
    H = 500.0
    ROSSBY = 0.2
    BURGER = 1.0
    EKMAN_LATERAL = 2.0e-3
    NX = NY = 24
    T_ND = 1.0
    N_STEPS = 80

    def gravity(self):
        return self.BURGER * (self.F0 * self.L) ** 2 / self.H

    def dimensional(self):
        return NonlinearShallowWater2D.create(
            nx=self.NX,
            ny=self.NY,
            Lx=self.L,
            Ly=self.L,
            g=self.gravity(),
            f0=self.F0,
            H0=self.H,
            lateral_viscosity=self.EKMAN_LATERAL * self.F0 * self.L**2,
        )

    def nondimensional(self):
        return NonlinearShallowWater2D.from_nondimensional(
            nx=self.NX,
            ny=self.NY,
            rossby=self.ROSSBY,
            burger=self.BURGER,
            ekman_lateral=self.EKMAN_LATERAL,
        )

    def dimensional_state(self, model, scales):
        rng = np.random.RandomState(3)
        shape = (model.grid.Ny, model.grid.Nx)
        return NonlinearSW2DState(
            h=jnp.asarray(self.H + 0.02 * scales.eta * rng.randn(*shape)),
            u=jnp.asarray(0.02 * scales.U * rng.randn(*shape)),
            v=jnp.asarray(0.02 * scales.U * rng.randn(*shape)),
        )

    def to_model_units(self, state):
        """Dimensional state -> the nondimensional model's own units."""
        velocity_unit = self.L * self.F0
        return NonlinearSW2DState(
            h=state.h / self.H,
            u=state.u / velocity_unit,
            v=state.v / velocity_unit,
        )

    def to_dimensional(self, state):
        velocity_unit = self.L * self.F0
        return NonlinearSW2DState(
            h=state.h * self.H,
            u=state.u * velocity_unit,
            v=state.v * velocity_unit,
        )

    def scales(self):
        return Scales.inertial(
            L=self.L,
            f0=self.F0,
            H=self.H,
            rossby=self.ROSSBY,
            g=self.gravity(),
        )

    def test_model_units_match_the_scale_set(self):
        """The unit conversion above is the one ``Scales`` describes."""
        scales = self.scales()
        assert pytest.approx(self.F0 * self.L * self.ROSSBY) == scales.U
        assert pytest.approx(1.0 / self.F0) == scales.T

    def test_tendencies_agree(self):
        """The sharpest check: one RHS evaluation, no accumulation."""
        dim = self.dimensional()
        nd, _ = self.nondimensional()
        state_dim = self.dimensional_state(dim, self.scales())
        state_nd = self.to_model_units(state_dim)

        acceleration_unit = self.L * self.F0 * self.F0
        thickness_rate_unit = self.H * self.F0
        got = nd.vector_field(0.0, state_nd)
        expected = dim.vector_field(0.0, state_dim)
        for field, unit in (
            ("h", thickness_rate_unit),
            ("u", acceleration_unit),
            ("v", acceleration_unit),
        ):
            scaled = np.asarray(getattr(got, field)) * unit
            assert relative_error(scaled, getattr(expected, field)) < 1e-4, field

    def test_trajectories_agree_under_the_scales_mapping(self):
        dim = self.dimensional()
        nd, _ = self.nondimensional()
        scales = self.scales()
        state_dim = self.dimensional_state(dim, scales)
        state_nd = self.to_model_units(state_dim)

        sol_nd = nd.integrate(
            state_nd, t0=0.0, t1=self.T_ND, dt=self.T_ND / self.N_STEPS
        ).ys
        sol_dim = dim.integrate(
            state_dim,
            t0=0.0,
            t1=self.T_ND * scales.T,
            dt=self.T_ND * scales.T / self.N_STEPS,
        ).ys
        recovered = self.to_dimensional(sol_nd)

        # Thickness is compared against its anomaly about the mean.
        anomaly = np.asarray(sol_dim.h) - self.H
        assert (
            np.abs(np.asarray(recovered.h) - np.asarray(sol_dim.h)).max()
            / np.abs(anomaly).max()
            < 1e-3
        )
        for field in ("u", "v"):
            assert (
                relative_error(getattr(recovered, field), getattr(sol_dim, field))
                < 1e-3
            ), field

    def test_the_states_actually_evolve(self):
        dim = self.dimensional()
        scales = self.scales()
        state = self.dimensional_state(dim, scales)
        evolved = dim.integrate(
            state,
            t0=0.0,
            t1=self.T_ND * scales.T,
            dt=self.T_ND * scales.T / self.N_STEPS,
        ).ys
        change = np.abs(np.asarray(evolved.u) - np.asarray(state.u)).max()
        assert change > 0.1 * np.abs(np.asarray(state.u)).max()


class TestBaroclinicDimensionalEquivalence:
    """An advective-set nondimensional QG run reproduces its dimensional twin."""

    L = 1.0e6
    U = 0.05
    ROSSBY = 0.02
    BETA_HAT = 15.0
    BURGER = (1.0, 0.05)
    THICKNESS = (1.0, 4.0)
    DELTA_M = 0.08
    NX = NY = 48

    def dimensional(self):
        f0 = self.U / (self.ROSSBY * self.L)
        beta = self.BETA_HAT * self.U / self.L**2
        H = tuple(h * 1000.0 for h in self.THICKNESS)
        g_prime = tuple(
            bu * (f0 * self.L) ** 2 / h for bu, h in zip(self.BURGER, H, strict=True)
        )
        return BaroclinicQG.create(
            nx=self.NX,
            ny=self.NY,
            Lx=self.L,
            Ly=self.L,
            f0=f0,
            beta=beta,
            n_layers=2,
            H=H,
            g_prime=g_prime,
            lateral_viscosity=self.DELTA_M**3 * self.L**3 * beta,
            # tau_hat = tau0 L**2 / (U**2 H_1): the RHS divides by H[0].
            wind_amplitude=self.BETA_HAT * self.U**2 * H[0] / self.L**2,
        )

    def nondimensional(self):
        return BaroclinicQG.from_nondimensional(
            nx=self.NX,
            ny=self.NY,
            rossby=self.ROSSBY,
            beta_hat=self.BETA_HAT,
            burger=list(self.BURGER),
            thickness_ratio=list(self.THICKNESS),
            delta_M=self.DELTA_M,
        )

    def test_burger_numbers_match_between_the_two(self):
        dim = self.dimensional()
        nd, _ = self.nondimensional()
        for model in (dim, nd):
            f0 = float(model.consts.f0)
            g_prime = np.asarray(model.strat.g_prime)
            H = np.asarray(model.strat.H)
            L = model.grid.Lx
            np.testing.assert_allclose(
                g_prime * H / (f0 * L) ** 2, self.BURGER, rtol=1e-4
            )

    def test_modal_radii_match_after_rescaling(self):
        dim = self.dimensional()
        nd, _ = self.nondimensional()
        dim_radii = np.asarray(dim.modal.rossby_radii) / self.L
        nd_radii = np.asarray(nd.modal.rossby_radii)
        finite = np.isfinite(dim_radii)
        np.testing.assert_allclose(dim_radii[finite], nd_radii[finite], rtol=1e-3)

    def test_trajectories_agree_under_the_scales_mapping(self):
        dim = self.dimensional()
        nd, _ = self.nondimensional()
        scales = Scales.advective(L=self.L, U=self.U, f0=float(dim.consts.f0))
        transform = StateAffine.from_scales(BaroclinicQGState, scales)

        rng = np.random.RandomState(5)
        field = rng.randn(2, dim.grid.Ny, dim.grid.Nx)
        field -= field.mean(axis=(1, 2), keepdims=True)
        state_dim = BaroclinicQGState(q=jnp.asarray(0.02 * scales.vorticity * field))
        state_nd = transform.forward(state_dim)

        t_nd, n = 0.3, 30
        sol_nd = nd.integrate(state_nd, t0=0.0, t1=t_nd, dt=t_nd / n).ys
        sol_dim = dim.integrate(
            state_dim, t0=0.0, t1=t_nd * scales.T, dt=t_nd * scales.T / n
        ).ys
        assert relative_error(transform.inverse(sol_nd).q, sol_dim.q) < 5e-3

    def test_the_states_actually_evolve(self):
        dim = self.dimensional()
        scales = Scales.advective(L=self.L, U=self.U, f0=float(dim.consts.f0))
        rng = np.random.RandomState(5)
        field = rng.randn(2, dim.grid.Ny, dim.grid.Nx)
        field -= field.mean(axis=(1, 2), keepdims=True)
        state = BaroclinicQGState(q=jnp.asarray(0.02 * scales.vorticity * field))
        evolved = dim.integrate(
            state, t0=0.0, t1=0.3 * scales.T, dt=0.3 * scales.T / 30
        ).ys
        change = np.abs(np.asarray(evolved.q) - np.asarray(state.q)).max()
        assert change > 0.01 * np.abs(np.asarray(state.q)).max()


class TestMultilayerStateTransform:
    def test_per_layer_thickness_scales_broadcast(self):
        """The (nl, 1, 1) override path, on a real multilayer state."""
        model, scales = MultilayerShallowWater2D.from_nondimensional(
            nx=12,
            ny=12,
            rossby=0.05,
            burger=[1.0, 0.1],
            thickness_ratio=[1.0, 9.0],
        )
        g_prime = jnp.asarray(model.strat.g_prime)[:, None, None]
        H_k = jnp.asarray(model.strat.H)[:, None, None]
        dH = scales.f0 * scales.U * scales.L / g_prime
        transform = StateAffine.from_scales(
            MultilayerSW2DState, scales, h={"loc": H_k, "scale": dH}
        )
        rng = np.random.RandomState(6)
        shape = (2, model.grid.Ny, model.grid.Nx)
        state = MultilayerSW2DState(
            h=H_k + jnp.asarray(0.01 * rng.randn(*shape)),
            u=jnp.asarray(0.01 * rng.randn(*shape)),
            v=jnp.asarray(0.01 * rng.randn(*shape)),
        )
        assert transform.scale.h.shape == (2, 1, 1)
        recovered = transform.inverse(transform.forward(state))
        np.testing.assert_allclose(recovered.h, state.h, rtol=1e-5)
