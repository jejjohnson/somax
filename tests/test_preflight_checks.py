"""Unit tests for the new preflight assertions (S5).

These build small real models (cheap, no integration) so they run in the
fast PR lane. The end-to-end runner wiring is covered by the integration
suite in ``tests/test_cli_assertions.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

from somax._src.cli._assertions import (
    PREFLIGHT_ASSERTIONS,
    AssertionFailedError,
    check_deformation_radius,
    check_pv_inversion,
    check_static_stability,
)
from somax._src.cli.spec import (
    DebugSpec,
    ModelSpec,
    OutputSpec,
    RunSpec,
    ScenarioSpec,
    TimesteppingSpec,
)


def _spec(scenario_name: str, model_name: str, **model_kw: Any) -> RunSpec:
    return RunSpec(
        scenario=ScenarioSpec(
            name=scenario_name,
            grid={"nx": 32, "ny": 32, "Lx": 1.0e6, "Ly": 1.0e6},
            consts={"f0": 1.0e-4, "beta": 1.6e-11},
            forcing={},
            initial_condition={"type": "at_rest"},
        ),
        model=ModelSpec(name=model_name, **model_kw),
        timestepping=TimesteppingSpec(t0=0.0, t1=600.0, dt=10.0, save_interval=600.0),
        output=OutputSpec(),
        debug=DebugSpec(),
    )


def _build(spec: RunSpec):
    from somax._src.cli._factories import build
    from somax._src.cli._run import _model_params, _scenario_params

    model, _ = build(
        spec.scenario.name,
        spec.model.name,
        scenario_params=_scenario_params(spec),
        model_params=_model_params(spec),
    )
    return model


def _multilayer_swm_model(H, g_prime, nx=32):
    spec = _spec(
        "double_gyre",
        "multilayer_nonlinear_swm",
        stratification={"H": list(H), "g_prime": list(g_prime)},
        params={"lateral_viscosity": 100.0},
    )
    spec.scenario.grid["nx"] = nx
    spec.scenario.grid["ny"] = nx
    return spec, _build(spec)


class TestRegistry:
    def test_new_checks_registered(self) -> None:
        for name in ("deformation_radius", "pv_inversion", "static_stability"):
            assert name in PREFLIGHT_ASSERTIONS


class TestDeformationRadius:
    def test_resolved_grid_passes(self) -> None:
        # H=(500,1500), g'=(9.81,0.02): L_d ~ sqrt(0.02*1500)/1e-4 ~ 54 km;
        # dx ~ 31 km -> ratio ~1.7 with default min 2 would FAIL, so use a
        # finer grid to clear the bar.
        spec, model = _multilayer_swm_model((500.0, 1500.0), (9.81, 0.05), nx=128)
        # L_d = sqrt(0.05*1500)/1e-4 ~ 86 km, dx ~ 7.8 km -> ratio ~11: passes.
        check_deformation_radius(spec, model)

    def test_underresolved_raises(self) -> None:
        # Coarse grid + small L_d -> ratio < 2 -> FAIL.
        spec, model = _multilayer_swm_model((500.0, 1500.0), (9.81, 0.001), nx=16)
        with pytest.raises(AssertionFailedError, match=r"under-resolved|L_d/dx"):
            check_deformation_radius(spec, model)

    def test_non_stratified_model_rejected(self) -> None:
        from types import SimpleNamespace

        bad = SimpleNamespace(grid=SimpleNamespace(dx=1.0, dy=1.0))
        with pytest.raises(AssertionFailedError):
            check_deformation_radius(_spec("double_gyre", "barotropic_qg"), bad)


class TestStaticStability:
    def test_stable_profile_passes(self) -> None:
        spec, model = _multilayer_swm_model((500.0, 1500.0), (9.81, 0.02))
        check_static_stability(spec, model)

    def test_unstable_profile_raises(self) -> None:
        spec, model = _multilayer_swm_model((500.0, 1500.0), (9.81, -0.02))
        with pytest.raises(AssertionFailedError, match="non-positive"):
            check_static_stability(spec, model)


class TestPVInversion:
    def test_barotropic_qg_round_trip_closes(self) -> None:
        spec = _spec(
            "double_gyre", "barotropic_qg", params={"lateral_viscosity": 100.0}
        )
        model = _build(spec)
        # at_rest -> zero PV -> trivially closes (early return).
        check_pv_inversion(spec, model, tol=1e-6)

    def test_non_qg_model_rejected(self) -> None:
        spec, model = _multilayer_swm_model((500.0, 1500.0), (9.81, 0.02))
        with pytest.raises(AssertionFailedError, match="no _invert_pv"):
            check_pv_inversion(spec, model)

    def test_baroclinic_qg_rejected(self) -> None:
        """Baroclinic QG (3-D modal-Helmholtz PV) is out of scope: the bare
        Laplacian round-trip omits the stretching term, so the check must
        refuse it rather than report a spurious O(1) residual."""
        spec = _spec(
            "double_gyre",
            "multilayer_qg",
            stratification={"H": [500.0, 1500.0], "g_prime": [9.81, 0.02]},
            params={},
        )
        model = _build(spec)
        with pytest.raises(AssertionFailedError, match=r"barotropic|3-D PV"):
            check_pv_inversion(spec, model)
