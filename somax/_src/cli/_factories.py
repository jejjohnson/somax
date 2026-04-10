"""Per-test-case adapters for somax-sim.

Each adapter takes the structured ``RunSpec.testcase`` blocks
(``grid`` / ``consts`` / ``stratification`` / ``params``) and dispatches
into the corresponding flat-kwargs factory in
:mod:`somax._src.models.gfd_testcases`.

Why an adapter layer? See Q-E in
``.plans/dvc-pipelines-design.md``: the structured shape mirrors how
somax separates ``Params`` (differentiable) from ``PhysConsts`` (frozen)
internally and gives researchers a cleaner mental model than flat
kwargs. The adapters keep ``gfd_testcases.py`` unchanged for in-Python
users while letting the CLI/config layer use the structured shape.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from somax._src.models import gfd_testcases


# ----------------------------------------------------------------------
# Adapter signature
# ----------------------------------------------------------------------
#
# Each adapter takes four kwargs (grid, consts, stratification, params),
# each a dict, and returns ``(model, state0)``. Stratification can be
# empty for single-layer models.

Adapter = Callable[..., tuple[Any, Any]]


# ----------------------------------------------------------------------
# Shallow water — single layer
# ----------------------------------------------------------------------


def barotropic_jet_instability(
    *,
    grid: dict[str, Any],
    consts: dict[str, Any],
    stratification: dict[str, Any],
    params: dict[str, Any],
) -> tuple[Any, Any]:
    """Barotropic jet instability test case."""
    return gfd_testcases.barotropic_jet_instability(
        nx=grid["nx"],
        ny=grid["ny"],
        Lx=grid["Lx"],
        Ly=grid["Ly"],
        f0=consts["f0"],
        beta=consts["beta"],
        H0=consts["H0"],
        jet_speed=params["jet_speed"],
        jet_width=params["jet_width"],
        perturbation=params["perturbation"],
        lateral_viscosity=params["lateral_viscosity"],
    )


# ----------------------------------------------------------------------
# QG — barotropic
# ----------------------------------------------------------------------


def doublegyre_qg(
    *,
    grid: dict[str, Any],
    consts: dict[str, Any],
    stratification: dict[str, Any],
    params: dict[str, Any],
) -> tuple[Any, Any]:
    """Single-layer wind-driven double-gyre QG."""
    return gfd_testcases.doublegyre_qg(
        nx=grid["nx"],
        ny=grid["ny"],
        Lx=grid["Lx"],
        Ly=grid["Ly"],
        f0=consts["f0"],
        beta=consts["beta"],
        lateral_viscosity=params["lateral_viscosity"],
        bottom_drag=params["bottom_drag"],
        wind_amplitude=params["wind_amplitude"],
    )


# ----------------------------------------------------------------------
# QG — multilayer baroclinic
# ----------------------------------------------------------------------


def doublegyre_baroclinic_qg(
    *,
    grid: dict[str, Any],
    consts: dict[str, Any],
    stratification: dict[str, Any],
    params: dict[str, Any],
) -> tuple[Any, Any]:
    """Multilayer wind-driven double-gyre baroclinic QG."""
    return gfd_testcases.doublegyre_baroclinic_qg(
        nx=grid["nx"],
        ny=grid["ny"],
        Lx=grid["Lx"],
        Ly=grid["Ly"],
        f0=consts["f0"],
        beta=consts["beta"],
        n_layers=consts["n_layers"],
        H=tuple(stratification["H"]),
        g_prime=tuple(stratification["g_prime"]),
        lateral_viscosity=params["lateral_viscosity"],
        bottom_drag=params["bottom_drag"],
        wind_amplitude=params["wind_amplitude"],
    )


# ----------------------------------------------------------------------
# Multilayer SWM — baroclinic instability
# ----------------------------------------------------------------------


def baroclinic_instability_swm(
    *,
    grid: dict[str, Any],
    consts: dict[str, Any],
    stratification: dict[str, Any],
    params: dict[str, Any],
) -> tuple[Any, Any]:
    """Two-layer baroclinic instability in the multilayer SWM."""
    return gfd_testcases.baroclinic_instability_swm(
        nx=grid["nx"],
        ny=grid["ny"],
        Lx=grid["Lx"],
        Ly=grid["Ly"],
        f0=consts["f0"],
        beta=consts["beta"],
        H=tuple(stratification["H"]),
        g_prime=tuple(stratification["g_prime"]),
        lateral_viscosity=params["lateral_viscosity"],
        bottom_drag=params["bottom_drag"],
        jet_speed=params["jet_speed"],
        jet_width=params["jet_width"],
        perturbation=params["perturbation"],
    )


# ----------------------------------------------------------------------
# Registry — single source of truth for the CLI dispatch
# ----------------------------------------------------------------------


TEST_CASES: dict[str, Adapter] = {
    "barotropic_jet_instability": barotropic_jet_instability,
    "doublegyre_qg": doublegyre_qg,
    "doublegyre_baroclinic_qg": doublegyre_baroclinic_qg,
    "baroclinic_instability_swm": baroclinic_instability_swm,
}


def list_test_cases() -> list[str]:
    """Return the list of registered test case names."""
    return sorted(TEST_CASES)


def get_adapter(name: str) -> Adapter:
    """Look up an adapter by name. Raises ``KeyError`` with a helpful message."""
    try:
        return TEST_CASES[name]
    except KeyError as exc:
        available = ", ".join(list_test_cases())
        raise KeyError(f"unknown test case {name!r}; available: {available}") from exc
