"""Authored config: 2-layer baroclinic instability in MultilayerSW2D.

Test case: ``baroclinic_instability_swm`` (see
:func:`somax._src.models.gfd_testcases.baroclinic_instability_swm`).

A small-basin (1000 km), 64-cell test that exercises the multilayer
nonlinear SWM with a baroclinically unstable jet. Cheap enough to run
in CI.
"""

from configs._authoring._common import (
    default_debug,
    default_timestepping,
    output_full,
)


SwmJetConfig: dict = {
    "testcase": {
        "name": "baroclinic_instability_swm",
        "grid": {"nx": 64, "ny": 64, "Lx": 1.0e6, "Ly": 1.0e6},
        "consts": {"f0": 1.0e-4, "beta": 1.6e-11},
        "stratification": {
            "H": [500.0, 4500.0],
            "g_prime": [9.81, 0.025],
        },
        "params": {
            # Lateral viscosity 1000 m²/s gives a viscous timescale of
            # dx²/nu = 15625² / 1000 ≈ 2.8 days, which is short enough
            # to damp grid-scale modes that would otherwise accumulate
            # energy and destabilize the integration after ~25 days.
            # See investigation in PR #71 — the original 100 m²/s was
            # not a CFL issue but a *slow numerical instability* due to
            # insufficient dissipation, only visible via the chunked
            # progress diagnostics.
            "lateral_viscosity": 1000.0,
            "bottom_drag": 1.0e-7,
            "jet_speed": 0.5,
            "jet_width": 5.0e4,
            "perturbation": 0.01,
        },
    },
    # 30 days at 64² is enough to see the jet meander develop.
    #
    # Two failure modes constrain dt:
    #
    # 1. Gravity-wave CFL: c = sqrt(g * H_total) = sqrt(9.81 * 5000) ≈ 221 m/s.
    #    dx = Lx / nx = 15625 m → max stable dt ≈ 0.5 * dx / c ≈ 35 s.
    #    The `cfl` assertion below enforces this preflight.
    #
    # 2. Slow numerical instability when lateral viscosity is too low for
    #    the resolution. Caught at runtime via the chunked-progress
    #    diagnostics, NOT by the CFL preflight (the integration is CFL-stable
    #    for the first ~23 days, then suddenly diverges). Mitigated by
    #    raising lateral_viscosity to 1000 above.
    "timestepping": default_timestepping(
        t1_seconds=30 * 86400.0,
        dt=20.0,
        save_interval_seconds=86400.0,
    ),
    "output": output_full(),
    "debug": default_debug(),
    "assertions": {
        "cfl": {
            "wave_speed_m_per_s": 221.0,  # sqrt(9.81 * 5000)
            "max_cfl": 0.5,
        },
    },
}
