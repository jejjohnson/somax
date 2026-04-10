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
            "lateral_viscosity": 100.0,
            "bottom_drag": 1.0e-7,
            "jet_speed": 0.5,
            "jet_width": 5.0e4,
            "perturbation": 0.01,
        },
    },
    # 30 days at 64² is enough to see the jet meander develop.
    "timestepping": default_timestepping(
        t1_seconds=30 * 86400.0,
        dt=300.0,
        save_interval_seconds=86400.0,
    ),
    "output": output_full(),
    "debug": default_debug(),
}
