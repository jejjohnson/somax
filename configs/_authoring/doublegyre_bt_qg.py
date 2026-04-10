"""Authored config: barotropic QG double-gyre.

Test case: ``doublegyre_qg`` (see
:func:`somax._src.models.gfd_testcases.doublegyre_qg`).

Small-basin (1000 km), 64-cell single-layer wind-driven double-gyre.
Exercises the barotropic QG vector field, PV inversion, and the
``"doublegyre"`` wind profile. Cheap enough for CI.
"""

from configs._authoring._common import (
    YEAR_SECONDS,
    default_debug,
    default_timestepping,
    output_full,
)


DoubleGyreBTQGConfig: dict = {
    "testcase": {
        "name": "doublegyre_qg",
        "grid": {"nx": 64, "ny": 64, "Lx": 1.0e6, "Ly": 1.0e6},
        "consts": {"f0": 1.0e-4, "beta": 1.6e-11},
        "stratification": {},  # single-layer barotropic — no stratification
        "params": {
            "lateral_viscosity": 500.0,
            "bottom_drag": 1.0e-7,
            "wind_amplitude": 1.0e-12,
        },
    },
    # 1 year at 64² — single-layer is fast.
    "timestepping": default_timestepping(
        t1_seconds=1 * YEAR_SECONDS,
        dt=600.0,
        save_interval_seconds=30 * 86400.0,  # monthly snapshots
    ),
    "output": output_full(),
    "debug": default_debug(),
}
