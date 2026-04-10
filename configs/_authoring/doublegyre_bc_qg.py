"""Authored config: 3-layer baroclinic QG double-gyre, 1-year production run.

Test case: ``doublegyre_baroclinic_qg``.

This is the production-side companion to ``spinup_bc_qg``: it integrates
for 1 simulated year starting from the spinup's ``final_state.zarr``.
The DVC pipeline wires the dependency, and ``somax-sim restart`` is the
runtime mechanism.

Default parameters mirror MQGeometry's 3-layer config (Thiry et al. 2024
GMD), which is the closest published analog to this setup. See
``content/notes/qg_spinup_durations.md`` for the spinup-duration ladder.
"""

from configs._authoring._common import (
    YEAR_SECONDS,
    default_debug,
    default_timestepping,
    output_full,
)


DoubleGyreBCQGConfig: dict = {
    "testcase": {
        "name": "doublegyre_baroclinic_qg",
        "grid": {"nx": 128, "ny": 128, "Lx": 4.0e6, "Ly": 4.0e6},
        "consts": {"f0": 9.375e-5, "beta": 1.754e-11, "n_layers": 3},
        "stratification": {
            "H": [400.0, 1100.0, 2600.0],
            "g_prime": [9.81, 0.025, 0.0125],
        },
        "params": {
            "lateral_viscosity": 15.0,
            "bottom_drag": 1.0e-7,
            "wind_amplitude": 1.3e-10,
        },
    },
    # 1-year production window, 30-day snapshots.
    "timestepping": default_timestepping(
        t1_seconds=1 * YEAR_SECONDS,
        dt=600.0,
        save_interval_seconds=30 * 86400.0,
    ),
    "output": output_full(),
    "debug": default_debug(),
}
