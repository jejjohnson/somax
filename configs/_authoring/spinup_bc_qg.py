"""Authored config: 3-year spinup of the baroclinic QG double-gyre.

Test case: ``doublegyre_baroclinic_qg``.

Spinup runs save only the endpoint state (no snapshots, no metrics).
The resulting ``final_state.zarr`` is the initial condition for the
production ``doublegyre_bc_qg`` run.

Spinup duration is 3 years, the v0.1 default per the literature
analysis in ``content/notes/qg_spinup_durations.md``:
- it clears the basin baroclinic Rossby crossing time τ_R ≈ 3 yr
- it produces a fully formed western boundary current with first eddies
- it runs in single-digit minutes at 128² on a modern CPU

For production-quality (10 yr) or publication-grade (40 yr) runs,
override at the experiment level via
``dvc exp run -S configs/simulation/spinup_bc_qg.yaml:timestepping.t1=...``
or by writing a sibling config.
"""

from configs._authoring._common import (
    YEAR_SECONDS,
    default_debug,
    default_timestepping,
    output_spinup,
)


SPINUP_YEARS: float = 3.0


SpinupBCQGConfig: dict = {
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
    # save_interval == window so only the endpoint is saved.
    "timestepping": default_timestepping(
        t1_seconds=SPINUP_YEARS * YEAR_SECONDS,
        dt=600.0,
        save_interval_seconds=SPINUP_YEARS * YEAR_SECONDS,
    ),
    "output": output_spinup(),
    "debug": default_debug(
        debug_t1_seconds=86400.0,
        debug_save_interval_seconds=86400.0,
    ),
}
