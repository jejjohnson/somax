"""Authored config: high-resolution, eddy-resolving barotropic QG double-gyre.

Scenario: ``double_gyre`` (rectangular, Stommel wind, at-rest IC).
Model: ``barotropic_qg`` (:class:`somax.models.BarotropicQG`).

This is the eddy-resolving sibling of :mod:`doublegyre_bt_qg`. Same
1000 km basin and forcing, but refined from 64 to **256 cells**
(dx 15.6 km -> 3.9 km) and with the lateral viscosity dropped from
500 to 15 m^2/s so the Munk boundary layer stays resolved at the
finer grid:

    delta_M = (A / beta)^(1/3) = (15 / 1.6e-11)^(1/3) ~= 9.8 km ~= 2.5 dx

At A=500 the same basin is viscosity-dominated and laminar; at A=15
the western boundary current goes unstable and sheds mesoscale
eddies into the interior -- the regime the higher resolution is for.

Not a CI config: ~16x the cells and a halved time step make this a
minutes-to-hours run rather than a smoke test. The ``debug`` block
still drops to 32^2 / 1 day for a quick plumbing check.
"""

from configs._authoring._common import (
    YEAR_SECONDS,
    default_debug,
    default_timestepping,
    output_full,
)


DoubleGyreBTQGHiResConfig: dict = {
    "scenario": {
        "name": "double_gyre",
        "grid": {"nx": 256, "ny": 256, "Lx": 1.0e6, "Ly": 1.0e6},
        "consts": {"f0": 1.0e-4, "beta": 1.6e-11},
        "forcing": {"wind_amplitude": 1.0e-12, "wind_profile": "doublegyre"},
        "initial_condition": {"type": "at_rest"},
    },
    "model": {
        "name": "barotropic_qg",
        "stratification": {},  # single-layer barotropic — no stratification
        "params": {
            # Eddy-resolving: Munk layer ~2.5 cells at dx=3.9 km.
            "lateral_viscosity": 15.0,
            "bottom_drag": 1.0e-7,
        },
    },
    # 1 year at 256². dt halved to 300 s for headroom in the more
    # energetic eddying regime; 10-day snapshots give finer crash
    # recovery (with --checkpoint-every-n-chunks) and resolve the
    # eddy field's evolution rather than just monthly endpoints.
    "timestepping": default_timestepping(
        t1_seconds=1 * YEAR_SECONDS,
        dt=300.0,
        save_interval_seconds=10 * 86400.0,  # 10-day snapshots
    ),
    "output": output_full(),
    "debug": default_debug(),
    # Advection CFL: dx=3.9 km, dt=300 s. Even at a 3 m/s jet peak,
    # CFL = 3 * 300 / 3906 ≈ 0.23 — comfortably under 0.5.
    "assertions": {
        "cfl": {"wave_speed_m_per_s": 3.0, "max_cfl": 0.5},
        "bounded_metric": {
            "name": "kinetic_energy",
            "min": 0.0,
            "max": 1.0e15,
        },
    },
}
