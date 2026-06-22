"""Authored config: high-resolution 3-layer baroclinic QG double-gyre spinup.

Scenario: ``double_gyre`` (rectangular, Stommel wind, at-rest IC).
Model: ``multilayer_qg`` (:class:`somax.models.BaroclinicQG`).

The eddy-permitting, higher-resolution sibling of :mod:`doublegyre_bc_qg`.
Same 4000 km basin, MQGeometry 3-layer stratification (Thiry et al. 2024
GMD) and forcing, but refined from 128 to **256 cells** (dx 31 -> 15.6 km)
so the first baroclinic deformation radius is actually sampled:

    Ld1 = 42.3 km  -> Ld1/dx = 2.7   (eddy-permitting)
    Ld2 = 24.9 km  -> Ld2/dx = 1.6   (2nd mode still marginal)

At 128 the deformation radius is sub-grid and baroclinic instability is
suppressed; at 256 the western boundary current goes baroclinically
unstable and sheds eddies, so the separated eastward jet and inertial
recirculation gyres — the structures a single-layer barotropic model
cannot sustain — emerge in the time mean.

Run from rest as a multi-year spinup with snapshots, so the jet's
development can be tracked. Not a CI config: ~1.5 h per simulated year
on CPU. dt stays at 600 s (advective CFL ~0.08 at this dx).
"""

from configs._authoring._common import (
    YEAR_SECONDS,
    default_debug,
    default_timestepping,
    output_full,
)


DoubleGyreBCQGHiResConfig: dict = {
    "scenario": {
        "name": "double_gyre",
        "grid": {"nx": 256, "ny": 256, "Lx": 4.0e6, "Ly": 4.0e6},
        "consts": {"f0": 9.375e-5, "beta": 1.754e-11},
        "forcing": {"wind_amplitude": 1.3e-10, "wind_profile": "doublegyre"},
        "initial_condition": {"type": "at_rest"},
    },
    "model": {
        "name": "multilayer_qg",
        "stratification": {
            "H": [400.0, 1100.0, 2600.0],
            "g_prime": [9.81, 0.025, 0.0125],
        },
        "params": {
            "lateral_viscosity": 15.0,
            "bottom_drag": 1.0e-7,
        },
    },
    # 5-year spinup from rest, 30-day snapshots — long enough to grow the
    # baroclinic jet and first eddies; extend via restart from the checkpoint.
    "timestepping": default_timestepping(
        t1_seconds=5 * YEAR_SECONDS,
        dt=600.0,
        save_interval_seconds=30 * 86400.0,
    ),
    "output": output_full(),
    "debug": default_debug(),
}
