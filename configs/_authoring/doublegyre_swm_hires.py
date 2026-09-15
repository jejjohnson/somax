"""Authored config: high-resolution 3-layer shallow-water double-gyre.

Scenario: ``double_gyre`` (rectangular, Stommel wind, at-rest IC).
Model: ``multilayer_nonlinear_swm``
(:class:`somax.models.MultilayerShallowWater2D`).

The shallow-water baseline, and the primitive-equation sibling of the QG
double-gyres: same 4000 km basin, 256^2 grid, f0/beta and MQGeometry
stratification as :mod:`doublegyre_bc_qg_hires`. Where QG solves the balanced
(slow-manifold) dynamics, the SW model solves the full rotating shallow-water
equations — keeping gravity waves, ageostrophy and finite layer-thickness
variations that QG drops.

**It is multilayer, not single-layer — deliberately.** A *single*-layer SW
double-gyre cannot reproduce the separated, eddying jet: with one layer there
is no baroclinic instability, so the gyre stays laminar (the same limitation
that made the barotropic QG so hard). The reference implementation
(louity/qgsw-pytorch, the same group as MQGeometry) confirms this — its
"SW double-gyre" is 3-layer with exactly this H / g'. Baroclinic instability
between the layers is what sheds the eddies, as in the baroclinic QG run.

**Matches the reference recipe, with one CPU concession:**

  - 3 layers, H = [400, 1100, 2600] m (MQGeometry / Thiry et al. 2024).
  - WENO-Z (``wenoz5``) flux reconstruction, NOT upwind1: the high-order,
    low-dissipation scheme the reference uses so eddies are not smeared.
    Numerical dissipation comes from WENO alone — NO explicit lateral
    viscosity (lateral_viscosity = 0), exactly as in the reference.
  - Wind on the *top layer only* (``tau0 * tau_x / H[0]``); free-slip walls;
    linear bottom drag on the deepest layer.
  - ``wind_amplitude = 8e-5`` is the kinematic wind stress tau/rho in m^2/s^2
    (= 0.08 Pa / 1000), the reference value; its curl matches the baroclinic
    QG forcing (1.3e-10 ~ 8e-5 * 2*pi / Ly).

  CPU concession — the external (barotropic) gravity wave. With the true free
  surface g'[0] = 9.81, c = sqrt(g'[0] * H_total) ~= 200 m/s forces dt ~= 40 s,
  i.e. *days* per simulated year on CPU (the reference is GPU-oriented). We
  instead set a reduced external gravity g'[0] = 0.2 (a quasi-rigid-lid: it
  only slows the fast surface wave we do not care about — c drops to ~29 m/s,
  dt rises to 220 s — while the *baroclinic* modes that drive the eddies live
  in g'[1], g'[2] and are untouched). The barotropic adjustment (basin/c ~ 1.5
  days) stays far faster than the gyre/eddy evolution, so the slow dynamics are
  faithful. If the eddies look wrong, raise g'[0] toward 9.81 (and pay the dt).

Run from rest as a multi-year spinup with snapshots; the separated jet and
eddies develop over ~1-2 years (cf. the reference saving after year 2). Not a
CI config. The solver is adaptive and rejects steps on CFL violation.
"""

from configs._authoring._common import (
    YEAR_SECONDS,
    default_debug,
    default_timestepping,
    output_full,
)


DoubleGyreSWMHiResConfig: dict = {
    "scenario": {
        "name": "double_gyre",
        # Same basin, grid, planetary params as the QG hires runs.
        "grid": {"nx": 256, "ny": 256, "Lx": 4.0e6, "Ly": 4.0e6},
        "consts": {"f0": 9.375e-5, "beta": 1.754e-11},
        # Kinematic wind stress tau/rho [m^2/s^2] (= 0.08 Pa / 1000, reference
        # value); applied to the top layer as tau0 * tau_x / H[0].
        "forcing": {"wind_amplitude": 8.0e-5, "wind_profile": "doublegyre"},
        "initial_condition": {"type": "at_rest"},
    },
    "model": {
        "name": "multilayer_nonlinear_swm",
        "stratification": {
            "H": [400.0, 1100.0, 2600.0],
            # g'[0] is the *external* gravity, reduced from 9.81 to 0.2 (quasi-
            # rigid-lid) so the surface gravity wave is slow enough for a CPU dt.
            # g'[1], g'[2] are the real internal interfaces (baroclinic eddies).
            "g_prime": [0.2, 0.025, 0.0125],
        },
        "params": {
            # No explicit viscosity — WENO-Z provides the dissipation (reference).
            "lateral_viscosity": 0.0,
            "bottom_drag": 1.0e-7,
            "bc": "wall",  # closed basin: free-slip, no-normal-flow walls
            "method": "wenoz5",  # high-order, low-dissipation (cf. reference WENO-Z)
        },
    },
    # 5-year spinup from rest, 30-day snapshots — matches the QG runs so the
    # three are directly comparable; extend via restart from the checkpoint.
    "timestepping": default_timestepping(
        t1_seconds=5 * YEAR_SECONDS,
        # External gravity-wave CFL: c = sqrt(g'[0]*H_total) ~= 29 m/s at
        # g'[0]=0.2; dt=220 s gives CFL ~0.4.
        dt=220.0,
        save_interval_seconds=30 * 86400.0,
    ),
    "output": output_full(),
    "debug": default_debug(),
    "assertions": {
        "cfl": {"wave_speed_m_per_s": 33.0, "max_cfl": 0.5},
    },
}
