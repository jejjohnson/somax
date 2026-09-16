"""Authored config: high-resolution, eddy-resolving barotropic QG double-gyre.

Scenario: ``double_gyre`` (rectangular, Stommel wind, at-rest IC).
Model: ``barotropic_qg`` (:class:`somax.models.BarotropicQG`).

The barotropic baseline, and the single-layer sibling of
:mod:`doublegyre_bc_qg_hires`: same 4000 km basin, 256^2 grid, f0/beta and
lateral viscosity (A=15). It asks whether a purely barotropic model can
reproduce the separated eastward jet + recirculation gyres the 3-layer
baroclinic run produces. The only physics removed is stratification — so the
deformation radius and baroclinic instability are gone, and the PV is pure
relative vorticity (q = nabla^2 psi, no stretching term). It can, but only
after tuning the wind and drag into a narrow window; the two reasons follow.

1. **Wind units differ between the barotropic and baroclinic models.** The
   baroclinic model applies ``tau0 * F_wind / H[0]`` to the top layer, so its
   ``wind_amplitude`` is a kinematic stress curl in m/s^2 (1.3e-10, ~0.08 N/m^2
   — a realistic wind). The barotropic model applies ``tau0 * F_wind`` directly
   to the single layer's vorticity, so its ``wind_amplitude`` is a vorticity
   forcing in 1/s^2. The realistic, same-stress value would be
   ``1.3e-10 / H_total = 1.3e-10 / 4100 ~= 3.2e-14`` — but at that forcing the
   barotropic gyre is laminar (Sverdrup-weak, viscosity-dominated): a smooth,
   non-separating "two gyres". Baroclinic instability lets the 3-layer run eddy
   at a realistic wind; barotropic has no such mechanism, so it only separates
   once the *inertial* boundary layer beats the Munk layer, which needs a much
   stronger wind. We therefore drive it well past the realistic regime:

       wind_amplitude = 4.0e-12  (1/s^2)  ~= 125x the same-stress value

   This is the honest finding, not a fudge: a barotropic ocean needs an
   unrealistically strong wind (or much weaker friction) to separate, because
   it lacks the baroclinic instability that does the work in the real ocean.
   The value was tuned at runtime: a first attempt at 8.0e-12 ran away from
   rest in ~3 months to a single basin-scale **Fofonoff inertial recirculation**
   (|u| ~ 25 m/s, Ro > 1.3, KE still climbing) rather than a separated jet —
   the low-friction trap barotropic gyres are famous for. Halving the wind and
   tripling the bottom drag (below) arrests that basin mode while keeping the
   western boundary current inertial enough to separate, go unstable, and shed
   eddies; the run then equilibrates to a meandering jet at Ro ~ 0.55,
   KE ~ 3.5e13 (statistically steady, no drift).

2. **The Munk layer is under-resolved here, by construction.** With no
   deformation radius to set the interior eddy scale, the smallest dynamical
   scale is the Munk layer delta_M = (A/beta)^(1/3) = (15/1.754e-11)^(1/3)
   ~= 9.5 km = 0.6 dx at dx = 15.6 km. (A 1000 km basin at 256^2 would keep
   delta_M ~= 2.5 dx, but cannot match the baroclinic basin.) You cannot
   resolve delta_M *and* stay inertial at 4000 km / 256^2: resolving it needs
   A ~= 500 (laminar). So the effective western-boundary dissipation is partly
   numerical, and the forward enstrophy cascade may pile up near the grid. The
   enstrophy spectrum is watched at runtime (as for the baroclinic run); if the
   noise band climbs, the answer is 512^2, not config tuning.

Run from rest as a multi-year spinup with snapshots so the jet's development
can be tracked. Not a CI config (the ``debug`` block drops to 32^2 / 1 day for
a plumbing check). dt stays at 600 s; the solver is adaptive, so it rejects
steps if a fast inertial jet violates CFL rather than blowing up.
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
        # Same basin, grid, and planetary parameters as doublegyre_bc_qg_hires.
        "grid": {"nx": 256, "ny": 256, "Lx": 4.0e6, "Ly": 4.0e6},
        "consts": {"f0": 9.375e-5, "beta": 1.754e-11},
        # Vorticity forcing [1/s^2] (NOT m/s^2 — see module docstring), set into
        # the inertial regime the barotropic gyre needs to separate (8e-12 ran
        # away to a basin-scale inertial mode; 4e-12 + stronger drag is bounded).
        "forcing": {"wind_amplitude": 4.0e-12, "wind_profile": "doublegyre"},
        "initial_condition": {"type": "at_rest"},
    },
    "model": {
        "name": "barotropic_qg",
        "stratification": {},  # single-layer barotropic — no stratification
        "params": {
            # Same lateral viscosity as doublegyre_bc_qg_hires; bottom drag is 3x
            # stronger (3e-7 vs 1e-7) to arrest the basin-scale inertial mode the
            # single layer is prone to without baroclinic instability.
            "lateral_viscosity": 15.0,
            "bottom_drag": 3.0e-7,
        },
    },
    # 5-year spinup from rest, 30-day snapshots — matches the baroclinic run so
    # the two are directly comparable; extend via restart from the checkpoint.
    "timestepping": default_timestepping(
        t1_seconds=5 * YEAR_SECONDS,
        dt=600.0,
        save_interval_seconds=30 * 86400.0,
    ),
    "output": output_full(),
    "debug": default_debug(),
    # Generous blow-up tripwire (the under-resolved Munk layer + non-dissipative
    # Arakawa advection is the risk); CFL wave speed allows a vigorous jet.
    "assertions": {
        "cfl": {"wave_speed_m_per_s": 5.0, "max_cfl": 0.5},
        "bounded_metric": {
            "name": "kinetic_energy",
            "min": 0.0,
            "max": 2.0e15,
        },
    },
}
