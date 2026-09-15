"""Authored config: barotropic QG double-gyre, specified dimensionlessly.

Scenario: ``double_gyre`` with a ``nondim:`` block instead of ``consts:``.
Model: ``barotropic_qg`` (:class:`somax.models.BarotropicQG`).

The dimensional companion is ``doublegyre_bt_qg``. Where that config
sets ``f0``, ``beta``, ``lateral_viscosity`` and ``bottom_drag`` in SI
units, this one names the dimensionless numbers and lets the factory
derive them:

* ``rossby`` fixes ``f0``,
* ``beta_hat`` is ``beta`` directly at unit scales,
* ``munk`` and ``stommel`` are the western-boundary-layer widths as
  fractions of the basin, which is what actually has to be resolved.

``munk = 0.05`` on a 64-cell grid puts 3.2 cells across the viscous
boundary layer, comfortably above the 2-cell floor the ``munk_width``
guard enforces. Times are in units of the advective scale ``T = L/U``,
so ``t1 = 20`` is twenty basin-crossing times rather than a number of
seconds.
"""

DoubleGyreBTQGNondimConfig: dict = {
    "scenario": {
        "name": "double_gyre",
        "grid": {"nx": 64, "ny": 64, "Lx": 1.0, "Ly": 1.0},
        # Mutually exclusive with `consts`: the two would describe the
        # same physics in different units.
        "nondim": {
            "rossby": 0.02,
            "beta_hat": 50.0,
            "munk": 0.05,
            "stommel": 0.02,
        },
        "forcing": {"wind_profile": "doublegyre"},
        "initial_condition": {"type": "at_rest"},
    },
    "model": {
        "name": "barotropic_qg",
        "stratification": {},
        # Viscosity and drag come from `munk` / `stommel` above.
        "params": {},
    },
    "timestepping": {
        "t0": 0.0,
        "t1": 20.0,
        "dt": 1.0e-3,
        "save_interval": 1.0,
    },
    "output": {"format": "zarr", "save_final_state": True},
    # Not ``default_debug()``: its SI-oriented values are wrong twice
    # over here. A 32-cell grid resolves the Munk layer with only 1.6
    # cells, so the factory's own guard rejects the run before it
    # starts; and its ``t1``/``save_interval`` are in *seconds*, which
    # at ``dt=1e-3`` crossing times would be 86 million steps. The
    # debug run stays nondimensional: a grid that still spans the Munk
    # layer, and a couple of crossing times.
    "debug": {
        "scenario": {"grid": {"nx": 64, "ny": 64}},
        "timestepping": {"t1": 2.0, "save_interval": 0.5},
    },
    "assertions": {
        "munk_width": {"n_cells_min": 2.0},
        "stommel_width": {"n_cells_min": 1.0},
    },
}
