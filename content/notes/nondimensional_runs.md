# Non-dimensional runs

A somax model can be built from dimensionless numbers instead of SI
coefficients. The same `StateAffine` that expresses that change of
variables is also what the data-assimilation bridge and the exporter
use, so one object covers all three surfaces.

## From the CLI

Give a `nondim:` block in place of `consts:`:

```yaml
scenario:
  name: double_gyre
  grid: {nx: 64, ny: 64, Lx: 1.0, Ly: 1.0}
  nondim:
    rossby: 0.02
    beta_hat: 50.0
    munk: 0.05      # delta_M, the Munk-layer width as a fraction of L
    stommel: 0.02   # delta_S
  forcing: {wind_profile: doublegyre}
  initial_condition: {type: at_rest}

model:
  name: barotropic_qg
  params: {}        # viscosity and drag come from munk / stommel

timestepping:
  t0: 0.0
  t1: 20.0          # in units of T = L/U, not seconds
  dt: 1.0e-3
  save_interval: 1.0
```

The two blocks are mutually exclusive. They describe the same physics
in different units, so accepting both would let a config state a pair
that disagrees; `RunSpec.validate` rejects it and names both sides.
A model with no `from_nondimensional` entry in the registry is rejected
too, rather than quietly falling back to the dimensional path. So are
dimensional knobs that the dimensionless numbers derive —
`model.params.lateral_viscosity`, `scenario.forcing.wind_amplitude` and
the like — since the run would not use the values given for them.
`aspect` comes from `scenario.grid`, and a duplicate in the `nondim`
block is accepted only if it agrees.

`configs/_authoring/doublegyre_bt_qg_nondim.py` is a worked example.
`doublegyre_bt_qg` is a *separate* dimensional example, not the same
run in other units: its basin gives `beta L/f0 = 0.16` where the
nondimensional config asks for `beta_hat * rossby = 1`, and its
boundary-layer widths and integration window differ too. Treat them as
two examples rather than a validation pair — for that, build both from
the same dimensionless numbers.

Because a nondimensional run has no SI units to report,
`field_units("nondim")` marks every field `-` in the run log, so the
numbers do not read as unlabelled SI values.

## Data assimilation

`state_to_vector` and `make_ensemble` both take an optional
`transform`:

```python
transform = StateAffine.from_scales(NonlinearSW2DState, scales)
vector, unravel = state_to_vector(state, transform)
ensemble = make_ensemble(state, key, size=64, std=0.2, transform=transform)
```

This is the fix for a badly scaled background covariance. Flattened
without a transform, `h ~ 1e3 m` and `u ~ 1e-1 m/s` share one vector, so
a single `std` perturbs thickness and velocity by the same *absolute*
amount — four orders of magnitude apart in relative terms. Perturbing in
transformed space gives a per-field covariance of
`std**2 * diag(scale**2)` instead. The returned `unravel` maps back, so
the caller still sees ordinary states.

## Exported metadata

`apply_scale_metadata` writes what an offline tool needs to
re-dimensionalise:

```python
ds = snapshots_to_dataset(snapshots, times)
apply_scale_metadata(ds, scales=scales, transform=transform, nondimensional=True)
```

Dataset attrs carry the scale set flat under `somax:scales:` — netCDF
and Zarr attrs must be scalars or strings — including the time scale
`T`, which cannot be recomputed from `L` and `U` because the scale
families disagree about it. Per-variable `loc` and `scale` attrs are
written only for scalar leaves: a per-gridpoint `loc` is an array the
size of the field and belongs in the data, not the metadata.

Re-dimensionalisation happens offline, not in somax. Diagnostics are
reported in whatever units the model runs in, and
`ConservationDriftMonitor` reports relative drift, which is
scale-invariant, so a nondimensional run gets the same conservation
signal as a dimensional one.
