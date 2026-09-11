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
too, rather than quietly falling back to the dimensional path.

`configs/_authoring/doublegyre_bt_qg_nondim.py` is a worked example, and
the dimensional companion `doublegyre_bt_qg` is the same run in SI.

Because a nondimensional run has no SI units to report,
`field_units("nondim")` marks every field `-` in the run log, so the
numbers do not read as unlabelled SI values.

## Spherical models

Spherical models use the **planetary** scale set: the planet radius is
the length scale and the rotation period the time scale, so
`a = Omega = H = 1`, `T = 1` and `f0 = 2 Omega = 2`.

That factor of two is the one thing worth remembering. On a sphere
`f(phi) = 2 Omega sin(phi)`, so `2 Omega` is what plays the role of a
Cartesian `f0`, and the Rossby number keeps its usual `U/(f0 L)`
meaning — the conventional spherical `U/(2 Omega a)`. Every rate
coefficient therefore picks up a `2`: `bottom_drag = 2 * ekman`,
`lateral_viscosity = 2 * ekman_lateral`.

Stratification can be given as `burger`, `froude` or `lamb`, and
exactly one of the three. `lamb` is the Lamb parameter
`eps = 4 Omega^2 a^2 / (gH)`, the inverse Burger number and the usual
spelling in the spherical literature.

```yaml
scenario:
  name: global_ocean
  grid: {nx: 128, ny: 64, lon_bounds: [0.0, 360.0], lat_bounds: [-80.0, 80.0]}
  nondim:
    rossby: 0.05
    lamb: 10.0      # or burger: 0.1 — the same thing inverted
    ekman: 0.01
  forcing: {wind_profile: zonal}
  initial_condition: {type: at_rest}

model:
  name: spherical_swm
```

`spherical_qg` takes no `lamb`/`burger`: barotropic QG is rigid-lid, so
it has no gravity wave. Neither takes a `beta_hat` — on a sphere the
planetary vorticity gradient is fixed by the geometry,
`beta = 2 Omega cos(phi)/a`, rather than being a free number.

The lat/lon bounds stay in degrees. They are a geometric choice, not a
scale: a nondimensional sphere is still a sphere.

### The equatorial deformation radius

The mid-latitude radius `sqrt(gH)/f0` diverges at the equator, where
`f` vanishes, so `deformation_radius` says nothing useful about a
global run. The finite scale that replaces it is the equatorial
deformation radius

```
L_eq = sqrt(c / beta_eq),  c = sqrt(gH),  beta_eq = 2 Omega / a
```

the trapping width of the equatorial waveguide — which in Burger terms
is simply `L_eq/a = Bu^(1/4)`. `SphericalSWM.from_nondimensional` runs
the `equatorial_deformation_radius` guard by default and raises if the
grid cannot span it; pass `check_resolution=False` to build a
deliberately coarse model. The guard compares against the *widest*
interior cell, since zonal cells are widest at the equator, which is
exactly where the waveguide sits.

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

## Learned priors (optional)

The `flows` extra bridges a `StateAffine` into flowjax, so a normalising
flow can be composed onto a fixed physical scaling instead of having to
relearn it:

```python
from somax.flows import learned_prior

prior = learned_prior(transform, state_example, flow, base_distribution)
```

The flow models the standardised state; the scaling maps its output back
to physical units, so the prior is over physical states while the flow
only ever sees O(1) numbers. `content/tutorials/learned_prior_flowjax.py`
works through it.

flowjax stays optional. Its own `Affine` forces the scale through a
trainable softplus parameterisation, which is the opposite of what a
fixed physical scale wants, and it brings the whole flow stack along.
The log-determinant only matters when transforming densities; every
other use of the affine map in somax needs no flow library.
