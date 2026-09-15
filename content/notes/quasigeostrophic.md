# Quasi-Geostrophic Models

The quasi-geostrophic (QG) equations describe large-scale ocean and atmospheric flow where the Rossby number is small.

## Barotropic QG

The barotropic QG equation for the streamfunction $\psi$:

$$
\frac{\partial q}{\partial t} + J(\psi, q) = \text{forcing} + \text{dissipation}
$$

where $q = \nabla^2 \psi + \beta y$ is the potential vorticity and $J$ is the Jacobian operator.

## Multi-Layer QG

The multi-layer extension couples multiple fluid layers through the stretching term, representing baroclinic instability — the primary energy source for mesoscale ocean eddies.

## Implementation in somax

somax uses a DST-based (Discrete Sine Transform) Poisson solver for the elliptic inversion $\nabla^2 \psi = q$ and the Arakawa Jacobian for the advection term.

## Non-dimensional form

`BarotropicQG.from_nondimensional` builds the model from dimensionless
numbers instead of SI coefficients. It uses the **advective** scale set
($L = U = H = 1$, so $T = L/U = 1$ and $f_0 = 1/Ro$), and in those units
the equation reads

$$
\partial_t q + J(\psi,\, q + \hat\beta y)
  = \delta_M^3 \hat\beta\, \nabla^2 q
  - \delta_S \hat\beta\, \nabla^2 \psi
  + \hat\tau\, \mathrm{curl}\,\tau .
$$

| Input | Definition | Sets |
|---|---|---|
| `rossby` | $Ro = U/(f_0 L)$ | `f0 = 1/Ro` |
| `beta_hat` | $\hat\beta = \beta L^2/U$ | `beta` |
| `delta_M` | $(\nu/\beta)^{1/3}/L$ | `lateral_viscosity` |
| `delta_S` | $\kappa/(\beta L)$ | `bottom_drag` |
| `delta_I` | $(U/\beta)^{1/2}/L$ | `wind_amplitude` |

```python
model, scales = BarotropicQG.from_nondimensional(
    nx=128,
    ny=128,
    rossby=0.02,
    beta_hat=50.0,
    delta_M=0.03,
    delta_S=0.01,
)
```

The point of specifying $\delta_M$ and $\delta_S$ rather than $\nu$ and
$\kappa$ is that the western-boundary-layer widths are what has to be
resolved. Picking them directly replaces tuning two coefficients until a
run is both stable and resolved, and it lets the factory check the grid:
the `munk_width` and `stommel_width` preflight assertions fail when
$\delta_M/\Delta x < 2$ or $\delta_S/\Delta x < 1$, and warn just above
those thresholds. Both guards compare two lengths taken from the same
model, so they apply to dimensional runs too:

```yaml
assertions:
  munk_width: {n_cells_min: 2.0}
  stommel_width: {n_cells_min: 1.0}
```

`assertions` is a flat `{name: params}` mapping — there is no
`preflight:` level. Which phase a check runs in comes from the registry
it is in, not from the config, and `run_preflight` treats every
top-level key as an assertion name.

The wind amplitude follows the Sverdrup balance. Left unspecified it is
$\hat\tau = \hat\beta$, the amplitude whose Sverdrup interior velocity is
exactly the velocity scale $U$; passing `delta_I` instead sets
$\hat\tau = \delta_I^2 \hat\beta^2$.

The returned `Scales` is the unit scale set the model runs in — $L = U
= H = 1$ — so it is not by itself enough to convert a dimensional
state: with $L = U = 1$ the vorticity scale is 1 and a
`StateAffine.from_scales` built from it would leave a dimensional $q$
untouched. Keep the *physical* scale set alongside it, the
`Scales.advective(L=..., U=...)` describing the run being reproduced,
and build the transform from that one. Times passed to the
nondimensional model are in units of $T = L/U$.

### Layered QG

`BaroclinicQG.from_nondimensional` and
`ReparameterizedQG.from_nondimensional` use the same advective set and
add stratification, given as one Burger number **per interface**:

$$
Bu_k = \frac{g'_k H_k}{(f_0 L)^2}
\quad\Longrightarrow\quad
g'_k = \frac{Bu_k (f_0 L)^2}{H_k}.
$$

```python
model, scales = BaroclinicQG.from_nondimensional(
    nx=128,
    ny=128,
    rossby=0.02,
    beta_hat=20.0,
    burger=[1.0, 0.02],
    thickness_ratio=[1.0, 4.0],
    delta_M=0.06,
)
```

The interface convention is not the per-mode one. The deformation
radius of vertical mode $m$ comes from the eigenproblem and combines
the interface values: for two layers the rigid-lid result is
$L_d^2 = g' H_1 H_2 / (f_0^2 (H_1+H_2))$, and the free surface shifts it
a few percent below that. Prescribing modal radii directly would mean
inverting the eigenproblem, so the factory takes the interface numbers
and the built model reports the resulting radii as
`model.modal.rossby_radii` — already in units of $L$, so directly
comparable with $\Delta x$. That is exactly what the `deformation_radius`
guard reads, and `from_nondimensional` runs it alongside the Munk and
Stommel guards.

The wind default is again Sverdrup-balanced, $\hat\tau = \hat\beta$. The
right-hand side applies the wind as $\tau_0 F / H_1$, so the
dimensionless group is $\hat\tau = \tau_0 L^2 / (U^2 H_1)$ and the
factory multiplies by the top-layer thickness on the way in.
