# Shallow Water Models

The shallow water equations are the foundation of geophysical fluid dynamics.

## Governing Equations

The rotating shallow water equations on an f-plane:

$$
\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} - fv = -g \frac{\partial h}{\partial x}
$$

$$
\frac{\partial v}{\partial t} + u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} + fu = -g \frac{\partial h}{\partial y}
$$

$$
\frac{\partial h}{\partial t} + \frac{\partial (hu)}{\partial x} + \frac{\partial (hv)}{\partial y} = 0
$$

## Implementation in somax

somax provides both linear and nonlinear shallow water model implementations using the Arakawa C-grid discretization and WENO reconstruction for advection terms.

## Non-dimensional form

`NonlinearShallowWater2D.from_nondimensional` and
`MultilayerShallowWater2D.from_nondimensional` use the **inertial**
scale set: $L = f_0 = H = 1$, so $T = 1/f_0 = 1$ and the velocity scale
is $U = f_0 L\,Ro = Ro$.

| Input | Definition | Sets |
|---|---|---|
| `rossby` | $Ro = U/(f_0 L)$ | the velocity scale |
| `burger` | $Bu = gH/(f_0 L)^2$ | `g` (single layer), `g_prime` (multilayer) |
| `froude` | $Fr = U/\sqrt{gH}$ | `g`, through $Bu = (Ro/Fr)^2$ |
| `beta_hat` | $\beta L/f_0$ | `beta` |
| `ekman` | $\kappa/f_0$ | `bottom_drag` |
| `ekman_lateral` | $\nu/(f_0 L^2)$ | `lateral_viscosity` |
| `wind_hat` | $\tau_0/(f_0 U)$ | `wind_amplitude` |

```python
model, scales = NonlinearShallowWater2D.from_nondimensional(
    nx=128,
    ny=128,
    rossby=0.05,
    burger=1.0,
    ekman=1e-3,
)

layered, scales = MultilayerShallowWater2D.from_nondimensional(
    nx=128,
    ny=128,
    rossby=0.05,
    burger=[1.0, 0.1],
    thickness_ratio=[1.0, 9.0],
)
```

`burger` and `froude` are not independent once $Ro$ is fixed, so exactly
one may be given; supplying both would let you state an inconsistent
pair, and the factory rejects it.

Note that the model's own velocity fields are $O(Ro)$, not $O(1)$: the
inertial set puts $L$, $f_0$ and $H$ at unity, which leaves velocity at
$Ro$. Wrap the model in a `ScaledModel` built from the returned `Scales`
to work in $O(1)$ state.

## Comparing against a dimensional run

A nondimensional run reproduces its dimensional counterpart exactly in
arithmetic, and to a few times $10^{-5}$ in float32 — with one caveat
worth knowing. The Bernoulli term carries $g h$ including the mean
thickness, so at small Rossby number the mean swamps the anomaly and
float32 cancellation alone separates two algebraically identical runs by
a few times $10^{-4}$. Compare thickness against its anomaly rather than
its absolute value, and prefer a moderate Rossby number when checking
equivalence.
