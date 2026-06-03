# API Reference

```{note}
This page is generated from the public-API docstrings by
`scripts/gen_api_reference.py`. Regenerate it with
`uv run python scripts/gen_api_reference.py` when the public API
changes; a CI smoke test keeps the committed page in sync.
```

## Core Contract

The base types every model and component is built on.

### `Diagnostics`

*class*

```python
Diagnostics() -> None
```

Base class for on-demand diagnostic quantities.

````{admonition} Details
:class: dropdown

```text
Computed from a model state via ``model.diagnose(state)``.
```
````

### `Params`

*class*

```python
Params() -> None
```

Base class for differentiable model parameters.

````{admonition} Details
:class: dropdown

```text
Fields on Params subclasses are visible to ``jax.grad`` by default.
Use ``eqx.field(static=True)`` for non-differentiable parameters.
```
````

### `PhysConsts`

*class*

```python
PhysConsts() -> None
```

Base class for frozen physical constants.

````{admonition} Details
:class: dropdown

```text
All fields should be marked ``static=True`` so they are invisible
to ``jax.grad`` and treated as compile-time constants.
```
````

### `SomaxModel`

*class*

```python
SomaxModel() -> None
```

Abstract base class defining the somax model contract.

````{admonition} Details
:class: dropdown

```text
All somax models follow this interface for interoperability with
diffrax, ``jax.grad``, and downstream tools like fourdvarjax.

Subclasses must implement:
    - ``vector_field``: the right-hand side of the ODE/PDE
    - ``apply_boundary_conditions``: boundary enforcement
```
````

### `State`

*class*

```python
State() -> None
```

Base class for model state vectors.

````{admonition} Details
:class: dropdown

```text
All model states should subclass this to enable interoperability
with the somax model contract and JAX transformations.
```
````

## Models

Dynamical-system and ocean model classes, with their state, parameter, and diagnostic companions.

### `BaroclinicQG`

*class*

```python
BaroclinicQG(params: 'BaroclinicQGParams', consts: 'BaroclinicQGPhysConsts', grid: 'CartesianGrid2D', diff: 'Difference2D', interp: 'Interpolation2D', mask: 'Mask2D | None', modal: 'ModalTransform', strat: 'StratificationProfile', beta_y: "Float[Array, 'Ny Nx']", wind_forcing: "Float[Array, 'Ny Nx']", helmholtz_lambdas: 'Array', poisson_bc: 'str' = 'dst') -> None
```

Multilayer quasi-geostrophic model on an Arakawa C-grid.

````{admonition} Details
:class: dropdown

```text
Solves the multilayer QG PV equation per layer k::

    dq_k/dt = -J(psi_k, q_k + beta*y)
              + tau0 * F_wind / H[0]  (top layer only)
              - kappa * zeta_{nl-1}   (bottom layer only)
              + nu * laplacian(q_k)

PV inversion uses vertical mode decomposition::

    q_modal = Cl2m @ q_layer
    (nabla^2 - f0^2 * lambda_m) psi_modal_m = q_modal_m
    psi_layer = Cm2l @ psi_modal

following the MQGeometry approach (louity/MQGeometry).

Args:
    params: Differentiable parameters.
    consts: Frozen physical constants.
    grid: 2D Arakawa C-grid.
    diff: Difference operators.
    interp: Interpolation operators.
    mask: Optional Arakawa C-grid mask (``None`` = all-ocean).
    modal: Precomputed modal transform.
    strat: Stratification profile.
    beta_y: Precomputed beta*(y - y0) field.
    wind_forcing: Normalised wind stress curl pattern.
    helmholtz_lambdas: f0^2 * eigenvalues per mode, shape ``(nl,)``.
    poisson_bc: Spectral solver BC type for PV inversion.
```
````

### `BarotropicQG`

*class*

```python
BarotropicQG(params: 'BarotropicQGParams', consts: 'BarotropicQGPhysConsts', grid: 'CartesianGrid2D', diff: 'Difference2D', interp: 'Interpolation2D', mask: 'Mask2D | None', beta_y: "Float[Array, 'Ny Nx']", wind_forcing: "Float[Array, 'Ny Nx']", poisson_bc: 'str' = 'dst') -> None
```

Barotropic quasi-geostrophic model on an Arakawa C-grid.

````{admonition} Details
:class: dropdown

```text
Solves the barotropic QG PV equation::

    dq/dt = -J(ψ, q) + tau0*F_wind - kappa*laplacian(psi) + nu*laplacian(q)

where:
    - q is the PV anomaly (relative vorticity, nabla^2 psi)
    - total PV is q + beta*y
    - psi is the streamfunction from inversion: nabla^2 psi = q
    - J(psi, q + beta*y) is the Arakawa Jacobian (energy+enstrophy conserving)
    - u = -dpsi/dy, v = dpsi/dx (geostrophic velocity)

Args:
    params: Differentiable parameters.
    consts: Frozen physical constants.
    grid: 2D Arakawa C-grid.
    diff: Difference operators.
    interp: Interpolation operators.
    mask: Optional Arakawa C-grid mask (``None`` = all-ocean).
    beta_y: Precomputed β·y field.
    wind_forcing: Precomputed wind stress curl pattern (normalised).
    poisson_bc: Spectral solver BC type for PV inversion.
```
````

### `Burgers1D`

*class*

```python
Burgers1D(params: 'Burgers1DParams', grid: 'CartesianGrid1D', diff: 'Difference1D', advection: 'Advection1D', mask: 'Mask1D | None', periodic: 'bool' = True, method: 'str' = 'upwind1') -> None
```

1D Burgers equation on an Arakawa C-grid.

````{admonition} Details
:class: dropdown

```text
Solves ``du/dt + u * du/dx = nu * d²u/dx²``, combining nonlinear
advection with viscous diffusion. The viscosity ``nu`` is learnable
and visible to ``jax.grad``.

Args:
    params: Differentiable parameters (viscosity ``nu``).
    grid: 1D Arakawa C-grid.
    diff: Difference operators (for diffusion).
    advection: Advection operator (for nonlinear convection).
    mask: Optional 1-D Arakawa C-grid mask (``None`` = all-ocean).
    periodic: Whether to use periodic boundary conditions.
    method: Reconstruction method for advection (default ``"upwind1"``).
```
````

### `Burgers2D`

*class*

```python
Burgers2D(params: 'Burgers2DParams', grid: 'CartesianGrid2D', diff: 'Difference2D', advection: 'FVXAdvection2D', interp: 'Interpolation2D', mask: 'Mask2D | None', method: 'str' = 'upwind1') -> None
```

2D Burgers equation on an Arakawa C-grid.

````{admonition} Details
:class: dropdown

```text
Solves the system::

    du/dt + u * du/dx + v * du/dy = nu * laplacian(u)
    dv/dt + u * dv/dx + v * dv/dy = nu * laplacian(v)

Args:
    params: Differentiable parameters (viscosity ``nu``).
    grid: 2D Arakawa C-grid.
    diff: Difference operators.
    advection: Advection operator.
    interp: Interpolation operators.
    mask: Optional Arakawa C-grid mask (``None`` = all-ocean).
    method: Reconstruction method for advection (default ``"upwind1"``).
```
````

### `Diffusion1D`

*class*

```python
Diffusion1D(params: 'Diffusion1DParams', grid: 'CartesianGrid1D', diff: 'Difference1D', mask: 'Mask1D | None', periodic: 'bool' = True) -> None
```

1D diffusion equation on an Arakawa C-grid.

````{admonition} Details
:class: dropdown

```text
Solves ``du/dt = nu * d²u/dx²`` where ``nu`` is a learnable
viscosity visible to ``jax.grad``.

Args:
    params: Differentiable parameters (viscosity ``nu``).
    grid: 1D Arakawa C-grid.
    diff: Difference operators.
    mask: Optional 1-D Arakawa C-grid mask (``None`` = all-ocean).
    periodic: Whether to use periodic boundary conditions.
```
````

### `Diffusion2D`

*class*

```python
Diffusion2D(params: 'Diffusion2DParams', grid: 'CartesianGrid2D', diff: 'Difference2D', mask: 'Mask2D | None') -> None
```

2D diffusion equation on an Arakawa C-grid.

````{admonition} Details
:class: dropdown

```text
Solves ``du/dt = nu * (d²u/dx² + d²u/dy²)``.

Args:
    params: Differentiable parameters (viscosity ``nu``).
    grid: 2D Arakawa C-grid.
    diff: Difference operators.
    mask: Optional Arakawa C-grid mask (``None`` = all-ocean).
```
````

### `HelmholtzSolver2D`

*class*

```python
HelmholtzSolver2D(grid: 'CartesianGrid2D', bc_type: 'str' = 'dirichlet', lambda_: 'float' = 1.0) -> None
```

Solve the 2D Helmholtz equation: :math:`(\nabla^2 - \lambda) \phi = f`.

````{admonition} Details
:class: dropdown

```text
Wraps finitevolX spectral solvers with a non-zero Helmholtz
parameter. This arises in quasi-geostrophic PV inversion
:math:`(\nabla^2 - F)\psi = q`, Yukawa screening, and
reaction-diffusion steady states.

Args:
    grid: 2D Arakawa C-grid.
    bc_type: Boundary condition type.
    lambda_: Helmholtz parameter (screening coefficient).
```
````

### `IncompressibleNS2D`

*class*

```python
IncompressibleNS2D(params: 'NSParams', grid: 'CartesianGrid2D', diff: 'Difference2D', interp: 'Interpolation2D', advection: 'FVXAdvection2D', mask: 'Mask2D | None', problem: 'str' = 'cavity', poisson_bc: 'str' = 'dst', u_lid: 'float' = 1.0, body_force: 'float' = 0.0, method: 'str' = 'upwind1') -> None
```

2D incompressible Navier-Stokes (vorticity-streamfunction).

````{admonition} Details
:class: dropdown

```text
Solves the vorticity transport equation::

    d(omega)/dt + u * d(omega)/dx + v * d(omega)/dy = nu * laplacian(omega)

where the velocity is recovered at each step via the Poisson
inversion :math:`\nabla^2 \psi = -\omega` and
:math:`u = \partial\psi/\partial y`,
:math:`v = -\partial\psi/\partial x`.

Supports two canonical benchmarks:

- **Lid-driven cavity**: Dirichlet BCs (``poisson_bc="dst"``),
  no-slip walls with moving lid at the top.
- **Channel flow**: Periodic in x, no-slip walls in y
  (``poisson_bc="dst"``), driven by a body force.

Args:
    params: Differentiable parameters (viscosity ``nu``).
    grid: 2D Arakawa C-grid.
    diff: Difference operators.
    interp: Interpolation operators.
    advection: Advection operator.
    mask: Optional Arakawa C-grid mask (``None`` = all-ocean).
    problem: Problem type (``"cavity"`` or ``"channel"``).
    poisson_bc: Spectral solver BC type for Poisson inversion.
    u_lid: Lid velocity for cavity flow (default 1.0).
    body_force: Constant vorticity source for channel flow.
    method: Advection reconstruction method.
```
````

### `LaplaceSolver2D`

*class*

```python
LaplaceSolver2D(grid: 'CartesianGrid2D', bc_type: 'str' = 'dirichlet') -> None
```

Solve the 2D Laplace equation: :math:`\nabla^2 \phi = 0`.

````{admonition} Details
:class: dropdown

```text
A thin wrapper around :class:`PoissonSolver2D` with zero RHS.
The solution is determined entirely by boundary conditions.

Args:
    grid: 2D Arakawa C-grid.
    bc_type: Boundary condition type.
```
````

### `LinearConvection1D`

*class*

```python
LinearConvection1D(params: 'LinearConvection1DParams', grid: 'CartesianGrid1D', diff: 'Difference1D', interp: 'Interpolation1D', mask: 'Mask1D | None', periodic: 'bool' = True) -> None
```

1D linear convection equation on an Arakawa C-grid.

````{admonition} Details
:class: dropdown

```text
Solves ``du/dt + c * du/dx = 0`` where ``c`` is a learnable wave
speed visible to ``jax.grad``.

Args:
    params: Differentiable parameters (wave speed ``c``).
    grid: 1D Arakawa C-grid.
    diff: Difference operators.
    interp: Interpolation operators.
    mask: Optional 1-D Arakawa C-grid mask (``None`` = all-ocean).
    periodic: Whether to use periodic boundary conditions.
```
````

### `LinearConvection2D`

*class*

```python
LinearConvection2D(params: 'LinearConvection2DParams', grid: 'CartesianGrid2D', diff: 'Difference2D', interp: 'Interpolation2D', mask: 'Mask2D | None') -> None
```

2D linear convection on an Arakawa C-grid.

````{admonition} Details
:class: dropdown

```text
Solves ``du/dt + cx * du/dx + cy * du/dy = 0``.

Args:
    params: Differentiable parameters (wave speeds ``cx``, ``cy``).
    grid: 2D Arakawa C-grid.
    diff: Difference operators.
    interp: Interpolation operators.
    mask: Optional Arakawa C-grid mask (``None`` = all-ocean).
```
````

### `LinearShallowWater1D`

*class*

```python
LinearShallowWater1D(params: 'LinearSW1DParams', consts: 'LinearSW1DPhysConsts', grid: 'CartesianGrid1D', diff: 'Difference1D', interp: 'Interpolation1D', mask: 'Mask1D | None') -> None
```

1D linear shallow water model on an Arakawa C-grid.

````{admonition} Details
:class: dropdown

```text
Solves the linearised shallow water equations::

    dh/dt = -H₀ · du/dx
    du/dt = -g · dh/dx + f₀·v + nu*laplacian(u) - kappa*u
    dv/dt = -f₀·u           + nu*laplacian(v) - kappa*v

where h is the height perturbation, (u, v) are velocities,
and the Coriolis term couples u ↔ v even in 1D.

Args:
    params: Differentiable parameters (viscosity, drag).
    consts: Frozen physical constants (g, f₀, H₀).
    grid: 1D Arakawa C-grid.
    diff: Difference operators.
    interp: Interpolation operators.
    mask: Optional 1-D Arakawa C-grid mask (``None`` = all-ocean).
```
````

### `LinearShallowWater2D`

*class*

```python
LinearShallowWater2D(params: 'LinearSW2DParams', consts: 'LinearSW2DPhysConsts', grid: 'CartesianGrid2D', diff: 'Difference2D', interp: 'Interpolation2D', coriolis: 'Coriolis2D', mask: 'Mask2D | None', f_field: "Float[Array, 'Ny Nx']", bc_type: 'str' = 'periodic') -> None
```

2D linear shallow water model on an Arakawa C-grid.

````{admonition} Details
:class: dropdown

```text
Solves the linearised shallow water equations::

    dh/dt = -H₀ · (du/dx + dv/dy)
    du/dt = -g · dh/dx + f·v + nu*laplacian(u) - kappa*u
    dv/dt = -g · dh/dy - f·u + nu*laplacian(v) - kappa*v

Supports both f-plane (β=0) and β-plane Coriolis.

Args:
    params: Differentiable parameters (viscosity, drag).
    consts: Frozen physical constants (g, f₀, β, H₀).
    grid: 2D Arakawa C-grid.
    diff: Difference operators.
    interp: Interpolation operators.
    coriolis: Coriolis operator.
    mask: Optional Arakawa C-grid mask (``None`` = all-ocean).
    f_field: Precomputed Coriolis parameter field f(y).
    bc_type: Boundary condition type (``"periodic"`` or ``"wall"``).
```
````

### `Lorenz63`

*class*

```python
Lorenz63(params: 'L63Params') -> None
```

Lorenz '63 three-variable chaotic system.

````{admonition} Details
:class: dropdown

```text
The canonical low-dimensional chaotic attractor::

    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - beta * z

Args:
    params: Differentiable parameters (sigma, rho, beta).
```
````

### `Lorenz96`

*class*

```python
Lorenz96(params: 'L96Params', advection: 'bool' = True) -> None
```

Lorenz '96 periodic 1D chaotic system.

````{admonition} Details
:class: dropdown

```text
N coupled ODEs with periodic boundary conditions::

    dX_k/dt = (X_{k+1} - X_{k-2}) * X_{k-1} - X_k + F

Args:
    params: Differentiable parameters (F).
    advection: Whether to include the nonlinear advection term.
```
````

### `Lorenz96t`

*class*

```python
Lorenz96t(params: 'L96TParams', advection: 'bool' = True) -> None
```

Lorenz '96 two-tier (slow-fast) coupled system.

````{admonition} Details
:class: dropdown

```text
Slow variables X couple to fast variables Y::

    dX_k/dt = (X_{k+1} - X_{k-2}) * X_{k-1} - X_k + F - (hc/b) * sum_j(Y_{j,k})
    dY_{j,k}/dt = cb * (Y_{j+1} - Y_{j-2}) * Y_{j-1} - cY + (hc/b) * X_k

Args:
    params: Differentiable parameters (F, h, b, c).
    advection: Whether to include nonlinear advection terms.
```
````

### `MultilayerShallowWater2D`

*class*

```python
MultilayerShallowWater2D(params: 'MultilayerSW2DParams', consts: 'MultilayerSW2DPhysConsts', grid: 'CartesianGrid2D', diff: 'Difference2D', interp: 'Interpolation2D', coriolis: 'Coriolis2D', vorticity: 'Vorticity2D', advection: 'FVXAdvection2D', diffusion: 'FVXDiffusion2D', mask: 'Mask2D | None', strat: 'StratificationProfile', modal: 'ModalTransform', f_field: "Float[Array, 'Ny Nx']", f_field_ml: "Float[Array, 'nl Ny Nx']", wind_stress_x: "Float[Array, 'Ny Nx']", wind_stress_y: "Float[Array, 'Ny Nx']", bc_type: 'str' = 'periodic', method: 'str' = 'upwind1') -> None
```

Multilayer 2D nonlinear shallow water model (vector-invariant form).

````{admonition} Details
:class: dropdown

```text
Solves the rotating shallow water equations per layer k::

    dh_k/dt = -div(h_k * u_k)
    du_k/dt = +q_k * (h_k v_k)_bar - dP_k/dx + forcing
    dv_k/dt = -q_k * (h_k u_k)_bar - dP_k/dy + forcing

where q_k = (zeta_k + f) / h_k is potential vorticity and
P_k = KE_k + p_k is the Bernoulli potential with hydrostatic
pressure coupling between layers:
p_k = sum_{j=0}^{k} g'_j * h_j (cumulative).

Wind forcing is applied to the top layer only; bottom drag
to the bottom layer only.

Args:
    params: Differentiable parameters.
    consts: Frozen physical constants.
    grid: 2D Arakawa C-grid.
    diff: Difference operators.
    interp: Interpolation operators.
    coriolis: Coriolis operator.
    vorticity: Vorticity/PV operator.
    advection: Scalar advection operator (for mass).
    diffusion: Diffusion operator.
    mask: Optional Arakawa C-grid mask (``None`` = all-ocean).
    strat: Stratification profile (layer depths and reduced gravities).
    modal: Precomputed modal transform.
    f_field: Precomputed Coriolis field f(y) at T-points.
    f_field_ml: Coriolis field broadcast to ``(nl, Ny, Nx)``.
    wind_stress_x: Precomputed x-wind stress pattern (normalised).
    wind_stress_y: Precomputed y-wind stress pattern (normalised).
    bc_type: Boundary condition type.
    method: Advection reconstruction method for mass equation.
```
````

### `NonlinearConvection1D`

*class*

```python
NonlinearConvection1D(grid: 'CartesianGrid1D', advection: 'Advection1D', mask: 'Mask1D | None', periodic: 'bool' = True, method: 'str' = 'upwind1') -> None
```

1D nonlinear convection (inviscid Burgers) on an Arakawa C-grid.

````{admonition} Details
:class: dropdown

```text
Solves ``du/dt + u * du/dx = 0`` using upwind flux reconstruction.

Args:
    grid: 1D Arakawa C-grid.
    advection: Advection operator.
    mask: Optional 1-D Arakawa C-grid mask (``None`` = all-ocean).
    periodic: Whether to use periodic boundary conditions.
    method: Reconstruction method for advection (default ``"upwind1"``).
```
````

### `NonlinearConvection2D`

*class*

```python
NonlinearConvection2D(grid: 'CartesianGrid2D', advection: 'FVXAdvection2D', interp: 'Interpolation2D', mask: 'Mask2D | None', method: 'str' = 'upwind1') -> None
```

2D nonlinear convection on an Arakawa C-grid.

````{admonition} Details
:class: dropdown

```text
Solves the system::

    du/dt + u * du/dx + v * du/dy = 0
    dv/dt + u * dv/dx + v * dv/dy = 0

using upwind flux reconstruction.

Args:
    grid: 2D Arakawa C-grid.
    advection: Advection operator.
    interp: Interpolation operators.
    mask: Optional Arakawa C-grid mask (``None`` = all-ocean).
    method: Reconstruction method (default ``"upwind1"``).
```
````

### `NonlinearShallowWater1D`

*class*

```python
NonlinearShallowWater1D(params: 'NonlinearSW1DParams', consts: 'NonlinearSW1DPhysConsts', grid: 'CartesianGrid1D', diff: 'Difference1D', interp: 'Interpolation1D', advection: 'Advection1D', mask: 'Mask1D | None', method: 'str' = 'upwind1') -> None
```

1D nonlinear shallow water model on an Arakawa C-grid.

````{admonition} Details
:class: dropdown

```text
Solves the nonlinear shallow water equations::

    dh/dt = -d(h·u)/dx
    du/dt = -u·du/dx - g·dh/dx + f₀·v + nu*laplacian(u) - kappa*u
    dv/dt = -f₀·u                      + nu*laplacian(v) - kappa*v

Args:
    params: Differentiable parameters (viscosity, drag).
    consts: Frozen physical constants (g, f₀, H₀).
    grid: 1D Arakawa C-grid.
    diff: Difference operators.
    interp: Interpolation operators.
    advection: Advection operator.
    mask: Optional 1-D Arakawa C-grid mask (``None`` = all-ocean).
    method: Advection reconstruction method.
```
````

### `NonlinearShallowWater2D`

*class*

```python
NonlinearShallowWater2D(params: 'NonlinearSW2DParams', consts: 'NonlinearSW2DPhysConsts', grid: 'CartesianGrid2D', diff: 'Difference2D', interp: 'Interpolation2D', coriolis: 'Coriolis2D', vorticity: 'Vorticity2D', advection: 'FVXAdvection2D', diffusion: 'FVXDiffusion2D', mask: 'Mask2D | None', f_field: "Float[Array, 'Ny Nx']", wind_stress_x: "Float[Array, 'Ny Nx']", wind_stress_y: "Float[Array, 'Ny Nx']", bc_type: 'str' = 'periodic', method: 'str' = 'upwind1') -> None
```

2D nonlinear shallow water model (vector-invariant form).

````{admonition} Details
:class: dropdown

```text
Solves the rotating shallow water equations in vector-invariant
form on an Arakawa C-grid::

    dh/dt = -div(h*u)
    du/dt = +q·h̄v - ∂P/∂x + nu*laplacian(u) - kappa*u
    dv/dt = -q·h̄u - ∂P/∂y + nu*laplacian(v) - kappa*v

where q = (ζ+f)/h is potential vorticity and P = KE + g·h
is the Bernoulli potential.

Args:
    params: Differentiable parameters.
    consts: Frozen physical constants.
    grid: 2D Arakawa C-grid.
    diff: Difference operators.
    interp: Interpolation operators.
    coriolis: Coriolis operator.
    vorticity: Vorticity/PV operator.
    advection: Scalar advection operator (for mass).
    diffusion: Diffusion operator.
    mask: Optional Arakawa C-grid mask (``None`` = all-ocean).
    f_field: Precomputed Coriolis field f(y).
    wind_stress_x: Precomputed x-wind stress pattern (normalised).
    wind_stress_y: Precomputed y-wind stress pattern (normalised).
    bc_type: Boundary condition type.
    method: Advection reconstruction method for mass equation.
```
````

### `PoissonSolver2D`

*class*

```python
PoissonSolver2D(grid: 'CartesianGrid2D', bc_type: 'str' = 'dirichlet') -> None
```

Solve the 2D Poisson equation: :math:`\nabla^2 \phi = f`.

````{admonition} Details
:class: dropdown

```text
Wraps finitevolX spectral solvers (DST for Dirichlet, DCT for
Neumann, FFT for periodic). The solver operates on interior cells
and returns a full-grid array including ghost cells.

Args:
    grid: 2D Arakawa C-grid.
    bc_type: Boundary condition type (``"dirichlet"``, ``"neumann"``,
        or ``"periodic"``).
```
````

### `ReparameterizedQG`

*class*

```python
ReparameterizedQG(swm: 'MultilayerShallowWater2D', helmholtz_lambdas: 'Array', poisson_bc: 'str' = 'dst') -> None
```

Reparameterized QG model: multilayer SWM + geostrophic projection.

````{admonition} Details
:class: dropdown

```text
Wraps a ``MultilayerShallowWater2D`` and adds a geostrophic
projection P = G . (Q.G)^{-1} . Q applied via
``apply_boundary_conditions``, keeping the state on the
geostrophic manifold at each time step.

The three operators are:

- **Q** (PV extraction): q = curl(u,v) - f0 * eta / H
- **(Q.G)^{-1}** (Helmholtz solve): modal decomposition + DST
- **G** (geostrophic reconstruction): p -> (u_g, v_g, h_g)

The projection is idempotent (P.P = P), so applying it before each
RHS evaluation is equivalent to projecting after each time step.

Args:
    swm: The underlying multilayer shallow water model.
    helmholtz_lambdas: f0^2 * eigenvalues per mode.
    poisson_bc: Spectral solver BC type for Helmholtz.
```
````

### `geostrophic_adjustment_2d`

*function*

```python
geostrophic_adjustment_2d(nx: 'int' = 128, ny: 'int' = 128, Lx: 'float' = 1000000.0, Ly: 'float' = 1000000.0, f0: 'float' = 0.0001, H0: 'float' = 100.0, eta_max: 'float' = 1.0) -> 'tuple[LinearShallowWater2D, LinearSW2DState]'
```

2D geostrophic adjustment: step-function height perturbation.

````{admonition} Details
:class: dropdown

```text
A north-south height step adjusts to geostrophic balance,
radiating gravity waves.

Args:
    nx: Interior cells in x.
    ny: Interior cells in y.
    Lx: Domain length in x (m).
    Ly: Domain length in y (m).
    f0: Coriolis parameter (1/s).
    H0: Mean layer depth (m).
    eta_max: Height perturbation amplitude (m).

Returns:
    ``(model, state0)`` tuple.
```
````

### `gravity_wave_1d`

*function*

```python
gravity_wave_1d(nx: 'int' = 400, Lx: 'float' = 1000000.0, g: 'float' = 9.81, H0: 'float' = 100.0, sigma: 'float' = 50000.0) -> 'tuple[LinearShallowWater1D, LinearSW1DState]'
```

1D gravity wave: Gaussian height perturbation, no rotation.

````{admonition} Details
:class: dropdown

```text
Phase speed c = sqrt(g*H0) ~ 31.3 m/s for default parameters.

Args:
    nx: Number of interior grid cells.
    Lx: Domain length (m).
    g: Gravitational acceleration (m/s²).
    H0: Mean layer depth (m).
    sigma: Gaussian width (m).

Returns:
    ``(model, state0)`` tuple.
```
````

### `inertial_oscillation_1d`

*function*

```python
inertial_oscillation_1d(nx: 'int' = 50, Lx: 'float' = 1000000.0, f0: 'float' = 0.0001, u_init: 'float' = 1.0) -> 'tuple[LinearShallowWater1D, LinearSW1DState]'
```

1D inertial oscillation: uniform initial u, period = 2*pi/f0.

````{admonition} Details
:class: dropdown

```text
Args:
    nx: Number of interior grid cells.
    Lx: Domain length (m).
    f0: Coriolis parameter (1/s).
    u_init: Initial x-velocity (m/s).

Returns:
    ``(model, state0)`` tuple.
```
````

### State / parameter / diagnostic companions

Each model carries dataclass companions for its state, differentiable parameters, frozen physical constants, and on-demand diagnostics:

- `BaroclinicQGDiagnostics` — Diagnostics for the multilayer QG model.
- `BaroclinicQGParams` — Differentiable parameters for the multilayer QG model.
- `BaroclinicQGPhysConsts` — Frozen physical constants for the multilayer QG model.
- `BaroclinicQGState` — State for the multilayer quasi-geostrophic model.
- `BarotropicQGDiagnostics` — Diagnostics for the barotropic QG model.
- `BarotropicQGParams` — Differentiable parameters for the barotropic QG model.
- `BarotropicQGPhysConsts` — Frozen physical constants for the barotropic QG model.
- `BarotropicQGState` — State for the barotropic quasi-geostrophic model.
- `Burgers1DDiagnostics` — Diagnostics for 1D Burgers equation.
- `Burgers1DParams` — Differentiable parameters for 1D Burgers equation.
- `Burgers1DState` — State for 1D Burgers equation.
- `Burgers2DDiagnostics` — Diagnostics for 2D Burgers equation.
- `Burgers2DParams` — Differentiable parameters for 2D Burgers equation.
- `Burgers2DState` — State for 2D Burgers equation.
- `Diffusion1DDiagnostics` — Diagnostics for 1D diffusion.
- `Diffusion1DParams` — Differentiable parameters for 1D diffusion.
- `Diffusion1DState` — State for 1D diffusion.
- `Diffusion2DDiagnostics` — Diagnostics for 2D diffusion.
- `Diffusion2DParams` — Differentiable parameters for 2D diffusion.
- `Diffusion2DState` — State for 2D diffusion.
- `L63Diagnostics` — On-demand diagnostics for the Lorenz '63 system.
- `L63Params` — Differentiable parameters for the Lorenz '63 system.
- `L63State` — State vector for the Lorenz '63 system.
- `L96Diagnostics` — On-demand diagnostics for the Lorenz '96 system.
- `L96Params` — Differentiable parameters for the Lorenz '96 system.
- `L96State` — State vector for the Lorenz '96 system.
- `L96TParams` — Differentiable parameters for the two-tier Lorenz '96 system.
- `L96TState` — State vector for the two-tier Lorenz '96 system.
- `LinearConvection1DDiagnostics` — Diagnostics for 1D linear convection.
- `LinearConvection1DParams` — Differentiable parameters for 1D linear convection.
- `LinearConvection1DState` — State for 1D linear convection.
- `LinearConvection2DDiagnostics` — Diagnostics for 2D linear convection.
- `LinearConvection2DParams` — Differentiable parameters for 2D linear convection.
- `LinearConvection2DState` — State for 2D linear convection.
- `LinearSW1DDiagnostics` — Diagnostics for the 1D linear shallow water model.
- `LinearSW1DParams` — Differentiable parameters for the 1D linear shallow water model.
- `LinearSW1DPhysConsts` — Frozen physical constants for the 1D linear shallow water model.
- `LinearSW1DState` — State for the 1D linear shallow water model.
- `LinearSW2DDiagnostics` — Diagnostics for the 2D linear shallow water model.
- `LinearSW2DParams` — Differentiable parameters for the 2D linear shallow water model.
- `LinearSW2DPhysConsts` — Frozen physical constants for the 2D linear shallow water model.
- `LinearSW2DState` — State for the 2D linear shallow water model.
- `MultilayerSW2DDiagnostics` — Diagnostics for the multilayer 2D shallow water model.
- `MultilayerSW2DParams` — Differentiable parameters for the multilayer 2D shallow water model.
- `MultilayerSW2DPhysConsts` — Frozen physical constants for the multilayer 2D shallow water model.
- `MultilayerSW2DState` — State for the multilayer 2D nonlinear shallow water model.
- `NSDiagnostics` — Diagnostics for incompressible Navier-Stokes.
- `NSParams` — Differentiable parameters for incompressible Navier-Stokes.
- `NSVorticityState` — State for the vorticity-streamfunction NS formulation.
- `NonlinearConvection1DDiagnostics` — Diagnostics for 1D nonlinear convection.
- `NonlinearConvection1DState` — State for 1D nonlinear convection.
- `NonlinearConvection2DDiagnostics` — Diagnostics for 2D nonlinear convection.
- `NonlinearConvection2DState` — State for 2D nonlinear convection.
- `NonlinearSW1DDiagnostics` — Diagnostics for the 1D nonlinear shallow water model.
- `NonlinearSW1DParams` — Differentiable parameters for the 1D nonlinear shallow water model.
- `NonlinearSW1DPhysConsts` — Frozen physical constants for the 1D nonlinear shallow water model.
- `NonlinearSW1DState` — State for the 1D nonlinear shallow water model.
- `NonlinearSW2DDiagnostics` — Diagnostics for the 2D nonlinear shallow water model.
- `NonlinearSW2DParams` — Differentiable parameters for the 2D nonlinear shallow water model.
- `NonlinearSW2DPhysConsts` — Frozen physical constants for the 2D nonlinear shallow water model.
- `NonlinearSW2DState` — State for the 2D nonlinear shallow water model.
- `ReparamQGDiagnostics` — Diagnostics for the reparameterized QG model.

## pipekit Operators

Bridge that exposes somax models as ``pipekit.Operator`` stages.

### `Burgers2DOp`

*class*

```python
Burgers2DOp(nx: 'int' = 64, ny: 'int' = 64, Lx: 'float' = 2.0, Ly: 'float' = 2.0, nu: 'float' = 0.01, method: 'str' = 'upwind1', imex: 'bool' = False, dt: 'float' = 0.001) -> 'None'
```

Serializable pipekit Operator for the term-based 2D Burgers model.

````{admonition} Details
:class: dropdown

```text
Wraps :class:`~somax._src.models.pde2d.burgers_terms.Burgers2DTermModel`.
All constructor arguments are JSON primitives, so::

    op = Burgers2DOp(nx=32, ny=32, nu=0.05, dt=1e-3)
    assert pipekit.loads(pipekit.dumps(op)).get_config() == op.get_config()

round-trips the build recipe. ``op(state)`` advances the Burgers
state by one ``dt`` step; the Operator drives ``pipekit_cycle.Cycle``
and composes with the rest of pipekit.

Args:
    nx: Interior cells in x.
    ny: Interior cells in y.
    Lx: Domain length in x.
    Ly: Domain length in y.
    nu: Kinematic viscosity (diffusion coefficient).
    method: Advection reconstruction method.
    imex: Tag diffusion implicit for IMEX integration (see
        :meth:`Burgers2DTermModel.create`).
    dt: Default step size for :meth:`_apply`.
```
````

### `SomaxModelOp`

*class*

```python
SomaxModelOp(model: 'Any', dt: 'float') -> 'None'
```

A pipekit Operator wrapping *any* built somax forward model.

````{admonition} Details
:class: dropdown

```text
Construct directly from a built model (``SomaxModelOp(model, dt)``)
or from a scenario x model pair via :meth:`from_registry`. The
Operator is a one-step stage (``op(state) -> next_state`` advances by
``dt``) so models compose with the rest of pipekit (``op | op``,
graphs) and drive ``pipekit_cycle.Cycle``; it also satisfies the
``pipekit_cycle.ForwardModel`` protocol (``step`` / ``dt`` /
``state_signature``).

Serialization: the wrapped model is an ``eqx.Module`` (grids,
finitevolx operators, term trees — not JSON primitives), so this
general form is **not** faithfully round-trippable through
``pipekit.serial`` — it sets ``forbid_in_yaml = True`` and an empty
auto-config. Subclasses whose construction is a *flat primitive
recipe* (e.g. :class:`Burgers2DOp`) re-enable the round-trip by
rebuilding the model from those primitives.

Args:
    model: A built somax model exposing ``step(state, dt)``.
    dt: Default step size used by :meth:`_apply`.
```
````

## Evaluation Metrics

Reference-free field diagnostics computed on a model's own grid.

### `compute_eval_metrics`

*function*

```python
compute_eval_metrics(model: 'Any', state: 'State') -> 'dict[str, float]'
```

Compute every applicable reference-free metric for ``(model, state)``.

````{admonition} Details
:class: dropdown

```text
A defensive dispatcher meant to be called unconditionally by the runner:
it inspects the model / state and returns only the metrics that make
sense. Non-fluid models (Lorenz, diffusion, …) and multilayer (3D) states
yield an empty dict rather than an error.

Scope: this targets **velocity-state Arakawa C-grid models** — those whose
state carries 2D ``u`` / ``v`` (SWM, Burgers). **Vorticity / streamfunction
models** (``barotropic_qg``, the vorticity Navier-Stokes) are intentionally
*not* covered: they evolve ``q`` / ``omega`` and never define a discrete
velocity divergence (non-divergence is only an analytic property, so a
divergence metric would be operator-dependent with no canonical zero —
misleading rather than diagnostic). Those models already report
``kinetic_energy`` and ``enstrophy`` through their own ``diagnose`` output,
so they are not metric-less.

Args:
    model: A constructed somax model.
    state: The state to evaluate (typically the final integrated state).

Returns:
    Flat ``{metric_name: float}`` dict. For velocity-state models a subset
    of ``rms_divergence`` / ``total_enstrophy`` / ``kinetic_energy`` /
    ``geostrophic_imbalance``; for QG (vorticity/streamfunction) models a
    ``qg_balance_residual``. Plus any conserved quantities the model's
    ``diagnose(state).invariants()`` advertises, prefixed ``invariant_``.
    Empty when no metric applies.
```
````

### `geostrophic_imbalance`

*function*

```python
geostrophic_imbalance(model: 'Any', state: 'State', *, interior: 'bool' = True, eps: 'float' = 1e-30) -> "Float[Array, '']"
```

Dimensionless ageostrophic fraction of a shallow-water-type state.

````{admonition} Details
:class: dropdown

```text
Geostrophic balance is ``f x u = -g ∇η`` — equivalently, the
pressure-gradient and Coriolis accelerations cancel. This metric forms
that residual using the model's *own* operators::

    r_u = -g ∂η/∂x + f·v        (at U-points)
    r_v = -g ∂η/∂y - f·u        (at V-points)

and returns ``rms(r) / rms(coriolis)`` — 0 for a perfectly geostrophic
flow, O(1) when ageostrophic accelerations rival the Coriolis term.
Reusing ``model.diff`` and ``model.coriolis`` keeps the residual exactly
consistent with the balance the model integrates around (staggering,
masks and the β-plane ``f`` field all included).

Args:
    model: A shallow-water-type model exposing ``diff`` (with
        ``diff_x_T_to_U`` / ``diff_y_T_to_V``), ``coriolis``,
        ``f_field`` and ``consts.gravity``.
    state: A state with ``h`` (height perturbation η), ``u`` and ``v``.
    interior: Drop the one-cell ghost halo before reducing (default).
    eps: Floor added to the denominator to keep a motionless state
        (zero Coriolis term) finite.

Returns:
    Scalar ageostrophic fraction in ``[0, ∞)``.
```
````

### `kinetic_energy`

*function*

```python
kinetic_energy(u: "Float[Array, 'Ny Nx']", v: "Float[Array, 'Ny Nx']", grid: 'Any', *, interior: 'bool' = True) -> "Float[Array, '']"
```

Domain-integrated kinetic energy ``0.5 ∫ (u² + v²) dA``.

````{admonition} Details
:class: dropdown

```text
Mirrors the velocity term of the models' ``diagnose`` energy (summing the
staggered components directly), so it tracks consistently alongside the
model's own energy diagnostic.

Args:
    u: x-velocity field, shape ``(Ny, Nx)``.
    v: y-velocity field, shape ``(Ny, Nx)``.
    grid: The model's ``CartesianGrid2D`` (for the ``dx·dy`` cell area).
    interior: Drop the one-cell ghost halo before reducing (default).

Returns:
    Scalar kinetic energy (per unit depth).
```
````

### `qg_balance_residual`

*function*

```python
qg_balance_residual(model: 'Any', state: 'State', *, interior: 'bool' = True, eps: 'float' = 1e-30) -> "Float[Array, '']"
```

Dimensionless PV-inversion residual for a **barotropic** QG model.

````{admonition} Details
:class: dropdown

```text
The QG balance analog of :func:`geostrophic_imbalance` for vorticity /
streamfunction models: instead of a velocity-divergence residual (which QG
models do not define), it measures how well the state's PV closes its own
inversion. For barotropic QG the PV *is* the relative vorticity,
``q = nabla^2 psi``, so with ``psi = L^{-1} q`` and
``q_hat = laplacian(psi)`` it returns

.. math::

    \frac{\lVert \hat q - q \rVert}{\lVert q \rVert + \epsilon},

reduced over the grid interior. For a freshly inverted state this is
~machine-eps; a large value flags a state inconsistent with the model's
elliptic operator.

Restricted to barotropic QG (a 2-D PV field). Baroclinic / reparameterized
QG invert a *modal Helmholtz* operator ``q = nabla^2 psi - f0^2 A psi``;
the bare Laplacian omits the stretching term, so this residual would be
O(1) even for a perfectly balanced state. Reconstructing the full
stretching operator here would duplicate model internals, so the check is
intentionally scoped to the barotropic case (use :meth:`model.diagnose`
invariants for the layered models). Returns ``0.0`` for a trivially zero
PV field.

Args:
    model: The constructed barotropic QG model.
    state: A state carrying a 2-D PV field ``q``.
    interior: Drop the one-cell ghost halo before reducing. Defaults True.
    eps: Small constant guarding the normalisation.

Returns:
    Scalar dimensionless residual.
```
````

### `rms_divergence`

*function*

```python
rms_divergence(u: "Float[Array, 'Ny Nx']", v: "Float[Array, 'Ny Nx']", diff: 'Any', *, interior: 'bool' = True) -> "Float[Array, '']"
```

Root-mean-square horizontal divergence ``sqrt(<(∇·u)²>)``.

````{admonition} Details
:class: dropdown

```text
A diagnostic of how non-divergent the flow is. For incompressible /
geostrophic flow it should stay near zero; a growing value flags
spurious compressibility or numerical noise.

Args:
    u: x-velocity field on its C-grid points, shape ``(Ny, Nx)``.
    v: y-velocity field on its C-grid points, shape ``(Ny, Nx)``.
    diff: A finitevolx ``Difference2D`` (the model's ``.diff``); its
        :meth:`divergence` lowers ``(u, v)`` to ``∇·u`` at tracer points.
    interior: Drop the one-cell ghost halo before reducing (default).

Returns:
    Scalar RMS divergence (same units as ``∇·u``, i.e. 1/s for m/s
    velocities on a metre grid).
```
````

### `total_enstrophy`

*function*

```python
total_enstrophy(u: "Float[Array, 'Ny Nx']", v: "Float[Array, 'Ny Nx']", diff: 'Any', grid: 'Any', *, interior: 'bool' = True) -> "Float[Array, '']"
```

Domain-integrated enstrophy ``0.5 ∫ ζ² dA`` with ``ζ = ∂v/∂x - ∂u/∂y``.

````{admonition} Details
:class: dropdown

```text
Enstrophy is a robust health metric for 2D / quasi-2D turbulence: in the
inviscid limit it is bounded, so runaway growth signals instability.

Args:
    u: x-velocity field, shape ``(Ny, Nx)``.
    v: y-velocity field, shape ``(Ny, Nx)``.
    diff: finitevolx ``Difference2D``; its :meth:`curl` returns ζ.
    grid: The model's ``CartesianGrid2D`` (for the ``dx·dy`` cell area).
    interior: Drop the one-cell ghost halo before reducing (default).

Returns:
    Scalar total enstrophy.
```
````

## In-JIT Guards

Fail-fast tripwires that halt a run at the offending step.

### `guard_ceiling`

*function*

```python
guard_ceiling(x: 'Array', *, where: 'str', ceil: 'float') -> 'Array'
```

Return ``x`` unchanged, raising in-JIT if ``|x|`` exceeds ``ceil``.

````{admonition} Details
:class: dropdown

```text
A magnitude tripwire for velocity-state models — an opt-in blow-up
ceiling that halts an obviously-diverging run early.

Args:
    x: Array to check (returned unchanged).
    where: Human-readable location for the error message.
    ceil: Maximum allowed absolute value.

Returns:
    ``x`` unchanged. Raises via :func:`equinox.error_if` when any
    ``|element| > ceil``.
```
````

### `guard_finite`

*function*

```python
guard_finite(x: 'Array', *, where: 'str') -> 'Array'
```

Return ``x`` unchanged, raising in-JIT if it holds any NaN/Inf.

````{admonition} Details
:class: dropdown

```text
Args:
    x: Array to check (returned unchanged).
    where: Human-readable location for the error message (e.g.
        ``"RHS"`` or ``"layer thickness h_k"``).

Returns:
    ``x`` unchanged. Raises at trace/run time via
    :func:`equinox.error_if` when any element is non-finite.
```
````

### `guard_positive`

*function*

```python
guard_positive(x: 'Array', *, where: 'str', floor: 'float' = 0.0) -> 'Array'
```

Return ``x`` unchanged, raising in-JIT if any element ``<= floor``.

````{admonition} Details
:class: dropdown

```text
The canonical use is layer-thickness positivity in multilayer SWM: the
PV ``q_k = (zeta_k + f) / h_k`` is singular as ``h_k -> 0+``, so a
non-positive thickness makes the remaining computation meaningless
(FAIL-HARD).

Args:
    x: Array to check (returned unchanged).
    where: Human-readable location for the error message.
    floor: Exclusive lower bound; elements must be strictly greater.
        Defaults to ``0.0``.

Returns:
    ``x`` unchanged. Raises via :func:`equinox.error_if` when any
    element is ``<= floor``.
```
````

## Monitors

Chunk-boundary observability for the ``somax-sim`` runner.

### `BaseMonitor`

*class*

```python
BaseMonitor()
```

Inert base monitor — override only the hooks you care about.

````{admonition} Details
:class: dropdown

```text
Subclasses typically set a class-level ``name`` and override
:meth:`on_chunk_end`. The default hooks do nothing (and
:meth:`on_chunk_end` returns an empty :class:`MonitorVerdict`), so a
one-hook monitor stays minimal.
```
````

### `ChunkInfo`

*class*

```python
ChunkInfo(index: 'int', n_chunks: 'int', t0: 'float', t1: 'float', wall_seconds: 'float', is_snapshot: 'bool', stats: 'dict[str, Any]' = <factory>) -> None
```

Context handed to a monitor at a diagnostic-chunk boundary.

````{admonition} Details
:class: dropdown

```text
Args:
    index: Diagnostic-chunk index just completed (0-based).
    n_chunks: Total number of diagnostic chunks in the run.
    t0: Simulation time at the chunk start (seconds).
    t1: Simulation time at the chunk end (seconds).
    wall_seconds: Wallclock seconds spent integrating this chunk.
    is_snapshot: Whether the chunk endpoint is a snapshot save boundary.
    stats: Diffrax solver stats for this chunk (e.g.
        ``num_accepted_steps`` / ``num_rejected_steps`` / ``result``), or
        an empty dict if the solver did not expose them. Consumed by
        :class:`somax.monitor.SolverHealthMonitor`.
```
````

### `ConservationDriftMonitor`

*class*

```python
ConservationDriftMonitor(rtol_warn: 'float' = 0.01, rtol_fail: 'float | None' = None)
```

MONITOR (optionally FAIL-HARD): track drift of conserved invariants.

````{admonition} Details
:class: dropdown

```text
Records the relative drift ``|I(t) - I(0)| / |I(0)|`` for every invariant
the model advertises via :meth:`somax.Diagnostics.invariants`. Warns above
``rtol_warn``; if ``rtol_fail`` is set, terminates when the worst drift
exceeds it.

Mass is conserved to machine precision (set a tight tolerance); energy /
enstrophy / Casimirs drift under implicit numerical dissipation, so use a
generous ``rtol_warn`` and read the metric as "quantify the dissipation",
not "drive to zero".

Args:
    rtol_warn: Relative-drift warning threshold. Defaults to 1e-2.
    rtol_fail: If not ``None``, terminate when the worst drift exceeds it.
```
````

### `EnergyGrowthMonitor`

*class*

```python
EnergyGrowthMonitor(factor: 'float' = 10.0, hard_factor: 'float | None' = None)
```

MONITOR (optionally FAIL-HARD): flag large energy jumps between chunks.

````{admonition} Details
:class: dropdown

```text
Warns when the run's energy-like scalar grows by more than ``factor``
relative to the previous chunk — an early instability signal before a NaN
appears. If ``hard_factor`` is set, growth beyond it requests termination.

Args:
    factor: Warn when ``|E(t)| > factor * |E(t-1)|``. Defaults to 10.0.
    hard_factor: If not ``None``, terminate when growth exceeds this.
```
````

### `Monitor`

*class*

```python
Monitor(*args, **kwargs)
```

Structural protocol for a simulation monitor.

````{admonition} Details
:class: dropdown

```text
Implementations need a ``name`` and the three lifecycle hooks. Most
monitors only care about one hook, so :class:`somax.monitor.BaseMonitor`
supplies inert defaults — subclass it and override what you need rather
than implementing this Protocol directly.
```
````

### `MonitorVerdict`

*class*

```python
MonitorVerdict(metrics: 'dict[str, float]' = <factory>, messages: 'tuple[str, ...]' = (), terminate: 'bool' = False, reason: 'str | None' = None) -> None
```

A monitor's response to a chunk. Inert by default.

````{admonition} Details
:class: dropdown

```text
Args:
    metrics: Scalar metrics to merge into the run's per-chunk diagnostics
        (name -> value). Empty by default.
    messages: Lines to emit to ``run.log`` (prefixed with the monitor
        name). Empty by default.
    terminate: Whether the monitor requests a clean stop of the run.
    reason: Why termination was requested. Required (non-``None``) when
        ``terminate`` is ``True``; ignored otherwise.
```
````

### `NonFiniteMonitor`

*class*

```python
NonFiniteMonitor()
```

FAIL-HARD: terminate the run if any state field holds NaN/Inf.

````{admonition} Details
:class: dropdown

```text
The pluggable form of the runner's original non-finite abort. A
non-finite state makes everything downstream meaningless, so this always
requests termination.
```
````

### `SolverHealthMonitor`

*class*

```python
SolverHealthMonitor()
```

MONITOR: surface diffrax per-chunk solver statistics.

````{admonition} Details
:class: dropdown

```text
Consumes the solver stats the runner now threads onto :class:`ChunkInfo`
(via a per-chunk ``stats`` attribute set by the runner). Reports the
rejected-step rate — often the earliest blow-up signal, before any field
check trips — and flags a non-success diffrax result.
```
````

### `ThroughputMonitor`

*class*

```python
ThroughputMonitor()
```

MONITOR: report simulated seconds per wallclock second per chunk.

````{admonition} Details
:class: dropdown

```text
A sudden drop flags a recompilation or a host-side stall.
```
````

### `WatchdogMonitor`

*class*

```python
WatchdogMonitor(max_wall_s: 'float')
```

FAIL-HARD: terminate if cumulative wallclock exceeds a ceiling.

````{admonition} Details
:class: dropdown

```text
A wallclock budget for the whole run. The runner already ticks an
alive-thread; this enforces a hard ceiling and stops cleanly at the next
chunk boundary rather than running indefinitely.

Args:
    max_wall_s: Maximum cumulative wallclock seconds before termination.
```
````

### `default_monitors`

*function*

```python
default_monitors() -> 'list[BaseMonitor]'
```

The runner's default monitor set — preserves legacy behavior.

````{admonition} Details
:class: dropdown

```text
``NonFiniteMonitor`` (the original abort) plus ``EnergyGrowthMonitor`` (the
original 10x warning), now pluggable. ``ConservationDriftMonitor``,
``SolverHealthMonitor`` and ``ThroughputMonitor`` add record-only
diagnostics on top without changing run outcomes.
```
````

## Solvers

Matrix-free IMEX integration helpers for stiff term models.

### `imex_solver`

*function*

```python
imex_solver(*, rtol: 'float' = 0.0001, atol: 'float' = 1e-06, gmres_restart: 'int' = 20) -> 'dfx.AbstractSolver'
```

Build a ``KenCarp3`` IMEX solver with a matrix-free implicit stage.

````{admonition} Details
:class: dropdown

```text
The implicit stage uses an ``optimistix.Newton`` root-finder backed by a
``lineax.GMRES`` linear solver, so the stiff (implicit) sub-problem is
solved Jacobian-free — O(N) memory instead of the O(N^2) dense Jacobian
that ``diffrax.KenCarp3()``'s default uses (the #55 OOM at 256x256).

Args:
    rtol: Relative tolerance for the implicit Newton solve.
    atol: Absolute tolerance for the implicit Newton solve.
    gmres_restart: GMRES restart length (Krylov subspace size before
        restart). Larger converges in fewer outer iterations at higher
        per-iteration memory; 20 is a safe default.

Returns:
    A ``diffrax`` IMEX solver suitable for ``model.integrate(...,
    solver=...)`` on a term model built with ``imex=True``. Use it with
    :func:`imex_stepsize_controller` (or any adaptive controller carrying
    matching tolerances).
```
````

### `imex_stepsize_controller`

*function*

```python
imex_stepsize_controller(*, rtol: 'float' = 0.0001, atol: 'float' = 1e-06) -> 'dfx.AbstractStepSizeController'
```

Build an adaptive PID controller for an IMEX solve.

````{admonition} Details
:class: dropdown

```text
A fixed-step controller with an implicit solver requires the implicit
tolerances to be specified and cannot reject a step when the Newton solve
fails to converge; an adaptive controller is the documented fix (and the
somax default ``ConstantStepSize`` does not satisfy the IMEX requirement).
The tolerances here should match those passed to :func:`imex_solver`.

Args:
    rtol: Relative tolerance for adaptive step-size control.
    atol: Absolute tolerance for adaptive step-size control.

Returns:
    A ``diffrax.PIDController`` for ``model.integrate(...,
    stepsize_controller=...)``.
```
````

## Data Assimilation

Adapters wiring somax models into the filterax / vardax DA stack (requires the optional ``da`` dependency group).

These symbols live in `somax.da` and require the optional `da` dependency group (`uv sync --group da`):

- `somax.da.SomaxDynamics`
- `somax.da.SomaxForwardModel`
- `somax.da.SubsampleObs`
- `somax.da.make_ensemble`
- `somax.da.state_to_vector`
