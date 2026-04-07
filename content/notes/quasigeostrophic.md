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
