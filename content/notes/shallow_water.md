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
