# somax: Simple Ocean Models in JAX

**somax** is a model zoo for studying ocean and atmosphere dynamics through simple ODEs and PDEs, built on JAX.

## Overview

This book provides both the **theory** and **practice** of building ocean dynamical models using modern scientific computing tools. It covers:

- **Arakawa C-grids** and finite volume discretization
- **Shallow Water Models** (linear and nonlinear)
- **Quasi-Geostrophic Models** (barotropic and multi-layer)
- **Chaotic Dynamical Systems** (Lorenz '63, Lorenz '96)

## Getting Started

```bash
# Install with pixi
pixi install

# Run a simulation
pixi run simulate-swm

# Launch Jupyter Lab
pixi run -e jupyterlab lab
```

## Architecture

somax is structured in three layers:

1. **Installable library** (`somax/`) — reusable model classes built on finitevolX
2. **Simulation infrastructure** (`scripts/`, `configs/`) — ready-to-run simulations with Hydra + DVC
3. **Jupyter Book** (`content/`) — theory and practice documentation
