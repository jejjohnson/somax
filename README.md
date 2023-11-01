# Simple Ocean Models JAX (In Progress)
[![CodeFactor](https://www.codefactor.io/repository/github/jejjohnson/somax/badge)](https://www.codefactor.io/repository/github/jejjohnson/somax)
[![codecov](https://codecov.io/gh/jejjohnson/somax/branch/main/graph/badge.svg?token=YGPQQEAK91)](https://codecov.io/gh/jejjohnson/somax)

> This repo hosts some models which are useful for studying ocean dynamics via *simple* ODES/PDEs.
> This is also a demonstration for how one can use different pieces from other packages.
> This repo is more of a model zoo to showcase how we can stitch these different pieces together to build simple, robust PDEs for research.


---
## Motivation

There exists many large-scale general circulation models within the oceanography community, e.g. MITGCM, NEMO, MOM6, etc.
There are even some more smaller scale GCMs that are written in more modern languages, e.g. veros, Oceananigans, etc.
However, there is a missing step for many researchers to experiment with different, simple ocean-like dynamical systems as a stepping-stone to get to the more complex system.
There exists a wealth of literature available for the community regarding the Lorenz family so that users can experiment with chaotic systems.
There is also a number of key packages to allow users to experiment with simple models like the Quasigeostrophic model.
There are also tidbit packages for shallow water models that are scattered through-out GitHub.
However, we're missing a simple package which tries to aggregate the latest research on models like the QG and SWM.
This package attempts to bridge this gap.


---

*Functional API**.
We use the `finitevolx` package for implementing spatial operators with finite volume considerations. 
We also use the `spectraldiffx` package for implementing spatial operators with the pseudospectral considerations.
For timesteppers, we use the `diffrax` package for implementing advanced time stepping schemes like N-order Runge-Kutta or Heun.

**Field API**. (TODO)
We have an operator API which will allow easier use of defining PDEs. 
This makes use of the `fieldx` package which defines custom spatiotemporal containers for fields which mimic the `xarray` package.


---
## Key Features

**Models**.
* [ ] 
* [ ] Linear Shallow Water Model
* [ ] Quasi-Geostrophic Model
* [ ] Shallow Water Model
* [ ] Multi-Layer Quasi-Geostrophic Modle
* [ ] Multi-Layer Shallow Water Model

**Elliptical Solvers**.
We have the Discrete Sine Transform for the fast Poisson solver. 
We also have other solvers available from the JAX community.

**Dynamical Model API**.
We have nicely bundled PDEs for specific use cases, e.g. shallow water, (multilayer) quasi-geostrophic, etc.

---
## Installation

We can install it directly through pip

```bash
pip install git+https://github.com/jejjohnson/somax
```

We also use poetry for the development environment.

```bash
git clone https://github.com/jejjohnson/somax.git
cd somax
conda create -n somax python=3.11 poetry
poetry install
```

 diffrax (and equinox) with an finite difference discretization