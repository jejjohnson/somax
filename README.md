# Simple Ocean Models JAX (In Progress)
[![CodeFactor](https://www.codefactor.io/repository/github/jejjohnson/somax/badge)](https://www.codefactor.io/repository/github/jejjohnson/somax)
[![codecov](https://codecov.io/gh/jejjohnson/somax/branch/main/graph/badge.svg?token=YGPQQEAK91)](https://codecov.io/gh/jejjohnson/somax)

> This



---
## Key Features

**Models**.
* [ ] Linear Shallow Water Model
* [ ] Quasi-Geostrophic Model
* [ ] Shallow Water Model
* [ ] Multi-Layer Quasi-Geostrophic Modle
* [ ] Multi-Layer Shallow Water Model

**Elliptical Solvers**.
We have the Discrete Sine Transform for the fast solver. 
We also have other solvers available from the JAX community.

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