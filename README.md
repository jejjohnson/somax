# somax — Simple Ocean Models in JAX

[![Tests](https://github.com/jejjohnson/somax/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/somax/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/somax/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/somax/actions/workflows/lint.yml)
[![codecov](https://codecov.io/gh/jejjohnson/somax/branch/main/graph/badge.svg?token=YGPQQEAK91)](https://codecov.io/gh/jejjohnson/somax)

> A model zoo for studying ocean and atmosphere dynamics through simple ODEs/PDEs, built on JAX.

---

## Motivation

Large-scale general circulation models (MITGCM, NEMO, MOM6) are powerful but complex. Smaller-scale GCMs in modern languages (Veros, Oceananigans) exist, but there is a gap for researchers who want to experiment with *simple* ocean-like dynamical systems as a stepping-stone to more complex models.

somax bridges this gap by aggregating canonical models — from Lorenz attractors to quasi-geostrophic and shallow water systems — with a unified JAX-based API that supports automatic differentiation, GPU acceleration, and modern scientific computing workflows.

---

## Architecture

somax is organized in three layers:

| Layer | Location | Purpose |
|-------|----------|---------|
| **Installable library** | `somax/` | Reusable model classes built on [finitevolX](https://github.com/jejjohnson/finitevolX) |
| **Simulation infrastructure** | `scripts/`, `configs/` | Ready-to-run simulations with Hydra + DVC |
| **Jupyter Book** | `content/` | Theory and practice documentation (MyST) |

### Key Dependencies

| Package | Role |
|---------|------|
| [finitevolX](https://github.com/jejjohnson/finitevolX) | Discrete operators on Arakawa C-grids |
| [diffrax](https://github.com/patrick-kidger/diffrax) | Time integration and adjoint methods |
| [equinox](https://github.com/patrick-kidger/equinox) | PyTree-based modules |
| [Hydra](https://hydra.cc/) / [hydra-zen](https://github.com/mit-ll-responsible-ai/hydra-zen) | Configuration composition |
| [DVC](https://dvc.org/) | Data versioning and pipeline DAGs |
| [pixi](https://pixi.sh/) | Environment management and task runner |

---

## Models

### Currently shipped in the installable library (`somax/`)

- [x] Lorenz '63 / '96 chaotic systems

### Planned (will be rewritten against the new Core API)

- [ ] Linear Shallow Water Model
- [ ] Shallow Water Model
- [ ] Quasi-Geostrophic Model (barotropic)
- [ ] Multi-Layer Quasi-Geostrophic Model
- [ ] Multi-Layer Shallow Water Model

---

## Installation

### With pixi (recommended)

[pixi](https://pixi.sh/) manages the full environment including non-Python dependencies (DVC, Node.js for docs, JupyterLab):

```bash
pixi install
```

### With uv (library only)

```bash
uv sync --all-groups
```

### With pip

```bash
pip install git+https://github.com/jejjohnson/somax.git
```

---

## Quick Start

### Run a simulation

```bash
# With pixi tasks
pixi run simulate-swm
pixi run simulate-mqg

# With DVC pipelines
pixi run dvc repro simulate-swm
```

### Override configuration from the command line (Hydra)

```bash
python scripts/swm.py domain.nx=400 domain.ny=200 timestepping.dt=150
```

### Run tests

```bash
# With pixi
pixi run test

# With uv
uv run pytest -v
```

### Lint and format

```bash
# With pixi
pixi run lint
pixi run format

# With uv / make
make lint
make format
```

---

## Project Structure

```
somax/
├── somax/              # Installable library
│   ├── _src/           # Internal implementation
│   │   ├── models/     # SWM, QG, Lorenz systems
│   │   ├── operators/  # Spatial and differential operators
│   │   ├── domain/     # Grid and domain definitions
│   │   └── ...
│   └── *.py            # Public API modules
├── scripts/            # Simulation entry points
├── configs/            # Hydra configuration
│   ├── simulation/     # SWM, MQG run configs
│   └── model/          # Model variant configs
├── content/            # Jupyter Book (MyST) documentation
├── tests/              # Test suite
├── data/               # DVC-managed simulation outputs
├── pixi.toml           # pixi environment + tasks
├── pyproject.toml      # PEP 621 project metadata (hatchling)
├── dvc.yaml            # DVC pipeline definition
└── myst.yml            # Jupyter Book configuration
```

---

## Development

```bash
# Install everything (deps + pre-commit hooks)
make install

# Run the full quality suite
make lint          # ruff check
make format        # ruff format + fix
make typecheck     # ty type checker
make test          # pytest (no coverage)
make test-cov      # pytest with coverage
make precommit     # all pre-commit hooks
```

---

## GPU Support

JAX supports GPU acceleration out of the box. After installing somax, install the GPU-enabled JAX build:

```bash
pip install --upgrade "jax[cuda12]"
```

See the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) for details.

---

## License

[MIT](LICENSE)
