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

somax is organized in three layers, with the simulation runner sitting on a five-tier stack:

| Layer | Location | Purpose |
|-------|----------|---------|
| **Installable library** | `somax/` | Reusable model classes built on [finitevolX](https://github.com/jejjohnson/finitevolX) |
| **Simulation runner** | `somax/_src/cli/`, `configs/`, `scripts/` | `somax-sim` CLI, RunSpec dataclass, authored configs, DVC pipeline |
| **Jupyter Book** | `content/` | Theory and practice documentation (MyST) |

The simulation runner stack:

```
DVC pipeline       → caches artifacts, tracks deps, drives `dvc exp show`
Authored configs   → Python in configs/_authoring/*.py → YAML in configs/simulation/*.yaml
Cyclopts CLI       → somax-sim run / spinup / restart / list-* / show-config
RunSpec dataclass  → canonical schema (testcase + timestepping + output + debug)
somax library      → models, factories, IO layer
```

### Key Dependencies

| Package | Role |
|---------|------|
| [finitevolX](https://github.com/jejjohnson/finitevolX) | Discrete operators on Arakawa C-grids |
| [diffrax](https://github.com/patrick-kidger/diffrax) | Time integration and adjoint methods |
| [equinox](https://github.com/patrick-kidger/equinox) | PyTree-based modules |
| [cyclopts](https://github.com/BrianPugh/cyclopts) | `somax-sim` CLI shell |
| [xarray](https://docs.xarray.dev) + [zarr](https://zarr.dev/) | Self-describing chunked snapshot persistence |
| [DVC](https://dvc.org/) | Data versioning and pipeline DAGs |
| [pixi](https://pixi.sh/) | Environment management and task runner |

---

## Models

Currently shipped in the installable library (`somax/`):

- [x] Lorenz '63 / '96 / '96-two-layer chaotic systems
- [x] 1D / 2D linear convection, diffusion, Burgers
- [x] 2D Laplace, Poisson, Helmholtz solvers
- [x] 2D incompressible Navier–Stokes (cavity, channel)
- [x] Linear and nonlinear Shallow Water Models (1D and 2D)
- [x] Multi-Layer Shallow Water Model
- [x] Barotropic Quasi-Geostrophic Model
- [x] Baroclinic (multi-layer) Quasi-Geostrophic Model
- [x] Reparameterized Quasi-Geostrophic Model

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

### Run a simulation via the cyclopts CLI

```bash
# Discover available test cases and models
somax-sim list-testcases
somax-sim list-models

# Run a single simulation from a config (and inspect the resolved spec)
somax-sim show-config configs/simulation/swm_jet.yaml
somax-sim run --config configs/simulation/swm_jet.yaml --output-dir data/simulations/swm_jet

# Smoke-test mode: smaller grid, shorter run, frequent snapshots
somax-sim run --config configs/simulation/doublegyre_bc_qg.yaml \
              --output-dir /tmp/smoke --debug
```

### Run the full reference pipeline (DVC)

```bash
# All canonical reference simulations end-to-end
pixi run sim                # = dvc repro

# Or one stage at a time
pixi run sim-swm-jet        # 30-day baroclinic-instability SWM (64²)
pixi run sim-bt-qg          # 1-year barotropic QG double-gyre (64²)
pixi run spinup-bc-qg       # 3-year spinup of multilayer QG (128²)
pixi run sim-bc-qg          # 1-year production restart (128²)
```

DVC tracks the snapshot zarr stores and the metrics JSON; `dvc exp show`
displays params and metrics across runs. See
[`content/notes/qg_spinup_durations.md`](content/notes/qg_spinup_durations.md)
for the spinup-duration ladder used by the BC QG stages.

### Spinup → restart workflow

```bash
# 1. Spinup writes a final_state.zarr restart artifact
somax-sim spinup --config configs/simulation/spinup_bc_qg.yaml \
                 --output-dir data/spinup/bc_qg

# 2. Production run starts from that state — model is rebuilt from
#    the production config; only the state is loaded.
somax-sim restart --config configs/simulation/doublegyre_bc_qg.yaml \
                  --from data/spinup/bc_qg/final_state.zarr \
                  --output-dir data/simulations/doublegyre_bc_qg
```

### Author your own configs

Pipeline configs are authored in Python under `configs/_authoring/*.py`
and materialized to YAML by `python scripts/build_configs.py` (which is
itself a DVC stage). Edit a Python file, run `dvc repro`, and DVC
re-runs only the downstream stages whose configs actually changed.

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
│   │   ├── models/     # SWM, QG, Lorenz, PDE systems
│   │   ├── core/       # Model contract, types, forcing, helmholtz
│   │   ├── io/         # zarr-backed xarray persistence layer
│   │   ├── cli/        # somax-sim CLI: spec, app, _run, _factories
│   │   └── ...
│   └── *.py            # Public API modules
├── scripts/
│   └── build_configs.py    # Materializes _authoring/*.py → simulation/*.yaml
├── configs/
│   ├── _authoring/         # Python config sources (edit these)
│   └── simulation/         # Generated YAMLs (do NOT edit)
├── content/                # Jupyter Book (MyST) documentation
├── tests/                  # Test suite (incl. test_io_xarray, test_cli_spec, test_configs)
├── data/                   # DVC-managed simulation + spinup outputs (gitignored)
├── pixi.toml               # pixi environment + tasks
├── pyproject.toml          # PEP 621 project metadata (hatchling)
├── dvc.yaml                # DVC pipeline definition
└── myst.yml                # Jupyter Book configuration
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
