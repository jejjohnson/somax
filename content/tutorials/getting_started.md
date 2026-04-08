# Getting Started

## Installation

### With pixi (recommended)

```bash
pixi install
```

### With uv (library only)

```bash
uv sync --all-groups
```

## Running Your First Simulation

### Shallow Water Model

```bash
# With pixi
pixi run simulate-swm

# With DVC pipeline
pixi run dvc repro simulate-swm
```

### Multi-Layer Quasi-Geostrophic Model

```bash
pixi run simulate-mqg
```

## Configuration

Simulations are configured via Hydra YAML files in `configs/`. Override parameters from the command line:

```bash
python scripts/swm.py domain.nx=400 domain.ny=200 timestepping.dt=150
```

## Project Structure

```
somax/
├── somax/              # Installable library
├── scripts/            # Simulation entry points
├── configs/            # Hydra configuration
├── data/               # DVC-managed simulation outputs
├── content/            # Jupyter Book documentation
└── tests/              # Test suite
```
