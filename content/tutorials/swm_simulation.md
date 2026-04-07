# Shallow Water Model Simulation

This tutorial walks through running and analyzing a shallow water model simulation.

## Configuration

The SWM simulation is configured via `configs/simulation/swm.yaml`:

```yaml
domain:
  nx: 200
  ny: 100
  dx: 5.0e3   # 5 km grid spacing
  dy: 5.0e3

physics:
  gravity: 9.81
  coriolis_f0: 1.0e-4
  coriolis_beta: 1.6e-11
  depth_h0: 500.0

timestepping:
  dt: 300.0       # 5-minute timestep
  t_end: 86400.0  # 1-day simulation
```

## Running

```bash
pixi run simulate-swm
```

## Analysis

Load and visualize the simulation output:

```python
import xarray as xr
import matplotlib.pyplot as plt

ds = xr.open_dataset("data/simulations/swm/output.nc")
ds.h.isel(time=-1).plot()
plt.title("Sea Surface Height (final timestep)")
plt.show()
```
