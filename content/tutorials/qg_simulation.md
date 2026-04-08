# Quasi-Geostrophic Model Simulation

This tutorial walks through running a multi-layer quasi-geostrophic double gyre simulation.

## Configuration

The MQG simulation is configured via `configs/simulation/mqg.yaml`:

```yaml
domain:
  nx: 256
  ny: 256
  dx: 2.0e3
  dy: 2.0e3

physics:
  num_layers: 2
  coriolis_f0: 9.375e-5
  coriolis_beta: 1.754e-11
  bottom_drag: 1.0e-7
  lateral_viscosity: 50.0
```

## Running

```bash
pixi run simulate-mqg
```

## Expected Behavior

The double gyre configuration produces:

- A subtropical gyre (anticyclonic) in the south
- A subpolar gyre (cyclonic) in the north
- A vigorous western boundary current separation
- Mesoscale eddy generation through baroclinic instability
