# Arakawa C-Grids

This section covers the fundamentals of staggered grids for ocean modeling.

## Introduction

The Arakawa C-grid is the standard staggering arrangement used in ocean general circulation models. Variables are placed at different locations on the grid cell:

- **h** (sea surface height, tracers) at cell centers
- **u** (zonal velocity) on east/west cell faces
- **v** (meridional velocity) on north/south cell faces
- **vorticity** at cell corners (nodes)

## Why Staggered Grids?

Staggered grids avoid the computational mode (checkerboard instability) that arises with co-located grids, and naturally represent the divergence and curl operators.

## finitevolX

The `finitevolX` library provides the discrete operators on Arakawa C-grids used by somax. See the [finitevolX repository](https://github.com/jejjohnson/finitevolX) for details.
