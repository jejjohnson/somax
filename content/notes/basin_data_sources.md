# Real-Basin Data Sources: Geometry + Forcing

> **TL;DR.** Phase 4 / Phase 5 of the scenario x model refactor (epic [#72](https://github.com/jejjohnson/somax/issues/72)) needs land masks, bathymetry, and surface forcing for four basins (`north_atlantic`, `med_sea`, `gulf_stream`, `southern_ocean`). This note pins the decisions so [#78](https://github.com/jejjohnson/somax/issues/78) and [#79](https://github.com/jejjohnson/somax/issues/79) can proceed:
>
> 1. **Bathymetry:** GEBCO 2024 (15 arc-sec global grid).
> 2. **Land mask:** derived from GEBCO (depth >= 0).
> 3. **Forcing:** ERA5 monthly-mean surface stresses + net heat flux, averaged over **1993-2023** into an annual-mean climatology. Seasonal cycles are a follow-up.
> 4. **Format:** one **Zarr v3** store per basin (`data/basin/<name>.zarr`) containing `mask`, `bathymetry`, `tau_x`, `tau_y`, `heat`, and lon/lat coordinate variables.
> 5. **Pipeline:** a DVC stage per basin (`build-basin-<name>`) in `dvc.yaml`, invoking `scripts/data/build_basin.py`. `dvc repro` regenerates when source data or the script changes.
> 6. **Storage:** **bring-your-own-remote**. somax ships the pipeline and CLI helpers; each user points DVC at their own remote (Google Drive, S3, Azure Blob, local disk, etc.). No canonical somax-hosted bundle.
> 7. **Helpers:** `somax-sim data {init,fetch,build,status}` wraps the DVC flow so users never have to learn DVC's command surface just to get basin bundles.
> 8. **Coordinates:** Cartesian beta-plane (Mercator-projected) for `north_atlantic`, `med_sea`, `gulf_stream`. Lat/lon native for `southern_ocean`.

This note closes the "open questions" in [#74](https://github.com/jejjohnson/somax/issues/74) (geometry) and [#75](https://github.com/jejjohnson/somax/issues/75) (forcing). The loader and per-basin `_build()` fillers belong to Phase 4 ([#78](https://github.com/jejjohnson/somax/issues/78)) / Phase 5 ([#79](https://github.com/jejjohnson/somax/issues/79)); this PR scaffolds the pipeline + helpers and ships a synthetic-data placeholder build so the wiring is exercised end-to-end today.

## Why both questions in one note

[#74](https://github.com/jejjohnson/somax/issues/74) and [#75](https://github.com/jejjohnson/somax/issues/75) were filed separately because geometry and forcing are physically distinct. But the *engineering* questions they raise — distribution, format, resolution, versioning, licensing — are identical. Answering them separately risks diverging on arbitrary details. One decision, one bundle, one pipeline.

## Per-basin canonical grids

Each basin ships at **one** canonical grid chosen to keep smoke tests fast and production runs tractable on a single GPU:

| Basin            | Shape   | Extent                                 | Reason                                 |
| ---------------- | ------- | -------------------------------------- | -------------------------------------- |
| `med_sea`        | 128x64  | ~3500 x 1500 km, ~25 km/cell           | Smallest basin — fast smoke target     |
| `gulf_stream`    | 256x128 | ~4000 x 2000 km, ~15 km/cell           | Eddy-resolving WBC reference           |
| `north_atlantic` | 256x256 | ~7000 x 7000 km, ~25-30 km/cell        | Full-basin QG reference                |
| `southern_ocean` | 360x90  | 0-360E, 70S-30S (spherical cap)        | ACC-relevant; eddy-permitting on sphere |

Users who want a different resolution invoke `somax-sim data build <name> --nx <N> --ny <M>`. The *committed* pipeline output is one canonical grid per basin.

## Bathymetry: GEBCO 2024

### Source

[GEBCO 2024](https://www.gebco.net/data_and_products/gridded_bathymetry_data/) is a 15 arc-second (~450 m at the equator) global bathymetry grid. De facto community standard for ocean modeling, freely redistributable with attribution.

### Rejected alternatives

- **ETOPO 2022** (1 arc-min, NOAA) — coarser than GEBCO at no ergonomic gain.
- **SRTM15+** (15 arc-sec, Scripps) — comparable to GEBCO but less widely cited in oceanography.
- **Model-native bathymetries** (e.g., MITgcm's `bathymetry.bin`) — not a general source; tied to one model's historical grids.

### Derived mask

The wet/dry mask for each basin is **derived from GEBCO** at build time: `mask = (bathymetry < 0)`. Natural Earth 10m coastlines are used only as a QC overlay, not as the mask itself. Rationale: a single dataset eliminates mask/bathymetry inconsistency bugs — land cells with finite ocean depth, or vice versa — which are the most common kind of real-basin configuration error.

### Minimum depth floor

Shallow shelves (depth < 20 m) are raised to a configurable floor `min_depth` before regridding to avoid sub-grid-scale topography destabilizing the time step. Default floor is 20 m; `--min-depth 0` disables it. The floor is recorded in the Zarr store's `attrs["history"]`.

## Surface forcing: ERA5

### Source

[ERA5](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means) (ECMWF Reanalysis v5) provides 0.25° x 0.25° global monthly-mean fields from 1940-present. Variables we consume:

| ERA5 variable | Meaning                                   | Unit   |
| ------------- | ----------------------------------------- | ------ |
| `metss`       | Mean eastward turbulent surface stress    | N/m²   |
| `mntss`       | Mean northward turbulent surface stress   | N/m²   |
| `msnlwrf`     | Mean surface net longwave radiation flux  | W/m²   |
| `msnswrf`     | Mean surface net shortwave radiation flux | W/m²   |
| `mslhf`       | Mean surface latent heat flux             | W/m²   |
| `msshf`       | Mean surface sensible heat flux           | W/m²   |

Net heat flux stored in the bundle is `heat = msnlwrf + msnswrf + mslhf + msshf` (CF sign convention: positive downward into the ocean).

### Temporal aggregation

**Annual mean over 1993-2023** (31 years). Rationale:

- 1993 onward is the altimetry era — forcing is observationally constrained.
- 31 years averages out ENSO-decadal variability while staying inside the satellite era.
- Annual-mean is sufficient for Phase 4 (stationary forcing). Monthly climatology and seasonal cycles are a follow-up.

### Rejected alternatives

- **CORE-II** (Large & Yeager, 1948-2009) — canonical for OMIP but older and no longer actively maintained.
- **NCEP / NCEP2** (1948-present) — coarser (1.875°) and known to have persistent biases in tropical heat fluxes.
- **JRA-55** (JMA) — comparable quality to ERA5 but with a less convenient open-data license for research redistribution.
- **OAFlux** (heat fluxes only) — considered as a merge with ERA5 but adds complexity for modest gain.

### Licensing

ERA5 is distributed under the [Copernicus License](https://apps.ecmwf.int/datasets/licences/copernicus/), which permits redistribution of derived products with attribution. Each basin's Zarr store carries a `source` attr citing "Copernicus Climate Change Service (C3S) — ERA5 monthly-mean single levels" and a DOI pointer.

## Spatial regridding

Regridding happens **once, at build time** (inside the DVC stage). Runtime loads the pre-regridded Zarr directly — no regridding dependencies at runtime.

- **Bathymetry:** conservative (area-weighted). Preserves ocean-basin volume — critical for QG inversion.
- **Wind stress, heat flux:** bilinear. Conservative is scientifically preferred but introduces an `xesmf` runtime dep; at these grid sizes, bilinear differs by a fraction of a percent.
- **Coast-adjacent cells:** fill-in via nearest-neighbor from the closest ocean cell before regridding, to avoid land-contaminated stresses at the coast.

Regridding code lives in `scripts/data/build_basin.py` (build-time only, not imported at runtime). somax's *runtime* deps stay `xarray + numpy + jax` — no `xesmf`, no `pyproj`.

## Coordinate handling

### Cartesian basins (north_atlantic, med_sea, gulf_stream)

Each basin is projected onto a local **Mercator-like Cartesian grid** at build time, using a basin-central reference latitude:

- `med_sea`: 38°N
- `gulf_stream`: 35°N
- `north_atlantic`: 40°N

The Zarr store records `(Lx, Ly)` in metres plus `lon`/`lat` coordinate variables for every T-cell (for plotting against a map). `Geometry.kind == "real_basin"` signals the Cartesian beta-plane regime to the compatibility checker.

### Spherical basin (southern_ocean)

`southern_ocean` stays in native lat/lon (spherical cap, 70°S-30°S). `Geometry.kind == "spherical_cap"`. Consumed by Phase 5 ([#79](https://github.com/jejjohnson/somax/issues/79)) + spherical models ([#73](https://github.com/jejjohnson/somax/issues/73)).

## Distribution: DVC + bring-your-own-remote

### Decision

somax does **not** host a canonical basin bundle. The repo ships:

1. **The pipeline.** `dvc.yaml` stages (`build-basin-med-sea`, ...) that regenerate each basin from source data.
2. **The build script.** `scripts/data/build_basin.py` — reads GEBCO + ERA5, regrids, writes `data/basin/<name>.zarr`.
3. **DVC tracking.** Each bundle is a DVC output; the `.dvc` pointer is committed to git; the Zarr store itself is cached by DVC (not in git).
4. **CLI helpers.** `somax-sim data {init,fetch,build,status}` — see below.

The user brings their own remote. First-time setup:

```console
$ somax-sim data init
Pick a DVC remote backend:
  [1] Google Drive  (free, personal projects)
  [2] Amazon S3
  [3] Azure Blob
  [4] Local filesystem  (single machine / NFS)
  [q] Quit
> 1
Folder ID (from your Google Drive URL): 1a2b3c...
Installing dvc-gdrive ...
Adding DVC remote 'somax-data' ...
$ somax-sim data fetch med_sea
Fetching data/basin/med_sea.zarr from remote 'somax-data' ...
$ somax-sim data build gulf_stream --push
Building data/basin/gulf_stream.zarr (takes ~5 min) ...
Pushing to remote 'somax-data' ...
```

### Rejected alternatives

- **Bundle in the repo** — rejected. Bundles are ~5-50 MB/basin; committing them bloats git and pushes the repo past comfortable clone sizes.
- **Git LFS** — rejected. LFS bandwidth on public GitHub is metered; CI pulls hit the quota.
- **pooch** — rejected. pooch exists to give you fetch-with-hash-cache semantics when you don't have a proper versioning system. somax *does* have DVC; pooch would be redundant.
- **somax-hosted canonical bundle** (Zenodo, author's GDrive, GitHub Release) — rejected. Forces us to pick a platform, maintain credentials, and handle takedown / license complaints for upstream (Copernicus / GEBCO) attribution. BYO-remote sidesteps all of it.

### Versioning

No canonical bundle means no canonical versioning. Each user's remote is independently versioned via DVC's content-hash mechanism. The *build script* is versioned in git — reproducing a bundle from a given git commit is deterministic if the user's source GEBCO/ERA5 inputs are pinned.

## File format: Zarr v3, one store per basin

### Layout (one `.zarr` dir per basin under `data/basin/`)

```
data/basin/med_sea.zarr/
  .zarray, .zgroup, .zattrs
  lon/         (y, x) float32  degrees_east
  lat/         (y, x) float32  degrees_north
  bathymetry/  (y, x) float32  m, positive downward
  mask/        (y, x) int8     1 = ocean, 0 = land
  tau_x/       (y, x) float32  Pa, eastward
  tau_y/       (y, x) float32  Pa, northward
  heat/        (y, x) float32  W/m^2, positive into ocean

root attrs:
  title: "somax basin bundle: med_sea"
  source: "GEBCO 2024 + ERA5 1993-2023 annual mean"
  history: "built by scripts/data/build_basin.py <commit>"
  conventions: "CF-1.10"
  somax_basin_version: "<semver>"
```

### Rejected alternatives

- **NetCDF4** — considered and rejected at reviewer's request. somax already uses Zarr v3 for *simulation outputs* ([`somax/_src/io/xarray.py`](../../somax/_src/io/xarray.py)); using the same format for inputs avoids a dual-format codebase.
- **HDF5 direct** — no CF semantics; reinvents the wheel.
- **Separate files per variable** (e.g., `med_sea_mask.npy`) — inflates the DVC pointer surface 5x and makes cross-variable coordinate consistency implicit rather than enforced at the store level.

## Synthetic-data placeholder (this PR)

[#78](https://github.com/jejjohnson/somax/issues/78) is where real GEBCO/ERA5 integration lands. Until then, `build_basin.py` runs in a **`--synthetic` mode** that constructs:

- A rectangular-basin mask with a plausible coastline shape per basin (hand-drawn polygon).
- A constant-depth bathymetry at 4000 m with shelf ramps near the coast.
- Stommel-style sinusoidal `tau_x`, zero `tau_y`, zero `heat`.

These are **placeholders** — the DVC pipeline is real, the Zarr schema is real, the CLI is real. Only the physical content is synthetic. Swapping in real data in [#78](https://github.com/jejjohnson/somax/issues/78) is a one-function change inside `build_basin.py`.

## Out of scope (explicit non-goals)

Deferred to future issues:

- **Time-varying forcing** (monthly climatology / seasonal cycle). Phase 4 is stationary.
- **Assimilation-ready data** (observational snapshots for DA experiments).
- **Tracer forcing** (salinity, nutrients, DIC).
- **Coupled / air-sea feedback** — somax ships prescribed forcing; bulk formulae are future work.
- **Tidal forcing.**
- **Sub-grid orography / mesoscale terrain beyond the canonical resolution.**
- **A somax-hosted canonical bundle.** Explicitly rejected above; re-raise only if a specific publication needs a DOI-tracked artifact.

## Summary — answers to the open questions

Cross-referenced to [#74](https://github.com/jejjohnson/somax/issues/74) and [#75](https://github.com/jejjohnson/somax/issues/75):

| Question                                             | Decision                                                  | Section |
| ---------------------------------------------------- | --------------------------------------------------------- | ------- |
| [#74](https://github.com/jejjohnson/somax/issues/74) Q1 — bathymetry source            | GEBCO 2024                                                | Bathymetry |
| [#74](https://github.com/jejjohnson/somax/issues/74) Q2 / [#75](https://github.com/jejjohnson/somax/issues/75) Q4 — distribution         | DVC + BYO remote                                          | Distribution |
| [#74](https://github.com/jejjohnson/somax/issues/74) Q3 / [#75](https://github.com/jejjohnson/somax/issues/75) Q2 — format               | Zarr v3, one store per basin                              | File format |
| [#74](https://github.com/jejjohnson/somax/issues/74) Q4 — resolution policy            | one canonical grid per basin, rebuildable via CLI         | Per-basin grids |
| [#74](https://github.com/jejjohnson/somax/issues/74) Q5 — coordinate handling          | Cartesian beta-plane per basin; spherical for southern_ocean | Coordinates |
| [#75](https://github.com/jejjohnson/somax/issues/75) Q1 — forcing source               | ERA5 monthly-means, 1993-2023                             | Surface forcing |
| [#75](https://github.com/jejjohnson/somax/issues/75) Q2 — time aggregation             | annual mean (Phase 4); seasonal deferred                  | Surface forcing |
| [#75](https://github.com/jejjohnson/somax/issues/75) Q3 — spatial regridding           | bilinear for stress/heat, conservative for bathymetry, precomputed | Regridding |
| [#75](https://github.com/jejjohnson/somax/issues/75) Q5 — stationary vs time-varying   | stationary only; time-varying is a follow-up              | Surface forcing |

Real-data loader and per-basin `_build()` implementations belong to Phase 4 ([#78](https://github.com/jejjohnson/somax/issues/78)) and Phase 5 ([#79](https://github.com/jejjohnson/somax/issues/79)).
