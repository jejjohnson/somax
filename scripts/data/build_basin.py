"""Build a real-basin Zarr bundle for Phase 4 / Phase 5 scenarios.

Emits ``data/basin/<name>.zarr`` conforming to the schema documented in
``content/notes/basin_data_sources.md``:

  coords:   lon, lat
  vars:     bathymetry, mask, tau_x, tau_y, heat
  attrs:    title, source, history, conventions, somax_basin_version

Two modes:

  --source synthetic  (default; Phase 3 placeholder)
      Constructs a deterministic toy bundle from analytical shapes.
      The DVC pipeline, Zarr schema, and CLI helpers are exercised
      end-to-end, but the physical content is placeholder — not real
      bathymetry or wind stress.

  --source real       (Phase 4 / #78)
      Reads GEBCO 2024 + ERA5 1993-2023 monthly-mean climatologies,
      regrids to the basin's canonical grid, writes the bundle.
      Currently raises NotImplementedError — real-data integration
      lands with the Phase 4 loader work.

Run manually:

    python scripts/data/build_basin.py med_sea
    python scripts/data/build_basin.py gulf_stream --nx 128 --ny 64
    python scripts/data/build_basin.py north_atlantic --out data/basin/na.zarr

Or as a DVC stage:

    dvc repro build-basin-med-sea

The canonical per-basin shapes + reference latitudes are pinned in
``BASIN_SPECS``. Users who want a different resolution pass
``--nx / --ny`` (which invalidates the DVC cache, triggering a
rebuild on the next ``dvc repro``).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr


SOMAX_BASIN_VERSION = "0.1.0"

BasinName = Literal["med_sea", "gulf_stream", "north_atlantic", "southern_ocean"]

SourceKind = Literal["synthetic", "real"]


@dataclass(frozen=True)
class BasinSpec:
    """Canonical geometry + forcing scale for a basin.

    Args:
        nx: Default zonal cells.
        ny: Default meridional cells.
        Lx: Zonal extent (m) — for Cartesian basins only.
        Ly: Meridional extent (m) — for Cartesian basins only.
        lon_bounds: ``(lon_min, lon_max)`` in degrees.
        lat_bounds: ``(lat_min, lat_max)`` in degrees.
        ref_lat: Reference latitude used for the local Mercator
            projection (Cartesian basins only).
        geometry_kind: ``"real_basin"`` for Cartesian masked basins
            (``north_atlantic`` / ``med_sea`` / ``gulf_stream``);
            ``"spherical_cap"`` for ``southern_ocean``.
        tau0: Stommel-style wind-stress amplitude (Pa) used by the
            synthetic mode; ignored in real mode.
    """

    nx: int
    ny: int
    Lx: float | None
    Ly: float | None
    lon_bounds: tuple[float, float]
    lat_bounds: tuple[float, float]
    ref_lat: float | None
    geometry_kind: Literal["real_basin", "spherical_cap"]
    tau0: float = 0.1


BASIN_SPECS: dict[BasinName, BasinSpec] = {
    "med_sea": BasinSpec(
        nx=128,
        ny=64,
        Lx=3.5e6,
        Ly=1.5e6,
        lon_bounds=(-6.0, 36.5),
        lat_bounds=(30.0, 46.0),
        ref_lat=38.0,
        geometry_kind="real_basin",
        tau0=0.05,
    ),
    "gulf_stream": BasinSpec(
        nx=256,
        ny=128,
        Lx=4.0e6,
        Ly=2.0e6,
        lon_bounds=(-82.0, -40.0),
        lat_bounds=(25.0, 45.0),
        ref_lat=35.0,
        geometry_kind="real_basin",
        tau0=0.1,
    ),
    "north_atlantic": BasinSpec(
        nx=256,
        ny=256,
        Lx=7.0e6,
        Ly=7.0e6,
        lon_bounds=(-80.0, 0.0),
        lat_bounds=(10.0, 70.0),
        ref_lat=40.0,
        geometry_kind="real_basin",
        tau0=0.1,
    ),
    "southern_ocean": BasinSpec(
        nx=360,
        ny=90,
        Lx=None,
        Ly=None,
        lon_bounds=(0.0, 360.0),
        lat_bounds=(-70.0, -30.0),
        ref_lat=None,
        geometry_kind="spherical_cap",
        tau0=0.15,
    ),
}


# ----------------------------------------------------------------------
# Build entry point
# ----------------------------------------------------------------------


def build_basin(
    name: BasinName,
    *,
    source: SourceKind = "synthetic",
    nx: int | None = None,
    ny: int | None = None,
    min_depth: float = 20.0,
    out_path: Path | None = None,
) -> Path:
    """Build a single basin bundle and write it to disk.

    Returns the absolute path of the resulting ``.zarr`` store.
    """
    if name not in BASIN_SPECS:
        raise ValueError(
            f"unknown basin {name!r}; known: {sorted(BASIN_SPECS)!r}"
        )
    spec = BASIN_SPECS[name]
    nx_ = int(nx) if nx is not None else spec.nx
    ny_ = int(ny) if ny is not None else spec.ny
    if nx_ <= 0 or ny_ <= 0:
        raise ValueError(f"nx, ny must be > 0 (got {nx_}, {ny_})")

    out = Path(out_path) if out_path is not None else Path("data/basin") / f"{name}.zarr"
    out.parent.mkdir(parents=True, exist_ok=True)

    if source == "synthetic":
        ds = _build_synthetic(name, spec, nx=nx_, ny=ny_, min_depth=min_depth)
    elif source == "real":
        raise NotImplementedError(
            "real-data build (GEBCO + ERA5 integration) is tracked in #78; "
            "use --source synthetic for now."
        )
    else:
        raise ValueError(f"unknown source kind {source!r}; use 'synthetic' or 'real'")

    ds.attrs.update(
        title=f"somax basin bundle: {name}",
        source=_source_attr(source),
        history=_history_attr(name, source, nx_, ny_, min_depth),
        conventions="CF-1.10",
        somax_basin_version=SOMAX_BASIN_VERSION,
        basin_name=name,
        geometry_kind=spec.geometry_kind,
    )

    # zarr_format=3 + consolidated=False matches somax/_src/io/xarray.py
    # (consolidated metadata is not part of the v3 spec; xarray warns).
    if out.exists():
        _safe_rmtree_zarr(out)
    ds.to_zarr(out, mode="w", zarr_format=3, consolidated=False)
    return out.resolve()


def _safe_rmtree_zarr(out: Path) -> None:
    """Remove an existing Zarr store with guardrails.

    A plain ``shutil.rmtree(out)`` would recursively delete *any* path
    the caller passes — so a typo like ``--out data/basin`` would wipe
    every materialized bundle. Guard against that:

    - The path must have a ``.zarr`` suffix.
    - If it's a directory, it must contain a Zarr root marker
      (``zarr.json`` for v3, ``.zgroup`` / ``.zarray`` for v2).

    Otherwise we refuse and ask the user for a clean output path.
    """
    import shutil

    if out.suffix != ".zarr":
        raise ValueError(
            f"refusing to overwrite existing path {out!r}: output must have "
            "'.zarr' suffix so build_basin doesn't wipe unrelated directories."
        )
    if out.is_dir():
        markers = ("zarr.json", ".zgroup", ".zarray")
        if not any((out / m).exists() for m in markers):
            raise ValueError(
                f"refusing to remove existing directory {out!r}: no Zarr root "
                f"marker found ({markers!r}). Pass a fresh output path."
            )
    shutil.rmtree(out)


# ----------------------------------------------------------------------
# Synthetic mode
# ----------------------------------------------------------------------


def _build_synthetic(
    name: BasinName,
    spec: BasinSpec,
    *,
    nx: int,
    ny: int,
    min_depth: float,
) -> xr.Dataset:
    """Construct a placeholder basin with the right schema.

    The mask + bathymetry are analytical proxies for the named basin —
    enough to exercise mask-aware model code paths, but not physically
    accurate. Swap for real GEBCO-derived data in #78.
    """
    lon_min, lon_max = spec.lon_bounds
    lat_min, lat_max = spec.lat_bounds
    # Cell-center longitudes/latitudes on a regular lon/lat grid.
    lon_1d = np.linspace(lon_min, lon_max, nx, endpoint=False) + (
        lon_max - lon_min
    ) / nx / 2.0
    lat_1d = np.linspace(lat_min, lat_max, ny, endpoint=False) + (
        lat_max - lat_min
    ) / ny / 2.0
    lon, lat = np.meshgrid(lon_1d, lat_1d)  # (ny, nx)

    mask = _synthetic_mask(name, lon, lat)
    bathymetry = _synthetic_bathymetry(mask, depth_abyss=4000.0, min_depth=min_depth)
    tau_x, tau_y = _synthetic_wind_stress(lat, lat_min, lat_max, tau0=spec.tau0)
    heat = np.zeros_like(lon, dtype=np.float32)

    # Apply mask: land cells get zero-valued forcing in this placeholder
    # (real data may prefer NaN or a sentinel; Phase 4 / #78 will decide).
    tau_x = np.where(mask, tau_x, 0.0).astype(np.float32)
    tau_y = np.where(mask, tau_y, 0.0).astype(np.float32)

    ds = xr.Dataset(
        data_vars={
            "bathymetry": (("y", "x"), bathymetry.astype(np.float32)),
            "mask": (("y", "x"), mask.astype(np.int8)),
            "tau_x": (("y", "x"), tau_x),
            "tau_y": (("y", "x"), tau_y),
            "heat": (("y", "x"), heat),
        },
        coords={
            "lon": (("y", "x"), lon.astype(np.float32)),
            "lat": (("y", "x"), lat.astype(np.float32)),
        },
    )
    # CF-style variable attrs.
    ds["bathymetry"].attrs.update(
        long_name="bathymetry", units="m", positive="down"
    )
    ds["mask"].attrs.update(long_name="wet/dry mask", flag_values=[0, 1], flag_meanings="land ocean")
    ds["tau_x"].attrs.update(long_name="eastward surface wind stress", units="Pa")
    ds["tau_y"].attrs.update(long_name="northward surface wind stress", units="Pa")
    ds["heat"].attrs.update(long_name="net surface heat flux", units="W m-2", positive="down")
    ds["lon"].attrs.update(long_name="longitude", units="degrees_east")
    ds["lat"].attrs.update(long_name="latitude", units="degrees_north")

    if spec.Lx is not None and spec.Ly is not None:
        ds.attrs["Lx_m"] = float(spec.Lx)
        ds.attrs["Ly_m"] = float(spec.Ly)
    if spec.ref_lat is not None:
        ds.attrs["ref_lat_deg"] = float(spec.ref_lat)
    ds.attrs["lon_bounds"] = [float(lon_min), float(lon_max)]
    ds.attrs["lat_bounds"] = [float(lat_min), float(lat_max)]
    return ds


def _synthetic_mask(name: BasinName, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Return a plausible-looking ocean mask (1 = ocean, 0 = land).

    These are *not* real coastlines — they're analytical cutouts chosen
    so each basin visually resembles its namesake. Swap for GEBCO in #78.
    """
    ones = np.ones_like(lon, dtype=bool)
    if name == "med_sea":
        # Narrow strip with cutouts for Iberia / Italy / Greece / Turkey.
        mask = ones.copy()
        # North African coast: lat < 31 is land.
        mask &= lat >= 31.0
        # European shore: lat > 45 is land.
        mask &= lat <= 45.0
        # Italy peninsula (rough lozenge).
        italy = ((lon >= 7.0) & (lon <= 18.0)) & ((lat >= 36.0) & (lat <= 44.5))
        italy_sea = ((lon - 13.0) ** 2 / 6.0 ** 2 + (lat - 40.0) ** 2 / 3.5 ** 2 < 1.0)
        mask &= ~(italy & ~italy_sea)
        return mask
    if name == "gulf_stream":
        # North-American coast sweeping NE; mask land west of a curved shore.
        shore_lon = -82.0 + 0.8 * (lat - 25.0)  # linear NE-trending coast
        mask = lon > shore_lon
        # Northern limit of domain.
        mask &= lat <= 45.0
        return mask
    if name == "north_atlantic":
        # Americas to the west, Europe/Africa to the east.
        west_shore = -80.0 + 0.6 * (lat - 10.0)
        east_shore = -20.0 + 0.4 * (lat - 10.0)
        mask = (lon > west_shore) & (lon < east_shore)
        # Greenland block in the NW corner.
        greenland = (lon > -55.0) & (lon < -20.0) & (lat > 60.0)
        mask &= ~greenland
        return mask
    if name == "southern_ocean":
        # Full zonal circumnavigation; block Antarctica below ~66°S.
        mask = lat > -66.0
        # Rough cut for the Antarctic Peninsula.
        peninsula = (lon > 280.0) & (lon < 305.0) & (lat < -62.0)
        mask &= ~peninsula
        return mask
    raise ValueError(f"no synthetic mask recipe for basin {name!r}")


def _synthetic_bathymetry(
    mask: np.ndarray,
    *,
    depth_abyss: float = 4000.0,
    min_depth: float = 20.0,
) -> np.ndarray:
    """Flat-abyss bathymetry with a shelf ramp near the coast.

    Returns positive-downward depth in metres. Land cells get 0.
    """
    shelf_width_cells = 8
    dist = _bfs_distance_to_land(mask, max_dist=shelf_width_cells)
    frac = np.clip(dist / float(shelf_width_cells), 0.0, 1.0)
    depth = min_depth + (depth_abyss - min_depth) * frac
    depth = np.where(mask, depth, 0.0)
    return depth


def _bfs_distance_to_land(mask: np.ndarray, *, max_dist: int) -> np.ndarray:
    """4-connected cell-distance from each ocean cell to the nearest land cell.

    Clipped at ``max_dist`` — anything farther than that returns
    ``max_dist``. Pure numpy (no SciPy), since we only need a few
    iterations for the shelf-ramp application. Boundary handling uses
    edge-replication, so we do not pull spurious "land" across the
    domain edges (important for ``southern_ocean``, which is zonally
    periodic but we treat as non-periodic for this placeholder).
    """
    is_ocean = mask.astype(bool)
    # -1 marks "unset"; 0 is set immediately for land cells.
    dist = np.where(is_ocean, -1, 0).astype(np.int32)
    frontier = ~is_ocean  # current set of "just-reached" cells
    for step in range(1, max_dist + 1):
        padded = np.pad(frontier, 1, mode="edge")
        neighbors = (
            padded[:-2, 1:-1]
            | padded[2:, 1:-1]
            | padded[1:-1, :-2]
            | padded[1:-1, 2:]
        )
        new_cells = neighbors & (dist == -1)
        if not new_cells.any():
            break
        dist = np.where(new_cells, step, dist)
        frontier = new_cells
    dist = np.where(dist == -1, max_dist, dist)
    return dist.astype(np.float64)


def _synthetic_wind_stress(
    lat: np.ndarray,
    lat_min: float,
    lat_max: float,
    *,
    tau0: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Stommel-style zonal wind stress: ``tau_x = -tau0 cos(pi (lat-lat0)/L)``."""
    L = lat_max - lat_min
    tau_x = -tau0 * np.cos(np.pi * (lat - lat_min) / L)
    tau_y = np.zeros_like(lat)
    return tau_x, tau_y


# ----------------------------------------------------------------------
# Bundle metadata helpers
# ----------------------------------------------------------------------


def _source_attr(source: SourceKind) -> str:
    if source == "synthetic":
        return (
            "synthetic placeholder (analytical mask + flat bathymetry + "
            "Stommel wind stress); real GEBCO + ERA5 integration is tracked in #78"
        )
    return "GEBCO 2024 + ERA5 1993-2023 annual mean"


def _history_attr(
    name: BasinName, source: SourceKind, nx: int, ny: int, min_depth: float
) -> str:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        commit = "unknown"
    return (
        f"built by scripts/data/build_basin.py (commit {commit}) — "
        f"basin={name} source={source} nx={nx} ny={ny} min_depth={min_depth}"
    )


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("basin", choices=sorted(BASIN_SPECS), help="Basin name.")
    parser.add_argument(
        "--source",
        choices=["synthetic", "real"],
        default="synthetic",
        help="Source data kind (default: synthetic placeholder).",
    )
    parser.add_argument("--nx", type=int, default=None, help="Override zonal cells.")
    parser.add_argument("--ny", type=int, default=None, help="Override meridional cells.")
    parser.add_argument(
        "--min-depth",
        type=float,
        default=20.0,
        help="Shelf depth floor in metres (default 20).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output .zarr path (default: data/basin/<basin>.zarr).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        path = build_basin(
            args.basin,
            source=args.source,
            nx=args.nx,
            ny=args.ny,
            min_depth=args.min_depth,
            out_path=args.out,
        )
    except NotImplementedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
