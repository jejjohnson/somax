"""Tests for the real-basin data CLI helpers (#74 + #75).

Covers:
  - ``scripts/data/build_basin.py`` synthetic mode: schema, mask
    coverage, CF attrs, error paths.
  - ``somax._src.cli._data`` URL validators + remote-name discovery.

No tests actually shell out to ``dvc``. The heavyweight integration
test is ``dvc repro`` itself, which runs in CI / by the developer on
first push.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr


# Make the repo root importable so ``from scripts.data.build_basin
# import ...`` works regardless of where pytest is invoked.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from scripts.data.build_basin import (
    BASIN_SPECS,
    SOMAX_BASIN_VERSION,
    build_basin,
)
from somax._src.cli._data import (
    BASIN_NAMES,
    REMOTE_NAME,
    _basin_zarr_path,
    _validate_url_for_backend,
)


# ----------------------------------------------------------------------
# build_basin.py — synthetic mode
# ----------------------------------------------------------------------


class TestBuildBasinSynthetic:
    @pytest.mark.parametrize("name", BASIN_NAMES)
    def test_each_basin_writes_valid_zarr(self, name, tmp_path):
        out = tmp_path / f"{name}.zarr"
        result = build_basin(name, source="synthetic", nx=16, ny=8, out_path=out)
        assert result == out.resolve()
        ds = xr.open_zarr(out, consolidated=False)
        # Schema: required variables.
        for var in ("bathymetry", "mask", "tau_x", "tau_y", "heat"):
            assert var in ds.data_vars, f"{name}: missing required variable {var!r}"
        for coord in ("lon", "lat"):
            assert coord in ds.coords, f"{name}: missing required coord {coord!r}"
        # Shape: (y=ny, x=nx).
        assert ds.mask.shape == (8, 16)
        # Attrs the runtime will consume.
        assert ds.attrs["basin_name"] == name
        assert ds.attrs["somax_basin_version"] == SOMAX_BASIN_VERSION
        assert ds.attrs["geometry_kind"] in ("real_basin", "spherical_cap")

    @pytest.mark.parametrize("name", BASIN_NAMES)
    def test_mask_has_both_ocean_and_land(self, name, tmp_path):
        """A plausible placeholder basin should have *some* ocean and *some* land.

        Otherwise the synthetic mask is either the whole domain (no mask
        at all) or empty (no ocean to simulate) — either way useless for
        mask-aware model paths.
        """
        out = tmp_path / f"{name}.zarr"
        build_basin(name, source="synthetic", nx=32, ny=16, out_path=out)
        ds = xr.open_zarr(out, consolidated=False)
        mask = np.asarray(ds.mask)
        ocean = int(mask.sum())
        total = mask.size
        assert 0 < ocean < total, (
            f"{name}: mask must have both ocean and land; "
            f"got {ocean}/{total} ocean cells"
        )

    def test_bathymetry_is_nonnegative_and_zero_on_land(self, tmp_path):
        out = tmp_path / "med_sea.zarr"
        build_basin("med_sea", source="synthetic", nx=32, ny=16, out_path=out)
        ds = xr.open_zarr(out, consolidated=False)
        bathy = np.asarray(ds.bathymetry)
        mask = np.asarray(ds.mask).astype(bool)
        assert (bathy >= 0).all(), (
            "depth must be positive downward; saw negative values"
        )
        # Land cells carry zero depth so downstream code doesn't silently
        # interpret land as ocean-with-finite-depth.
        assert (bathy[~mask] == 0).all(), (
            "land cells must have zero depth (else downstream will treat them as ocean)"
        )

    def test_min_depth_floor_is_respected_on_ocean_cells(self, tmp_path):
        out = tmp_path / "med_sea.zarr"
        build_basin(
            "med_sea", source="synthetic", nx=32, ny=16, min_depth=50.0, out_path=out
        )
        ds = xr.open_zarr(out, consolidated=False)
        bathy = np.asarray(ds.bathymetry)
        mask = np.asarray(ds.mask).astype(bool)
        # Ocean cells should all be at least min_depth.
        assert bathy[mask].min() >= 50.0, (
            f"min_depth=50 floor not respected on ocean cells; "
            f"got min depth {bathy[mask].min()}"
        )

    def test_unknown_basin_raises(self, tmp_path):
        with pytest.raises(ValueError, match="unknown basin"):
            build_basin("no_such_basin", out_path=tmp_path / "x.zarr")  # type: ignore[arg-type]

    def test_bad_grid_shape_raises(self, tmp_path):
        with pytest.raises(ValueError, match=r"nx, ny must be > 0"):
            build_basin("med_sea", nx=0, ny=8, out_path=tmp_path / "x.zarr")

    def test_real_source_raises_notimplemented(self, tmp_path):
        """Real-data build (GEBCO + ERA5) is tracked in #78.

        Guard against accidentally flipping it on before the loader
        lands — the error message must point at the tracking issue.
        """
        with pytest.raises(NotImplementedError, match="#78"):
            build_basin("med_sea", source="real", out_path=tmp_path / "x.zarr")

    def test_overwrites_existing_zarr(self, tmp_path):
        """Running the build twice must not trip on leftover zarr files."""
        out = tmp_path / "med_sea.zarr"
        build_basin("med_sea", source="synthetic", nx=16, ny=8, out_path=out)
        # Second call should not raise.
        build_basin("med_sea", source="synthetic", nx=16, ny=8, out_path=out)
        assert out.exists()


# ----------------------------------------------------------------------
# BASIN_SPECS integrity
# ----------------------------------------------------------------------


class TestBasinSpecs:
    def test_spec_names_match_cli_basin_names(self):
        assert set(BASIN_SPECS) == set(BASIN_NAMES)

    @pytest.mark.parametrize("name", BASIN_NAMES)
    def test_cartesian_specs_have_Lx_Ly_and_ref_lat(self, name):
        spec = BASIN_SPECS[name]
        if spec.geometry_kind == "real_basin":
            assert spec.Lx is not None, f"{name}: real_basin must have Lx"
            assert spec.Ly is not None, f"{name}: real_basin must have Ly"
            assert spec.ref_lat is not None, f"{name}: real_basin must have ref_lat"
        else:  # spherical_cap
            assert spec.Lx is None
            assert spec.Ly is None
            assert spec.ref_lat is None

    @pytest.mark.parametrize("name", BASIN_NAMES)
    def test_lon_lat_bounds_are_ordered(self, name):
        spec = BASIN_SPECS[name]
        lo, hi = spec.lon_bounds
        assert lo < hi, f"{name}: lon_bounds must be (min, max); got {spec.lon_bounds}"
        lo, hi = spec.lat_bounds
        assert lo < hi, f"{name}: lat_bounds must be (min, max); got {spec.lat_bounds}"


# ----------------------------------------------------------------------
# _data.py — URL validators + path helpers
# ----------------------------------------------------------------------


class TestValidateUrlForBackend:
    @pytest.mark.parametrize(
        "backend,url",
        [
            ("gdrive", "gdrive://1a2b3c"),
            ("s3", "s3://my-bucket/somax"),
            ("azure", "azure://container/somax"),
            ("gcs", "gs://my-bucket/somax"),
            ("local", "/data/somax"),
        ],
    )
    def test_well_formed_urls_pass(self, backend, url):
        _validate_url_for_backend(backend, url)  # no exception

    @pytest.mark.parametrize(
        "backend,url,match",
        [
            ("gdrive", "https://drive.google.com/...", "gdrive://"),
            ("s3", "my-bucket/somax", "s3://"),
            ("azure", "https://example.com/", "azure://"),
            ("gcs", "bucket/somax", "gs://"),
            ("local", "relative/path", "absolute path"),
        ],
    )
    def test_malformed_urls_raise(self, backend, url, match):
        with pytest.raises(SystemExit, match=match):
            _validate_url_for_backend(backend, url)


class TestBasinZarrPath:
    @pytest.mark.parametrize("name", BASIN_NAMES)
    def test_path_matches_dvc_stage_output(self, name):
        """The path the CLI computes must match the stage's ``outs:`` entry.

        If the helpers look at ``data/basin/<name>.zarr`` but DVC writes
        to ``data/basins/<name>.zarr`` (or similar), ``fetch`` and
        ``build --push`` would silently reference different paths. Lock
        it down here.
        """
        assert _basin_zarr_path(name) == Path("data") / "basin" / f"{name}.zarr"


class TestRemoteNameIsStable:
    def test_remote_name_constant(self):
        """The remote name is baked into user-facing docs (the design
        note + subcommand help). Don't rename it without updating both."""
        assert REMOTE_NAME == "somax-data"
