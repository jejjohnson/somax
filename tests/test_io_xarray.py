"""Tests for the xarray + zarr IO layer."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest


pytest.importorskip("xarray")
pytest.importorskip("zarr")

from somax import io
from somax._src.models.qg.baroclinic import BaroclinicQGState
from somax._src.models.qg.barotropic import BarotropicQGState
from somax._src.models.swm.linear_2d import LinearSW2DState
from somax._src.models.swm.nonlinear_2d import NonlinearSW2DState


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _make_swm_state(ny: int = 8, nx: int = 12) -> NonlinearSW2DState:
    return NonlinearSW2DState(
        h=jnp.ones((ny, nx)) * 100.0,
        u=jnp.zeros((ny, nx)),
        v=jnp.zeros((ny, nx)),
    )


def _make_linear_swm_state(ny: int = 8, nx: int = 12) -> LinearSW2DState:
    return LinearSW2DState(
        h=jnp.ones((ny, nx)) * 100.0,
        u=jnp.zeros((ny, nx)),
        v=jnp.zeros((ny, nx)),
    )


def _make_bc_qg_state(nl: int = 3, ny: int = 8, nx: int = 12) -> BaroclinicQGState:
    return BaroclinicQGState(q=jnp.zeros((nl, ny, nx)))


def _make_bt_qg_state(ny: int = 8, nx: int = 12) -> BarotropicQGState:
    return BarotropicQGState(q=jnp.zeros((ny, nx)))


# ----------------------------------------------------------------------
# state_to_dataset / dataset_to_state — single state
# ----------------------------------------------------------------------


class TestSingleState:
    def test_swm_2d_round_trip_in_memory(self) -> None:
        state = _make_swm_state()
        ds = io.state_to_dataset(state)
        assert set(ds.data_vars) == {"h", "u", "v"}
        assert ds["h"].dims == ("y", "x")
        assert ds["h"].shape == (8, 12)
        assert ds.attrs["state_class"] == "NonlinearSW2DState"

        recovered = io.dataset_to_state(ds, NonlinearSW2DState)
        np.testing.assert_array_equal(np.asarray(recovered.h), np.asarray(state.h))
        np.testing.assert_array_equal(np.asarray(recovered.u), np.asarray(state.u))
        np.testing.assert_array_equal(np.asarray(recovered.v), np.asarray(state.v))

    def test_bc_qg_3d_round_trip_in_memory(self) -> None:
        state = _make_bc_qg_state(nl=3, ny=6, nx=8)
        ds = io.state_to_dataset(state)
        assert ds["q"].dims == ("layer", "y", "x")
        assert ds["q"].shape == (3, 6, 8)

        recovered = io.dataset_to_state(ds, BaroclinicQGState)
        np.testing.assert_array_equal(np.asarray(recovered.q), np.asarray(state.q))

    def test_bt_qg_2d_round_trip(self) -> None:
        state = _make_bt_qg_state()
        ds = io.state_to_dataset(state)
        assert ds["q"].dims == ("y", "x")
        recovered = io.dataset_to_state(ds, BarotropicQGState)
        np.testing.assert_array_equal(np.asarray(recovered.q), np.asarray(state.q))

    def test_with_explicit_time_adds_singleton_axis(self) -> None:
        state = _make_swm_state()
        ds = io.state_to_dataset(state, time=42.0)
        assert ds["h"].dims == ("time", "y", "x")
        assert ds["h"].shape == (1, 8, 12)
        assert "time" in ds.coords
        np.testing.assert_array_equal(ds["time"].values, np.array([42.0]))

    def test_extra_attrs_are_merged(self) -> None:
        state = _make_swm_state()
        ds = io.state_to_dataset(
            state, attrs={"experiment": "smoke", "config_hash": "abc"}
        )
        assert ds.attrs["experiment"] == "smoke"
        assert ds.attrs["config_hash"] == "abc"
        assert ds.attrs["state_class"] == "NonlinearSW2DState"

    def test_dataset_to_state_auto_recovers_class(self) -> None:
        """Without an explicit state_class arg, attrs should drive reconstruction."""
        state = _make_swm_state()
        ds = io.state_to_dataset(state)
        recovered = io.dataset_to_state(ds)
        assert isinstance(recovered, NonlinearSW2DState)
        np.testing.assert_array_equal(np.asarray(recovered.h), np.asarray(state.h))

    def test_dataset_to_state_missing_attrs_raises(self) -> None:
        import xarray as xr

        ds = xr.Dataset({"h": (("y", "x"), np.zeros((4, 4)))})
        # No state_class in attrs and no explicit class → should fail loudly.
        with pytest.raises(ValueError, match="state_class"):
            io.dataset_to_state(ds)

    def test_dataset_to_state_missing_variable_raises(self) -> None:
        import xarray as xr

        ds = xr.Dataset({"h": (("y", "x"), np.zeros((4, 4)))})
        with pytest.raises(ValueError, match="missing variable 'u'"):
            io.dataset_to_state(ds, NonlinearSW2DState)


# ----------------------------------------------------------------------
# snapshots_to_dataset — time-stacked
# ----------------------------------------------------------------------


class TestSnapshots:
    def test_swm_2d_stacked_snapshots(self) -> None:
        nt, ny, nx = 5, 8, 12
        snapshots = NonlinearSW2DState(
            h=jnp.zeros((nt, ny, nx)),
            u=jnp.zeros((nt, ny, nx)),
            v=jnp.zeros((nt, ny, nx)),
        )
        ts = jnp.linspace(0.0, 4.0, nt)
        ds = io.snapshots_to_dataset(snapshots, ts)

        assert ds["h"].dims == ("time", "y", "x")
        assert ds["h"].shape == (nt, ny, nx)
        np.testing.assert_array_equal(ds["time"].values, np.asarray(ts))
        assert ds.attrs["state_class"] == "NonlinearSW2DState"

    def test_bc_qg_stacked_snapshots_3d_per_step(self) -> None:
        nt, nl, ny, nx = 4, 2, 6, 8
        snapshots = BaroclinicQGState(q=jnp.zeros((nt, nl, ny, nx)))
        ts = jnp.linspace(0.0, 3.0, nt)
        ds = io.snapshots_to_dataset(snapshots, ts)
        assert ds["q"].dims == ("time", "layer", "y", "x")
        assert ds["q"].shape == (nt, nl, ny, nx)

    def test_snapshots_round_trip_via_last_time_step(self) -> None:
        """Common restart pattern: snapshots → final state."""
        nt, ny, nx = 5, 8, 12
        h_stack = jnp.arange(nt * ny * nx, dtype=jnp.float32).reshape((nt, ny, nx))
        snapshots = NonlinearSW2DState(
            h=h_stack,
            u=jnp.zeros((nt, ny, nx)),
            v=jnp.zeros((nt, ny, nx)),
        )
        ts = jnp.linspace(0.0, 4.0, nt)
        ds = io.snapshots_to_dataset(snapshots, ts)

        # Default time_index=-1 picks the last snapshot.
        final = io.dataset_to_state(ds, NonlinearSW2DState)
        np.testing.assert_array_equal(np.asarray(final.h), np.asarray(h_stack[-1]))

    def test_snapshots_ts_length_mismatch_raises(self) -> None:
        snapshots = NonlinearSW2DState(
            h=jnp.zeros((5, 4, 4)),
            u=jnp.zeros((5, 4, 4)),
            v=jnp.zeros((5, 4, 4)),
        )
        with pytest.raises(ValueError, match="leading dim 5 but ts has length 3"):
            io.snapshots_to_dataset(snapshots, jnp.array([0.0, 1.0, 2.0]))

    def test_snapshots_ts_must_be_1d(self) -> None:
        snapshots = NonlinearSW2DState(
            h=jnp.zeros((3, 4, 4)),
            u=jnp.zeros((3, 4, 4)),
            v=jnp.zeros((3, 4, 4)),
        )
        with pytest.raises(ValueError, match="ts must be 1-D"):
            io.snapshots_to_dataset(snapshots, jnp.zeros((3, 1)))


# ----------------------------------------------------------------------
# zarr persistence — round trip via disk
# ----------------------------------------------------------------------


class TestZarrPersistence:
    def test_swm_state_save_load_round_trip(self, tmp_path: Path) -> None:
        state = _make_swm_state()
        ds = io.state_to_dataset(state, attrs={"experiment": "test"})

        store = tmp_path / "state.zarr"
        io.save_dataset(ds, store)
        assert store.is_dir()

        loaded = io.load_dataset(store)
        recovered = io.dataset_to_state(loaded)
        assert isinstance(recovered, NonlinearSW2DState)
        np.testing.assert_array_equal(np.asarray(recovered.h), np.asarray(state.h))
        # Attrs survive zarr round trip.
        assert loaded.attrs["experiment"] == "test"

    def test_bc_qg_snapshots_save_load_round_trip(self, tmp_path: Path) -> None:
        nt, nl, ny, nx = 3, 2, 6, 8
        snapshots = BaroclinicQGState(
            q=jnp.arange(nt * nl * ny * nx, dtype=jnp.float32).reshape((nt, nl, ny, nx))
        )
        ts = jnp.linspace(0.0, 2.0, nt)
        ds = io.snapshots_to_dataset(snapshots, ts)

        store = tmp_path / "snapshots.zarr"
        io.save_dataset(ds, store)

        loaded = io.load_dataset(store)
        np.testing.assert_array_equal(loaded["q"].values, np.asarray(snapshots.q))
        np.testing.assert_array_equal(loaded["time"].values, np.asarray(ts))

    def test_append_extends_time_axis(self, tmp_path: Path) -> None:
        nt1, ny, nx = 3, 4, 4
        first = NonlinearSW2DState(
            h=jnp.ones((nt1, ny, nx)),
            u=jnp.zeros((nt1, ny, nx)),
            v=jnp.zeros((nt1, ny, nx)),
        )
        ds1 = io.snapshots_to_dataset(first, jnp.arange(nt1, dtype=jnp.float32))
        store = tmp_path / "appendable.zarr"
        io.save_dataset(ds1, store)

        nt2 = 2
        second = NonlinearSW2DState(
            h=jnp.full((nt2, ny, nx), 2.0),
            u=jnp.zeros((nt2, ny, nx)),
            v=jnp.zeros((nt2, ny, nx)),
        )
        ds2 = io.snapshots_to_dataset(
            second, jnp.arange(nt1, nt1 + nt2, dtype=jnp.float32)
        )
        io.append_to_dataset(ds2, store, append_dim="time")

        loaded = io.load_dataset(store)
        assert loaded["h"].shape == (nt1 + nt2, ny, nx)
        np.testing.assert_array_equal(
            loaded["time"].values, np.arange(nt1 + nt2, dtype=np.float32)
        )

    def test_linear_swm_state_round_trip(self, tmp_path: Path) -> None:
        state = _make_linear_swm_state(ny=6, nx=10)
        ds = io.state_to_dataset(state)
        store = tmp_path / "linear_swm.zarr"
        io.save_dataset(ds, store)

        loaded = io.load_dataset(store)
        recovered = io.dataset_to_state(loaded)
        assert isinstance(recovered, LinearSW2DState)
        np.testing.assert_array_equal(np.asarray(recovered.h), np.asarray(state.h))


# ----------------------------------------------------------------------
# Allowlisted auto-import — Copilot review on PR #71 flagged that the
# attr-driven importlib.import_module call is a sharp edge for untrusted
# zarr stores. The loader now allowlists modules under ``somax.``.
# ----------------------------------------------------------------------


class TestAutoImportAllowlist:
    def _make_dataset_with_module(self, tmp_path: Path, module_name: str):
        import xarray as xr

        ds = xr.Dataset(
            {"h": (("y", "x"), np.zeros((4, 4)))},
            attrs={"state_class": "Whatever", "state_module": module_name},
        )
        store = tmp_path / "tampered.zarr"
        io.save_dataset(ds, store)
        return io.load_dataset(store)

    def test_non_somax_module_rejected(self, tmp_path: Path) -> None:
        loaded = self._make_dataset_with_module(tmp_path, "os")
        with pytest.raises(ValueError, match=r"refusing to auto-import"):
            io.dataset_to_state(loaded)

    def test_somax_prefix_required_not_substring(self, tmp_path: Path) -> None:
        # 'somaxxx' starts with 'somax' as a substring but is not under
        # the somax namespace; the prefix check uses 'somax.' explicitly.
        loaded = self._make_dataset_with_module(tmp_path, "somaxxx.evil")
        with pytest.raises(ValueError, match=r"refusing to auto-import"):
            io.dataset_to_state(loaded)

    def test_explicit_state_class_bypasses_allowlist(self, tmp_path: Path) -> None:
        # If the caller passes state_class explicitly, the auto-import
        # path is skipped entirely — no allowlist check is needed.
        loaded = self._make_dataset_with_module(tmp_path, "os")
        # NonlinearSW2DState has fields h, u, v — h is present, u/v are
        # missing, so we expect a "missing variable" error from the
        # downstream field walk, NOT the allowlist refusal.
        with pytest.raises(ValueError, match=r"missing variable"):
            io.dataset_to_state(loaded, state_class=NonlinearSW2DState)
