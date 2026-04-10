"""xarray + zarr IO for somax model states and snapshots.

Provides round-trippable conversion between somax ``State`` instances
(equinox modules with ``Float[Array, ...]`` fields) and self-describing
xarray ``Dataset`` objects, plus zarr-backed save/load helpers.

The functions here are the canonical persistence layer for the somax
simulation runner. They are also used by the ``somax-sim`` CLI for
restart artifacts and snapshot output.

Conventions
-----------
- 1D arrays use dim ``("x",)``
- 2D arrays use dims ``("y", "x")``
- 3D arrays (multilayer) use dims ``("layer", "y", "x")``
- Time-stacked snapshots prepend a ``"time"`` dimension
- Dataset attrs always carry ``state_class`` (qualname) and
  ``state_module`` so :func:`dataset_to_state` can auto-recover the
  target class without an explicit argument.
"""

from __future__ import annotations

import dataclasses
import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np


if TYPE_CHECKING:
    import xarray as xr

    from somax._src.core.types import State


_DIM_NAMES_BY_RANK: dict[int, tuple[str, ...]] = {
    1: ("x",),
    2: ("y", "x"),
    3: ("layer", "y", "x"),
}


def _infer_dims(rank: int, *, time_axis: bool) -> tuple[str, ...]:
    """Infer canonical dim names from a leaf array rank.

    Args:
        rank: Number of axes in the leaf array (excluding time, if any).
        time_axis: Whether to prepend a ``"time"`` dimension.

    Returns:
        Tuple of dimension names. Falls back to ``("dim0", "dim1", ...)``
        for unknown ranks.
    """
    base = _DIM_NAMES_BY_RANK.get(rank)
    if base is None:
        base = tuple(f"dim{i}" for i in range(rank))
    return ("time", *base) if time_axis else base


def _state_attrs(state_class: type) -> dict[str, str]:
    """Build the standard ``state_class`` / ``state_module`` attrs."""
    return {
        "state_class": state_class.__name__,
        "state_module": state_class.__module__,
    }


def state_to_dataset(
    state: State,
    *,
    time: float | None = None,
    attrs: dict[str, Any] | None = None,
) -> xr.Dataset:
    """Convert a single model State to an xarray Dataset.

    Each field of the State PyTree becomes a Dataset variable. Dimension
    names are inferred from array rank using the canonical convention
    (1D → ``x``; 2D → ``y x``; 3D → ``layer y x``).

    Args:
        state: A somax ``State`` instance (equinox.Module subclass).
        time: Optional scalar time at which this state was sampled. If
            provided, a singleton ``time`` dimension is prepended to every
            variable and a matching coordinate is created.
        attrs: Extra attrs to merge into the Dataset (in addition to the
            automatic ``state_class`` / ``state_module`` markers).

    Returns:
        Self-describing xarray Dataset suitable for round-tripping via
        :func:`dataset_to_state`.
    """
    import xarray as xr

    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    for field in dataclasses.fields(state):
        leaf = np.asarray(getattr(state, field.name))
        rank = leaf.ndim
        dims = _infer_dims(rank, time_axis=time is not None)
        if time is not None:
            leaf = leaf[None]
        data_vars[field.name] = (dims, leaf)

    coords: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    if time is not None:
        coords["time"] = (("time",), np.asarray([time], dtype=np.float64))

    ds = xr.Dataset(data_vars, coords=coords)
    ds.attrs.update(_state_attrs(type(state)))
    if attrs:
        ds.attrs.update(attrs)
    return ds


def snapshots_to_dataset(
    snapshots: State,
    ts: jnp.ndarray | np.ndarray,
    *,
    state_class: type[State] | None = None,
    attrs: dict[str, Any] | None = None,
) -> xr.Dataset:
    """Convert a time-stacked PyTree of states to an xarray Dataset.

    ``diffrax.diffeqsolve`` returns a ``Solution`` whose ``ys`` field has
    the same PyTree structure as ``y0`` but with each leaf array stacked
    along a leading time axis. This function maps that structure into an
    xarray Dataset with a proper ``time`` coordinate.

    Args:
        snapshots: PyTree (typically ``sol.ys``) where each leaf has shape
            ``(Nt, ...)``. Must be the same class as a single State.
        ts: 1-D array of save times of length ``Nt``.
        state_class: Optional explicit State class. If omitted, the class
            of ``snapshots`` is used (this is the common case for
            ``sol.ys`` since diffrax preserves the y0 type).
        attrs: Extra attrs to merge into the Dataset.

    Returns:
        Dataset with a ``time`` coordinate and one variable per State
        field, each carrying the canonical dimension names.
    """
    import xarray as xr

    if state_class is None:
        state_class = type(snapshots)

    ts_np = np.asarray(ts)
    if ts_np.ndim != 1:
        raise ValueError(f"ts must be 1-D, got shape {ts_np.shape}")

    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    for field in dataclasses.fields(state_class):
        leaf = np.asarray(getattr(snapshots, field.name))
        if leaf.shape[0] != ts_np.shape[0]:
            raise ValueError(
                f"snapshot leaf {field.name!r} has leading dim {leaf.shape[0]} "
                f"but ts has length {ts_np.shape[0]}"
            )
        rank = leaf.ndim - 1  # subtract the time axis
        dims = _infer_dims(rank, time_axis=True)
        data_vars[field.name] = (dims, leaf)

    ds = xr.Dataset(data_vars, coords={"time": (("time",), ts_np)})
    ds.attrs.update(_state_attrs(state_class))
    if attrs:
        ds.attrs.update(attrs)
    return ds


def dataset_to_state(
    ds: xr.Dataset,
    state_class: type[State] | None = None,
    *,
    time_index: int = -1,
) -> State:
    """Reconstruct a single State from an xarray Dataset.

    The inverse of :func:`state_to_dataset` and :func:`snapshots_to_dataset`.
    If the Dataset has a ``time`` axis, ``time_index`` selects which slice
    to materialize (default: the last slice, suitable for restart from a
    final state).

    Args:
        ds: Dataset previously produced by ``state_to_dataset`` or
            ``snapshots_to_dataset``.
        state_class: Target State class. If omitted, recovered from the
            Dataset attrs (``state_class`` and ``state_module``).
        time_index: Time slice to extract when ``ds`` has a ``time``
            dimension. Defaults to ``-1``.

    Returns:
        State instance whose fields are JAX arrays matching the Dataset
        variables.
    """
    if state_class is None:
        try:
            module_name = ds.attrs["state_module"]
            class_name = ds.attrs["state_class"]
        except KeyError as exc:
            raise ValueError(
                "dataset_to_state called without state_class and the Dataset "
                "is missing 'state_class'/'state_module' attrs"
            ) from exc
        # Auto-import is allowlisted to modules under ``somax.`` so that
        # loading a zarr store with attacker-controlled attrs cannot
        # trigger arbitrary import side effects. If the persisted
        # ``state_module`` is anywhere else, the caller must pass
        # ``state_class`` explicitly.
        if not (module_name == "somax" or module_name.startswith("somax.")):
            raise ValueError(
                f"refusing to auto-import state_module {module_name!r}: only "
                f"modules under 'somax.' are allowlisted for auto-recovery. "
                f"Pass state_class explicitly to load this dataset."
            )
        module = importlib.import_module(module_name)
        state_class = getattr(module, class_name)

    kwargs = {}
    for field in dataclasses.fields(state_class):
        if field.name not in ds.variables:
            raise ValueError(
                f"Dataset is missing variable {field.name!r} required by "
                f"{state_class.__name__}"
            )
        var = ds[field.name]
        if "time" in var.dims:
            var = var.isel(time=time_index)
        kwargs[field.name] = jnp.asarray(var.values)

    return state_class(**kwargs)


def save_dataset(ds: xr.Dataset, path: str | Path, *, mode: str = "w") -> None:
    """Persist a Dataset to a zarr store.

    Uses zarr v3 by default. The store is a directory tree at ``path``.

    Args:
        ds: Dataset to write.
        path: Filesystem path for the zarr store (directory).
        mode: ``"w"`` to overwrite, ``"w-"`` to fail if exists,
            ``"a"`` to append.
    """
    path = Path(path)
    # consolidated=False because consolidated metadata is not part of the
    # zarr v3 spec; xarray emits a ZarrUserWarning otherwise.
    ds.to_zarr(path, mode=mode, zarr_format=3, consolidated=False)


def load_dataset(path: str | Path) -> xr.Dataset:
    """Open a zarr store as an xarray Dataset.

    Returns a Dataset backed by the on-disk store. Variables are read
    lazily by xarray on first access (no extra chunking framework
    required). Pass the result through ``.load()`` to materialize, or
    use dask-aware tooling if you need parallel chunking.

    Args:
        path: Filesystem path to a zarr store written by
            :func:`save_dataset`.
    """
    import xarray as xr

    return xr.open_zarr(Path(path), consolidated=False)


def append_to_dataset(
    ds: xr.Dataset,
    path: str | Path,
    *,
    append_dim: str = "time",
) -> None:
    """Append a Dataset to an existing zarr store along a dimension.

    Used by chunked-integration workflows (see GitHub issue #70) where
    snapshots are written incrementally rather than all at once.

    Args:
        ds: Dataset to append. Must share dim ordering and non-append
            shapes with the existing store.
        path: Existing zarr store written by :func:`save_dataset`.
        append_dim: Dimension to append along. Defaults to ``"time"``.
    """
    ds.to_zarr(Path(path), mode="a", append_dim=append_dim, consolidated=False)
