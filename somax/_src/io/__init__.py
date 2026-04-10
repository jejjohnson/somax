"""IO utilities for somax models.

This subpackage handles converting model states and snapshots to and from
disk-friendly representations (xarray Datasets, zarr stores). Optional
dependencies (``xarray``, ``zarr``) are imported lazily inside functions
so the somax library remains importable without them.
"""

from somax._src.io.xarray import (
    append_to_dataset,
    dataset_to_state,
    load_dataset,
    save_dataset,
    snapshots_to_dataset,
    state_to_dataset,
)


__all__ = [
    "append_to_dataset",
    "dataset_to_state",
    "load_dataset",
    "save_dataset",
    "snapshots_to_dataset",
    "state_to_dataset",
]
