"""IO utilities for somax models.

This subpackage handles converting model states and snapshots to and from
disk-friendly representations (xarray Datasets, zarr stores). Optional
dependencies (``xarray``, ``zarr``) are imported lazily inside functions
so the somax library remains importable without them.
"""

from somax._src.io.xarray import (
    append_to_dataset,
    apply_scale_metadata,
    dataset_to_state,
    load_dataset,
    save_dataset,
    scales_attrs,
    snapshots_to_dataset,
    state_to_dataset,
    transform_attrs,
)


__all__ = [
    "append_to_dataset",
    "apply_scale_metadata",
    "dataset_to_state",
    "load_dataset",
    "save_dataset",
    "scales_attrs",
    "snapshots_to_dataset",
    "state_to_dataset",
    "transform_attrs",
]
