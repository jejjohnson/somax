"""Public IO API for somax (zarr-backed xarray).

Re-exports from :mod:`somax._src.io`. See that subpackage for the canonical
docstrings.
"""

from somax._src.io import (
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
