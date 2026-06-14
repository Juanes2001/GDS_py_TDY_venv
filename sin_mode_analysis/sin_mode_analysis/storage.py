"""
storage.py — Shared HDF5 cache / resume utilities.

Every sweep engine in this package follows the same fault-tolerant pattern,
inherited unchanged from the original notebook:

    1. If the HDF5 file exists, load it and read the boolean `flags/computed`
       dataset.  Any point with computed[i] == True is skipped.
    2. After computing each point, write it to disk and `flush()` immediately,
       so an interrupted run can be resumed exactly where it stopped.

The per-engine `_init_hdf5` builders live next to their engines because their
*dataset schemas differ* (width sweep vs ring radius sweep vs coupler gap
sweep).  This module provides the generic glue used by all of them.

Units: arrays are stored in SI or in the unit named by the dataset; never
silently mixed.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

from .config import log


def cache_exists(path: Path) -> bool:
    """True if an HDF5 cache file already exists on disk."""
    return Path(path).exists()


def write_metadata_attrs(group: h5py.Group, attrs: dict) -> None:
    """Attach a dict of scalar attributes to an HDF5 group, plus a start stamp."""
    for k, v in attrs.items():
        group.attrs[k] = v
    group.attrs["timestamp_start"] = datetime.now().isoformat()


def stamp_end(hf: h5py.File, n_computed: int) -> None:
    """Record an end timestamp and the number of completed points."""
    hf["metadata"].attrs["timestamp_end"] = datetime.now().isoformat()
    hf["metadata"].attrs["runs_completed"] = int(n_computed)


def report_cache(computed: np.ndarray, total: int) -> int:
    """Log how many points are cached / remaining; return remaining count."""
    n_cached = int(np.asarray(computed).sum())
    remaining = total - n_cached
    log.info(f"Cached: {n_cached}/{total}  |  Remaining: {remaining}")
    return remaining
