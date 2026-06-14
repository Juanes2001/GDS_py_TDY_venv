"""
platform_bridge.py — load the MODE→INTERCONNECT bridge file.

This is the consuming half of the explicit bridge described in CONTEXTO_2
(opportunity *G*).  The FDE mode-analysis package writes a ``platform_bridge.json``
holding the per-ring radii / n_eff / n_g / kappa^2 / loss and the sensor-ring
sweep arrays (SWEEP_NEFF / SWEEP_NG).  This module reads that file and returns the
arrays as a dict so the circuit package can be driven by *freshly simulated*
parameters instead of the hard-coded study values in :mod:`config`.

Design choice
-------------
:mod:`config` ships with the original study's values so the package runs
**standalone** out of the box.  Loading a bridge is therefore *optional*:
``main.py`` calls :func:`apply_bridge` only when a bridge file is present, and
otherwise proceeds with the config defaults.  This keeps the package runnable on
its own while still honouring the "single source of truth" recommendation when
both projects are used together.

Usage
-----
    from sin_interconnect_cascade import platform_bridge, config
    bridge = platform_bridge.load_bridge("path/to/platform_bridge.json")
    platform_bridge.apply_bridge(bridge)   # overrides config arrays in place
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config
from .config import log

BRIDGE_FIELDS = (
    "RING_RADIUS_M", "RING_LAMBDA_RES_M", "RING_NEFF_TE", "RING_NG_TE",
    "RING_KAPPA_INPUT_SQ", "RING_KAPPA_DROP_SQ", "RING_LOSS_DB_PER_M",
    "SWEEP_NEFF", "SWEEP_NG",
)


def default_bridge_path() -> Path:
    """The conventional location of the FDE-exported bridge file."""
    return config.DATA_DIR / "platform_bridge.json"


def load_bridge(path: str | Path | None = None) -> dict | None:
    """
    Read a ``platform_bridge.json`` file into a dict of numpy arrays.

    Returns ``None`` (and logs a notice) if the file does not exist, so the
    caller can fall back to the config defaults without error handling.
    """
    path = Path(path) if path else default_bridge_path()
    if not path.exists():
        log.info(f"No platform bridge at {path}; using config defaults.")
        return None
    raw = json.loads(path.read_text())
    bridge = {k: np.asarray(v, dtype=float)
              for k, v in raw.items() if k in BRIDGE_FIELDS}
    log.info(f"Loaded platform bridge ({sorted(bridge)}) from {path}")
    return bridge


def apply_bridge(bridge: dict | None) -> None:
    """
    Overwrite the corresponding :mod:`config` arrays with bridge values.

    Only fields actually present in the bridge are overridden; everything else
    keeps its config default.  Mutating ``config`` in place means every module
    that did ``from .config import *`` already holds references that are updated
    *before* the sweeps run (``main.py`` applies the bridge first).
    """
    if not bridge:
        return
    for field in BRIDGE_FIELDS:
        if field in bridge:
            setattr(config, field, bridge[field])
    log.info(f"Applied bridge overrides: {sorted(bridge)}")
