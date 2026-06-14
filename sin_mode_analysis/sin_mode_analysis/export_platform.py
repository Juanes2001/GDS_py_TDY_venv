"""
export_platform.py — write the MODE→INTERCONNECT bridge file.

Rationale (see CONTEXTO_2, restructuring opportunity *G*): the per-ring radii,
n_eff, n_g, kappa^2 and loss — together with the sensor-ring aqueous-index sweep
arrays SWEEP_NEFF / SWEEP_NG — are *products of this FDE study* that the
INTERCONNECT circuit package consumes.  Instead of copy-pasting those numbers
between projects by hand, this module serialises them to a single JSON file that
the circuit package can load with ``platform_bridge.load_bridge()``.

The exporter is deliberately defensive: it harvests whatever the mode-analysis
run has accumulated in the shared ``state`` dict (the notebook's old kernel
namespace), falling back to the design values in ``config`` for any field the
current run did not (re)compute.  This means a partial run still yields a
complete, self-consistent bridge file.

Usage
-----
    from sin_mode_analysis import export_platform
    export_platform.export_bridge(state)            # writes data dir / platform_bridge.json
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import DATA_DIR, log

# The canonical field list the INTERCONNECT package expects.
BRIDGE_FIELDS = (
    "RING_RADIUS_M", "RING_LAMBDA_RES_M", "RING_NEFF_TE", "RING_NG_TE",
    "RING_KAPPA_INPUT_SQ", "RING_KAPPA_DROP_SQ", "RING_LOSS_DB_PER_M",
    "SWEEP_NEFF", "SWEEP_NG",
)

# Where each bridge field may live in the mode-run state, in priority order.
_ALIASES = {
    "RING_RADIUS_M":       ("RING_RADIUS_M", "ring_radius_m", "cc_radius_m"),
    "RING_LAMBDA_RES_M":   ("RING_LAMBDA_RES_M", "ring_lambda_res_m"),
    "RING_NEFF_TE":        ("RING_NEFF_TE", "cc_neff", "ring_neff_te"),
    "RING_NG_TE":          ("RING_NG_TE", "cc_ng", "ring_ng_te"),
    "RING_KAPPA_INPUT_SQ": ("RING_KAPPA_INPUT_SQ", "kappa_input_sq", "k1_sq"),
    "RING_KAPPA_DROP_SQ":  ("RING_KAPPA_DROP_SQ", "kappa_drop_sq", "k2_sq"),
    "RING_LOSS_DB_PER_M":  ("RING_LOSS_DB_PER_M", "ring_loss_db_per_m"),
    "SWEEP_NEFF":          ("SWEEP_NEFF", "ais_neff"),
    "SWEEP_NG":            ("SWEEP_NG", "ais_ng"),
}


def _harvest(state: dict, field: str):
    """Return the first present alias for `field` from state, else None."""
    for key in _ALIASES[field]:
        if key in state and state[key] is not None:
            return state[key]
    return None


def export_bridge(state: dict | None = None,
                  out_path: Path | None = None) -> Path:
    """
    Write ``platform_bridge.json`` from the mode-analysis ``state``.

    Inputs
    ------
    state : dict | None
        The shared namespace produced by the step ``run()`` calls.  Any missing
        field is reported and simply omitted (the INTERCONNECT package then keeps
        its own ``config`` default for that field).
    out_path : Path | None
        Destination JSON path (defaults to ``DATA_DIR/platform_bridge.json``).

    Outputs
    -------
    Path
        The path of the written JSON bridge file.
    """
    state = state or {}
    out_path = Path(out_path) if out_path else (DATA_DIR / "platform_bridge.json")

    payload, missing = {}, []
    for field in BRIDGE_FIELDS:
        val = _harvest(state, field)
        if val is None:
            missing.append(field)
            continue
        payload[field] = np.asarray(val, dtype=float).tolist()

    meta = {
        "generated": datetime.now().isoformat(),
        "source": "sin_mode_analysis FDE study",
        "platform": "SiN strip-waveguide 14-ring Pb/RI biosensor @ 1550 nm",
        "fields_present": sorted(payload.keys()),
        "fields_missing": missing,
        "note": ("SWEEP_NEFF/SWEEP_NG are the sensor-ring aqueous-index sweep "
                 "(n_clad 1.33->1.37) products ais_neff/ais_ng."),
    }
    out_path.write_text(json.dumps({"_meta": meta, **payload}, indent=2))
    if missing:
        log.warning(f"Bridge written with {len(missing)} field(s) missing "
                    f"(INTERCONNECT will use its config defaults): {missing}")
    log.info(f"Platform bridge → {out_path}")
    return out_path
