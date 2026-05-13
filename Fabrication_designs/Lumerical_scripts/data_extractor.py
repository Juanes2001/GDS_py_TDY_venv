"""
data_extractor.py — Extract all 8 ONA ports from the 7-ring cascade
====================================================================
Extracts one transmission spectrum per ONA input port and packages
everything into a ResultBundle for saving and plotting.

Port mapping (from config.ONA_PORT_LABELS):
  ONA input 1  → RING_1 through
  ONA input 2  → RING_1 drop  (= RING_2 input signal)
  ONA input 3  → RING_2 drop
  ONA input 4  → RING_3 drop
  ONA input 5  → RING_4 drop
  ONA input 6  → RING_5 drop
  ONA input 7  → RING_6 drop
  ONA input 8  → RING_7 through / drop
"""

import json
import logging
from dataclasses import dataclass, field
from datetime    import datetime
from pathlib     import Path
from typing      import Any, Dict, List, Optional

import numpy  as np
import pandas as pd
import h5py

import config

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# ResultBundle — structured container for one simulation run
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ResultBundle:
    """
    All extracted data for one simulation run.

    Attributes
    ----------
    run_id       : unique identifier string
    params       : dict of sweep parameters for this run
    frequency    : 1-D array, Hz
    wavelength   : 1-D array, nm
    spectra      : dict { port_label → complex 1-D array }
                   One entry per ONA input port.
    metadata     : arbitrary extra info (timestamp, config snapshot, …)
    """
    run_id     : str
    params     : Dict[str, Any]                    = field(default_factory=dict)
    frequency  : Optional[np.ndarray]              = None
    wavelength : Optional[np.ndarray]              = None
    spectra    : Optional[Dict[str, np.ndarray]]   = None
    metadata   : Dict[str, Any]                    = field(default_factory=dict)

    def __post_init__(self):
        """Derive wavelength (nm) from frequency if not supplied."""
        if self.frequency is not None and self.wavelength is None:
            self.wavelength = (config.C_LIGHT / self.frequency) * 1e9  # → nm

    def transmission_db(self, port_label: str) -> np.ndarray:
        """Return |T|² in dB for a given port label."""
        T = self.spectra[port_label]
        return 10 * np.log10(np.abs(T) ** 2 + 1e-30)

    def all_db(self) -> Dict[str, np.ndarray]:
        """Return {port_label → dB array} for all ports."""
        return {k: self.transmission_db(k) for k in self.spectra}


# ══════════════════════════════════════════════════════════════════════════════
# Core extraction
# ══════════════════════════════════════════════════════════════════════════════

def _result_path(port_index: int) -> str:
    """
    Build the result path string for ONA input port N.

    INTERCONNECT result browser path format:
        "input N/mode 1/transmission"

    ← FILL THIS IN / VERIFY: open the result browser after a GUI run and
      confirm the exact path string for your version.
    """
    return f"input {port_index}/mode 1/transmission"


def extract_one_port(
    ic,
    ona_name   : str,
    port_index : int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract frequency and complex transmission from one ONA input port.

    Parameters
    ----------
    ic         : active lumapi.INTERCONNECT session
    ona_name   : ONA element name  (e.g. "ONA_1")
    port_index : 1-based port number  (1 … 8)

    Returns
    -------
    frequency    : np.ndarray, Hz
    transmission : np.ndarray, complex
    """
    result_path = _result_path(port_index)
    try:
        result = ic.getresult(ona_name, result_path)
        freq   = np.asarray(result["f"]).ravel()
        T      = np.asarray(result["T"]).ravel()
        log.debug(f"Extracted ONA port {port_index}: {len(freq)} points")
        return freq, T
    except Exception as e:
        log.error(f"Failed to extract ONA port {port_index} ({result_path}): {e}")
        raise


def extract_all_ports(
    ic,
    ona_name    : str  = "ONA_1",
    num_ports   : int  = None,
    port_labels : List[str] = None,
) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Extract all ONA input ports and return a labelled dict of spectra.

    Parameters
    ----------
    ic          : active session (simulation must have run)
    ona_name    : ONA element name
    num_ports   : number of ports to extract (default: config.ONA_NUM_INPUT_PORTS)
    port_labels : human-readable labels; defaults to config.ONA_PORT_LABELS

    Returns
    -------
    frequency : np.ndarray, Hz  (same for all ports)
    spectra   : dict { label → complex np.ndarray }
    """
    num_ports   = num_ports   or config.ONA_NUM_INPUT_PORTS
    port_labels = port_labels or config.ONA_PORT_LABELS

    if len(port_labels) < num_ports:
        # Auto-generate labels for any unlabelled ports
        extra = [f"port {i}" for i in range(len(port_labels) + 1, num_ports + 1)]
        port_labels = list(port_labels) + extra

    spectra  = {}
    freq_ref = None

    for idx in range(1, num_ports + 1):
        label = port_labels[idx - 1]
        try:
            freq, T = extract_one_port(ic, ona_name, idx)
            spectra[label] = T
            if freq_ref is None:
                freq_ref = freq
        except Exception:
            log.warning(f"Skipping port {idx} due to extraction error.")
            spectra[label] = np.full_like(freq_ref, np.nan) if freq_ref is not None else None

    log.info(f"Extracted {len(spectra)} ports from {ona_name}")
    return freq_ref, spectra


# ══════════════════════════════════════════════════════════════════════════════
# High-level: build ResultBundle from a completed run
# ══════════════════════════════════════════════════════════════════════════════

def build_result_bundle(
    ic,
    run_id    : str,
    params    : dict,
    ona_name  : str  = "ONA_1",
) -> ResultBundle:
    """
    Build a complete ResultBundle by extracting all 8 ONA ports.

    Parameters
    ----------
    ic       : active session (after ic.run())
    run_id   : identifier for this simulation run
    params   : the parameter dict used for this run
    ona_name : ONA element name

    Returns
    -------
    ResultBundle
    """
    freq, spectra = extract_all_ports(ic, ona_name=ona_name)

    bundle = ResultBundle(
        run_id   = run_id,
        params   = params,
        frequency= freq,
        spectra  = spectra,
        metadata = {
            "timestamp"         : datetime.now().isoformat(),
            "lumerical_version" : config.LUMERICAL_VERSION,
            "ona_name"          : ona_name,
            "num_ports"         : config.ONA_NUM_INPUT_PORTS,
            "ring_lengths_um"   : config.RING_LENGTHS_UM,
        },
    )
    log.info(f"ResultBundle built: {run_id}  ({len(spectra)} spectra)")
    return bundle


# ══════════════════════════════════════════════════════════════════════════════
# Save — HDF5
# ══════════════════════════════════════════════════════════════════════════════

def save_hdf5(bundle: ResultBundle, filepath: Path | str) -> None:
    """
    Save a ResultBundle to an HDF5 file.

    Layout:
        /<run_id>/
            frequency          (N,) float64  Hz
            wavelength         (N,) float64  nm
            spectra/
                RING_1_through  (N,) complex128
                RING_1_drop     (N,) complex128
                ...
            attrs: params, metadata as JSON
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    copt = dict(compression="gzip", compression_opts=config.HDF5_COMPRESSION_LEVEL)

    with h5py.File(filepath, "a") as f:
        # Remove existing group if re-running the same run_id
        if bundle.run_id in f:
            del f[bundle.run_id]

        grp = f.create_group(bundle.run_id)

        if bundle.frequency  is not None:
            grp.create_dataset("frequency",  data=bundle.frequency,  **copt)
        if bundle.wavelength is not None:
            grp.create_dataset("wavelength", data=bundle.wavelength, **copt)

        if bundle.spectra:
            sgrp = grp.create_group("spectra")
            for label, arr in bundle.spectra.items():
                if arr is not None:
                    safe = label.replace(" ", "_").replace("—", "").replace("/", "_").strip("_")
                    sgrp.create_dataset(safe, data=arr, **copt)

        grp.attrs["params"]   = json.dumps(bundle.params,   default=str)
        grp.attrs["metadata"] = json.dumps(bundle.metadata, default=str)

    log.info(f"Saved HDF5: {bundle.run_id} → {filepath}")


def load_hdf5(filepath: Path | str, run_id: str) -> ResultBundle:
    """Load a ResultBundle from an HDF5 file."""
    filepath = Path(filepath)
    with h5py.File(filepath, "r") as f:
        grp    = f[run_id]
        params = json.loads(grp.attrs.get("params",   "{}"))
        meta   = json.loads(grp.attrs.get("metadata", "{}"))

        freq = np.array(grp["frequency"])  if "frequency"  in grp else None
        wav  = np.array(grp["wavelength"]) if "wavelength" in grp else None

        spectra = {}
        if "spectra" in grp:
            for k in grp["spectra"]:
                spectra[k] = np.array(grp["spectra"][k])

    return ResultBundle(
        run_id   = run_id,
        params   = params,
        frequency= freq,
        wavelength=wav,
        spectra  = spectra or None,
        metadata = meta,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Save — CSV
# ══════════════════════════════════════════════════════════════════════════════

def save_csv(bundle: ResultBundle, directory: Path | str) -> Path:
    """
    Save spectra as a single CSV file.
    Columns: frequency_Hz, wavelength_nm, <port>_re, <port>_im, <port>_dB, …
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    filepath  = directory / f"{bundle.run_id}.csv"

    data = {}
    if bundle.frequency  is not None: data["frequency_Hz"]  = bundle.frequency
    if bundle.wavelength is not None: data["wavelength_nm"] = bundle.wavelength

    if bundle.spectra:
        for label, arr in bundle.spectra.items():
            if arr is None:
                continue
            safe = label.replace(" ", "_").replace("—", "").strip("_")
            data[f"{safe}_re"] = arr.real
            data[f"{safe}_im"] = arr.imag
            data[f"{safe}_dB"] = 10 * np.log10(np.abs(arr) ** 2 + 1e-30)

    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    log.info(f"Saved CSV: {filepath}")
    return filepath


# ══════════════════════════════════════════════════════════════════════════════
# Unified save dispatcher
# ══════════════════════════════════════════════════════════════════════════════

def save_results(bundle: ResultBundle, base_dir: Path | str = None) -> None:
    """Save a ResultBundle in the format(s) specified by config.SAVE_FORMAT."""
    base_dir = Path(base_dir or config.RESULTS_DIR)
    fmt      = config.SAVE_FORMAT.lower()

    if fmt in ("hdf5", "both"):
        save_hdf5(bundle, base_dir / "results.h5")
    if fmt in ("csv", "both"):
        save_csv(bundle, base_dir / "csv")
