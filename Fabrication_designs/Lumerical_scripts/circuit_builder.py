"""
circuit_builder.py — 7-ring add-drop cascade circuit builder
=============================================================
Builds the exact circuit seen in the GUI screenshot:

  ONA_1 (1 output, 8 inputs)
    │
    output ──────────────────────────► RING_1 input
                                            │ through ──► ONA input 1
                                            │ drop    ──► RING_2 input
                                                              │ drop ──► ONA input 2 (*)
                                                              │ through ──► RING_3 input
                                                                                │ ...
                                                              (chain continues)
                                                                                │
                                                                           RING_7
                                                                              │ through ──► ONA input 7
                                                                              │ drop    ──► ONA input 8

  (*) Per the user's description: RING_1 drop feeds RING_2 input AND ONA input 2.
      In INTERCONNECT this is done with a Y-junction or the port is connected to only
      one element — verify if RING_1 drop goes to ONA input 2 OR only to RING_2.
      The current implementation connects RING_N drop → RING_(N+1) input,
      and RING_N drop → ONA input (N+1).  Use a splitter if power needs to be split.
      ← FILL THIS IN / VERIFY the exact wiring intent.

Public API:
    build_ring_chain(ic) → dict   builds the full circuit, returns element names
    reset_circuit(ic)             deletes all elements
"""

import logging
from typing import Any

import config

log = logging.getLogger(__name__)

NUM_RINGS: int = 7


# ══════════════════════════════════════════════════════════════════════════════
# Low-level helpers
# ══════════════════════════════════════════════════════════════════════════════

def _set_props(ic, element_name: str, props: dict) -> None:
    """Select element and apply a {property: value} dict."""
    ic.select(element_name)
    for prop, value in props.items():
        try:
            ic.set(prop, value)
            log.debug(f"  {element_name}.{prop} = {value}")
        except Exception as e:
            log.warning(f"  Could not set '{prop}' on {element_name}: {e}")


def _connect(ic, elem_a: str, port_a: str, elem_b: str, port_b: str) -> None:
    """Wire port_a of elem_a to port_b of elem_b with error logging."""
    try:
        ic.connect(elem_a, port_a, elem_b, port_b)
        log.debug(f"  Connected  {elem_a}:{port_a}  →  {elem_b}:{port_b}")
    except Exception as e:
        log.error(f"  FAILED connect {elem_a}:{port_a} → {elem_b}:{port_b}: {e}")
        raise


# ══════════════════════════════════════════════════════════════════════════════
# Element constructors
# ══════════════════════════════════════════════════════════════════════════════

def add_ona(ic, name: str = "ONA_1") -> str:
    """
    Add the Optical Network Analyzer with 8 input ports and 1 output port.
    Analysis type is set to 'scattering data' as seen in the GUI.
    """
    ic.addelement("Optical Network Analyzer")
    _set_props(ic, "Optical Network Analyzer", {
        "name"                  : name,
        "analysis type"         : config.ONA_ANALYSIS_TYPE,
        "start frequency"       : config.DEFAULT_START_FREQ,
        "stop frequency"        : config.DEFAULT_STOP_FREQ,
        "number of points"      : config.DEFAULT_NUM_POINTS,
        "number of input ports" : config.ONA_NUM_INPUT_PORTS,
        "number of output ports": config.ONA_NUM_OUTPUT_PORTS,
    })
    log.info(f"Added ONA: {name}  ({config.ONA_NUM_INPUT_PORTS} inputs, "
             f"{config.DEFAULT_START_FREQ/1e12:.2f}–{config.DEFAULT_STOP_FREQ/1e12:.2f} THz, "
             f"{config.DEFAULT_NUM_POINTS} pts)")
    return name


def add_ring(ic, name: str, length_m: float, coupling: float = None) -> str:
    """
    Add one double-bus add-drop ring resonator.

    Parameters
    ----------
    name     : element name, e.g. "RING_1"
    length_m : ring circumference in metres (from config.RING_LENGTHS_M)
    coupling : power coupling coefficient (0–1); None → config default

    neff and ng are intentionally NOT set — left at INTERCONNECT defaults.
    """
    coupling = coupling if coupling is not None else config.DEFAULT_RING_COUPLING

    ic.addelement(config.RING_ELEMENT_LIBRARY_NAME)

    props = {
        "name"   : name,
        "length" : length_m,   # ring circumference, metres
    }

    # Coupling — the property name may differ between element versions:
    #   "coupling coefficient 1"  (input coupling)
    #   "coupling coefficient 2"  (drop coupling)
    # ← FILL THIS IN / VERIFY exact property names for your ring element
    props["coupling coefficient 1"] = coupling
    props["coupling coefficient 2"] = coupling

    # Loss — only set if non-zero
    if config.DEFAULT_RING_LOSS_DB != 0.0:
        props["loss"] = config.DEFAULT_RING_LOSS_DB   # ← FILL THIS IN property name

    # neff and ng are NOT set (USE_DEFAULT_NEFF_NG = True)
    if not config.USE_DEFAULT_NEFF_NG:
        # Only reaches here if you set USE_DEFAULT_NEFF_NG = False in config.py
        props["effective index 1"] = 2.4   # ← FILL THIS IN
        props["group index"]       = 4.2   # ← FILL THIS IN

    _set_props(ic, config.RING_ELEMENT_LIBRARY_NAME, props)
    log.info(f"Added ring: {name}  L={length_m*1e6:.4f} µm  κ={coupling:.3f}")
    return name


# ══════════════════════════════════════════════════════════════════════════════
# Full circuit builder
# ══════════════════════════════════════════════════════════════════════════════

def build_ring_chain(
    ic,
    coupling    : float = None,
    ring_lengths: dict  = None,
) -> dict:
    """
    Build the complete 7-ring add-drop cascade as seen in the GUI.

    Wiring logic
    ─────────────
    ONA_1 output  ──► RING_1 input

    For N = 1 … 6:
        RING_N through  ──► ONA_1 input N          (monitored through port)
        RING_N drop     ──► RING_(N+1) input        (cascade: drop feeds next ring)

    RING_7 through  ──► ONA_1 input 7
    RING_7 drop     ──► ONA_1 input 8

    NOTE ON DROP-TO-ONA:
    The user's description says "drop of RING_N to one input of ONA" AND
    "drop feeds RING_(N+1)".  In INTERCONNECT a port can only connect to ONE
    element.  Two options:
      A) Use a Y-branch splitter between RING_N drop and (ONA + RING_(N+1)).
      B) The drop port connects ONLY to RING_(N+1); the through port of RING_(N+1)
         is what goes to ONA — this is the standard demux / spectral slicer topology.
    This script implements option B (standard cascaded drop-port demux) unless
    you set WIRE_DROP_TO_ONA_DIRECTLY = True below.

    ← FILL THIS IN: set the flag that matches your actual intended circuit.

    Parameters
    ----------
    ic           : open lumapi.INTERCONNECT session
    coupling     : override default coupling (None → config.DEFAULT_RING_COUPLING)
    ring_lengths : override ring lengths dict (None → config.RING_LENGTHS_M)

    Returns
    -------
    dict with keys: "ona", "rings" (list of ring names)
    """

    # ── Configuration ─────────────────────────────────────────────────────
    # Set to True if RING_N drop connects DIRECTLY to both ONA and RING_(N+1)
    # (requires a Y-branch; only valid if you added splitters in the GUI).
    # Set to False for standard cascaded through-port monitoring topology.
    WIRE_DROP_TO_ONA_DIRECTLY: bool = False   # ← FILL THIS IN

    ring_lengths = ring_lengths or config.RING_LENGTHS_M

    # ── Add ONA ────────────────────────────────────────────────────────────
    ona_name = add_ona(ic, name="ONA_1")

    # ── Add all 7 rings ────────────────────────────────────────────────────
    ring_names = []
    for i in range(1, NUM_RINGS + 1):
        rname = f"RING_{i}"
        add_ring(
            ic,
            name     = rname,
            length_m = ring_lengths[rname],
            coupling = coupling,
        )
        ring_names.append(rname)

    # ── Wire ONA output → RING_1 input ────────────────────────────────────
    _connect(ic,
             ona_name,          config.ONA_OUTPUT_PORT,
             ring_names[0],     config.RING_PORT_INPUT)

    # ── Wire the cascade chain ─────────────────────────────────────────────
    # ONA input port counter starts at 1
    ona_input_idx = 1

    for i, ring_name in enumerate(ring_names):
        is_last = (i == NUM_RINGS - 1)
        next_ring = ring_names[i + 1] if not is_last else None

        # ── Through port → ONA input N ────────────────────────────────────
        _connect(ic,
                 ring_name,   config.RING_PORT_THROUGH,
                 ona_name,    config.ONA_INPUT_PORT_FMT % ona_input_idx)
        log.info(f"  {ring_name} through → ONA input {ona_input_idx}")
        ona_input_idx += 1

        if not is_last:
            if WIRE_DROP_TO_ONA_DIRECTLY:
                # Option A: drop → ONA input, AND drop → next ring input
                # This requires a Y-branch splitter element between them.
                # ← FILL THIS IN: add splitter element and wire it
                # Example (pseudocode):
                #   splitter = add_ybranch(ic, name=f"YB_{i+1}")
                #   _connect(ic, ring_name,  RING_PORT_DROP,   splitter, "input")
                #   _connect(ic, splitter,   "output 1",       next_ring, RING_PORT_INPUT)
                #   _connect(ic, splitter,   "output 2",       ona_name,  ONA_INPUT_PORT_FMT % ona_input_idx)
                #   ona_input_idx += 1
                log.warning(f"WIRE_DROP_TO_ONA_DIRECTLY=True but Y-branch not implemented. "
                            f"Connecting drop only to next ring.")
                _connect(ic,
                         ring_name,   config.RING_PORT_DROP,
                         next_ring,   config.RING_PORT_INPUT)
            else:
                # Option B (default): drop → next ring input only
                _connect(ic,
                         ring_name,   config.RING_PORT_DROP,
                         next_ring,   config.RING_PORT_INPUT)
                log.info(f"  {ring_name} drop → {next_ring} input")
        else:
            # Last ring: drop → ONA final input
            _connect(ic,
                     ring_name,   config.RING_PORT_DROP,
                     ona_name,    config.ONA_INPUT_PORT_FMT % ona_input_idx)
            log.info(f"  {ring_name} drop → ONA input {ona_input_idx}")
            ona_input_idx += 1

    log.info(f"Ring chain built: {NUM_RINGS} rings, {ona_input_idx - 1} ONA inputs wired.")
    return {"ona": ona_name, "rings": ring_names}


def reset_circuit(ic) -> None:
    """Delete all elements — start with a clean canvas."""
    ic.deleteall()
    log.info("Circuit reset.")
