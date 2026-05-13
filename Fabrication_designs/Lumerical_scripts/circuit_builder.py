"""
circuit_builder.py — 7-ring add-drop cascade circuit builder
=============================================================
Builds the exact circuit seen in the GUI screenshot.

  Confirmed wiring topology
  ──────────────────────────

  RING_1  (unique behaviour):
      ONA_1  output  ──► RING_1  input
      RING_1 through ──► ONA_1   input 1      ← through monitored directly
      RING_1 drop    ──► RING_2  input         ← drop feeds the cascade

  RING_2 … RING_6  (intermediate rings, standard behaviour):
      RING_N  drop    ──► ONA_1      input N   ← resonant channel monitored
      RING_N  through ──► RING_(N+1) input     ← residual light cascades forward

  RING_7  (last ring):
      RING_7  drop    ──► ONA_1  input 7
      RING_7  through ──► ONA_1  input 8

  ONA input map:
      input 1  ←  RING_1 through
      input 2  ←  RING_2 drop
      input 3  ←  RING_3 drop
      input 4  ←  RING_4 drop
      input 5  ←  RING_5 drop
      input 6  ←  RING_6 drop
      input 7  ←  RING_7 drop
      input 8  ←  RING_7 through
                                      Total: 8 inputs ✓

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
    Build the complete 7-ring add-drop cascade.

    Wiring logic
    ─────────────
    RING_1 is unique:
        ONA_1  output  ──► RING_1 input
        RING_1 through ──► ONA_1  input 1     (through monitored on first ring)
        RING_1 drop    ──► RING_2 input        (drop feeds the cascade forward)

    RING_2 … RING_6 (intermediate, standard):
        RING_N  drop    ──► ONA_1      input N
        RING_N  through ──► RING_(N+1) input

    RING_7 (final ring):
        RING_7  drop    ──► ONA_1  input 7
        RING_7  through ──► ONA_1  input 8

    ONA input count:
        1  (RING_1 through)
      + 5  (RING_2 … RING_6 drops)
      + 1  (RING_7 drop)
      + 1  (RING_7 through)
      = 8  ✓

    Parameters
    ----------
    ic           : open lumapi.INTERCONNECT session
    coupling     : coupling coefficient for all rings (None → config default)
    ring_lengths : ring circumference dict in metres (None → config default)

    Returns
    -------
    dict : {"ona": "ONA_1", "rings": ["RING_1", …, "RING_7"]}
    """
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

    # ── ONA output → RING_1 input ──────────────────────────────────────────
    _connect(ic,
             ona_name,       config.ONA_OUTPUT_PORT,
             ring_names[0],  config.RING_PORT_INPUT)
    log.info("  ONA output → RING_1 input")

    # ── RING_1 special case ────────────────────────────────────────────────
    # Through → ONA input 1  (unique to RING_1)
    # Drop    → RING_2 input (starts the cascade)
    _connect(ic,
             ring_names[0],  config.RING_PORT_THROUGH,
             ona_name,       config.ONA_INPUT_PORT_FMT % 1)
    log.info("  RING_1 through → ONA input 1")

    _connect(ic,
             ring_names[0],  config.RING_PORT_DROP,
             ring_names[1],  config.RING_PORT_INPUT)
    log.info("  RING_1 drop    → RING_2 input")

    # ── RING_2 … RING_6 : drop → ONA, through → next ring ─────────────────
    ona_input_idx = 2   # next available ONA input port

    for i in range(1, NUM_RINGS - 1):   # indices 1..5 → RING_2..RING_6
        ring_name = ring_names[i]
        next_ring = ring_names[i + 1]

        _connect(ic,
                 ring_name,  config.RING_PORT_DROP,
                 ona_name,   config.ONA_INPUT_PORT_FMT % ona_input_idx)
        log.info(f"  RING_{i+1} drop    → ONA input {ona_input_idx}")
        ona_input_idx += 1

        _connect(ic,
                 ring_name,  config.RING_PORT_THROUGH,
                 next_ring,  config.RING_PORT_INPUT)
        log.info(f"  RING_{i+1} through → RING_{i+2} input")

    # ── RING_7 : both ports → ONA ──────────────────────────────────────────
    last = ring_names[-1]   # "RING_7"

    _connect(ic,
             last,      config.RING_PORT_DROP,
             ona_name,  config.ONA_INPUT_PORT_FMT % ona_input_idx)
    log.info(f"  RING_7 drop    → ONA input {ona_input_idx}")
    ona_input_idx += 1

    _connect(ic,
             last,      config.RING_PORT_THROUGH,
             ona_name,  config.ONA_INPUT_PORT_FMT % ona_input_idx)
    log.info(f"  RING_7 through → ONA input {ona_input_idx}")
    ona_input_idx += 1

    log.info(f"Ring chain complete: {NUM_RINGS} rings — {ona_input_idx - 1} ONA inputs wired.")
    return {"ona": ona_name, "rings": ring_names}


def reset_circuit(ic) -> None:
    """Delete all elements — start with a clean canvas."""
    ic.deleteall()
    log.info("Circuit reset.")
