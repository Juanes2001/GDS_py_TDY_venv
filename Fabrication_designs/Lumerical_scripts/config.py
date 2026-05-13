"""
config.py — Global configuration for the 7-ring add-drop chain
===============================================================
Circuit topology (from GUI screenshot):
  One ONA_1 with 8 input ports drives a cascade of 7 double-bus add-drop rings.
  The drop port of RING_N feeds the input of RING_(N+1).
  Every drop and the final through port connect back to individual ONA inputs.

  ONA output ──► RING_1 input
  RING_1 through ──► ONA input 1
  RING_1 drop    ──► RING_2 input
  RING_2 through ──► (not connected — absorbed in ring)  (*see note below)
  RING_2 drop    ──► ONA input 2   ... and so on.
  RING_7 through ──► ONA input 7 (or 8)
  RING_7 drop    ──► ONA input 8 (or 7)

  (*) Re-read your GUI wiring carefully and adjust ONA_PORT_LABELS + wiring
      in circuit_builder.py if the through ports of intermediate rings are
      also monitored.

INSTRUCTIONS:
    Search for  ← FILL THIS IN  and complete every one before running.
    Values already set were read directly from the GUI screenshot.
"""

from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# 1.  LUMERICAL INSTALLATION
# ══════════════════════════════════════════════════════════════════════════════

LUMERICAL_API_PATH: str = r"C:\Program Files\Lumerical\v242\api\python"  # ← FILL THIS IN
LUMERICAL_VERSION : str = "v242"                                          # ← FILL THIS IN
HIDE_GUI          : bool = True   # False → show GUI (useful for debugging layout)

# ══════════════════════════════════════════════════════════════════════════════
# 2.  PROJECT PATHS  (auto-derived — no edits needed)
# ══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT    : Path = Path(__file__).parent.resolve()
SIMULATIONS_DIR : Path = PROJECT_ROOT / "simulations"
RESULTS_DIR     : Path = PROJECT_ROOT / "results"
FIGURES_DIR     : Path = PROJECT_ROOT / "figures"
LOGS_DIR        : Path = PROJECT_ROOT / "logs"

for _d in (SIMULATIONS_DIR, RESULTS_DIR, FIGURES_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 3.  ONA SWEEP PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

# Optical frequency sweep range.  Rings with circumferences ~515–546 µm
# (≈ FSR ~0.5 nm at 1550 nm) → suggest sweeping at least 5–10 nm.
DEFAULT_START_FREQ : float = 192.5e12   # ← FILL THIS IN  Hz  (~1557 nm)
DEFAULT_STOP_FREQ  : float = 194.5e12   # ← FILL THIS IN  Hz  (~1542 nm)
DEFAULT_NUM_POINTS : int   = 10000      # ← FILL THIS IN  (high count needed for sharp resonances)

C_LIGHT            : float = 299_792_458.0   # m/s — do not change
DEFAULT_START_WAV  : float = C_LIGHT / DEFAULT_STOP_FREQ
DEFAULT_STOP_WAV   : float = C_LIGHT / DEFAULT_START_FREQ

# Must match the GUI setting shown in the screenshot
ONA_ANALYSIS_TYPE  : str   = "scattering data"

# 8 input ports on ONA_1 (confirmed from GUI: 8 green connectors on left side)
ONA_NUM_INPUT_PORTS : int = 8
ONA_NUM_OUTPUT_PORTS: int = 1

# ══════════════════════════════════════════════════════════════════════════════
# 4.  RING ELEMENT LIBRARY NAME
#     Open INTERCONNECT → Element Library → search for your ring element.
#     Copy the exact string here.
# ══════════════════════════════════════════════════════════════════════════════

RING_ELEMENT_LIBRARY_NAME: str = "Ring Modulator"  # ← FILL THIS IN / VERIFY

# ══════════════════════════════════════════════════════════════════════════════
# 5.  RING LENGTHS  (read from GUI Properties panel — already filled in)
#     Values are in micrometres as shown; converted to metres for lumapi.
# ══════════════════════════════════════════════════════════════════════════════

RING_LENGTHS_UM: dict = {
    "RING_1": 537.8868,
    "RING_2": 545.1331,
    "RING_3": 515.3218,
    "RING_4": 545.5105,
    "RING_5": 545.6993,
    "RING_6": 515.8881,
    "RING_7": 516.0769,
}

# SI units for lumapi — do not change this line
RING_LENGTHS_M: dict = {k: v * 1e-6 for k, v in RING_LENGTHS_UM.items()}

# ══════════════════════════════════════════════════════════════════════════════
# 6.  RING OPTICAL PROPERTIES
#     neff and ng are left at INTERCONNECT defaults (as instructed).
#     Only coupling and loss need to be set.
# ══════════════════════════════════════════════════════════════════════════════

# Power coupling coefficient (same for input-bus and drop-bus coupling regions)
DEFAULT_RING_COUPLING : float = 0.1    # ← FILL THIS IN  (0–1, e.g. 0.1 = 10 %)

# Round-trip amplitude loss coefficient (0 = lossless, use INTERCONNECT default)
DEFAULT_RING_LOSS_DB  : float = 0.0    # ← FILL THIS IN  (or keep 0.0 for lossless model)

# True → script will NOT call set() for "effective index" or "group index"
# keeping whatever INTERCONNECT default values your element library uses
USE_DEFAULT_NEFF_NG   : bool  = True

# ══════════════════════════════════════════════════════════════════════════════
# 7.  PORT NAMES ON THE RING ELEMENT
#     Verify these by hovering over ports in INTERCONNECT GUI.
#     The GUI screenshot shows: "input", "output 1", "output 2" as visible labels.
# ══════════════════════════════════════════════════════════════════════════════

RING_PORT_INPUT   : str = "input"     # ← FILL THIS IN / VERIFY  — bus 1 input
RING_PORT_THROUGH : str = "output 1"  # ← FILL THIS IN / VERIFY  — bus 1 through
RING_PORT_ADD     : str = "input 2"   # ← FILL THIS IN / VERIFY  — bus 2 add (may be unused)
RING_PORT_DROP    : str = "output 2"  # ← FILL THIS IN / VERIFY  — bus 2 drop

# ONA port label formats
ONA_OUTPUT_PORT    : str = "output"      # ← FILL THIS IN / VERIFY
ONA_INPUT_PORT_FMT : str = "input %d"    # produces "input 1", "input 2", …

# ══════════════════════════════════════════════════════════════════════════════
# 8.  WIRING MAP
#     Defines which ONA input port receives which ring port.
#     Index 0 = ONA input 1, index 7 = ONA input 8.
#     Edit this list if your actual GUI wiring differs.
# ══════════════════════════════════════════════════════════════════════════════

# Each entry: (ring_name, port_on_ring)
ONA_INPUT_WIRING: list = [
    ("RING_1", "through"),   # ONA input 1  ← RING_1 through port
    ("RING_1", "drop"),      # ONA input 2  ← RING_1 drop port  (feeds RING_2 AND ONA? verify)
    ("RING_2", "drop"),      # ONA input 3  ← RING_2 drop port
    ("RING_3", "drop"),      # ONA input 4  ← RING_3 drop port
    ("RING_4", "drop"),      # ONA input 5  ← RING_4 drop port
    ("RING_5", "drop"),      # ONA input 6  ← RING_5 drop port
    ("RING_6", "drop"),      # ONA input 7  ← RING_6 drop port
    ("RING_7", "through"),   # ONA input 8  ← RING_7 through port
    # NOTE: RING_7 drop is also wired per the description — add a 9th port if needed
    # ← FILL THIS IN / VERIFY against your actual GUI
]

# ══════════════════════════════════════════════════════════════════════════════
# 9.  SWEEP CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

MAX_PARALLEL_SESSIONS : int = 1    # ← FILL THIS IN  (limited by license seats)
SIMULATION_TIMEOUT    : int = 120  # seconds per run

# ══════════════════════════════════════════════════════════════════════════════
# 10. OUTPUT FORMAT
# ══════════════════════════════════════════════════════════════════════════════

SAVE_FORMAT             : str = "hdf5"   # "hdf5" | "csv" | "both"
HDF5_COMPRESSION_LEVEL  : int = 4

# ══════════════════════════════════════════════════════════════════════════════
# 11. PLOTTING DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════

PLOT_DPI      : int   = 300
PLOT_FIGSIZE  : tuple = (10, 6)
PLOT_FORMAT   : str   = "pdf"
PLOT_COLORMAP : str   = "plasma"
PLOT_STYLE    : str   = "seaborn-v0_8-paper"

# Human-readable labels for each ONA input port (used in plot legends)
ONA_PORT_LABELS: list = [
    "RING_1 — through",   # ONA input 1
    "RING_1 — drop",      # ONA input 2
    "RING_2 — drop",      # ONA input 3
    "RING_3 — drop",      # ONA input 4
    "RING_4 — drop",      # ONA input 5
    "RING_5 — drop",      # ONA input 6
    "RING_6 — drop",      # ONA input 7
    "RING_7 — through",   # ONA input 8
]

# ══════════════════════════════════════════════════════════════════════════════
# 12. LOGGING
# ══════════════════════════════════════════════════════════════════════════════

LOG_LEVEL   : str  = "INFO"
LOG_TO_FILE : bool = True
