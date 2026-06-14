"""
config.py — Centralised, editable configuration for the SiN FDE mode-analysis
package.

This module collects every constant that the original monolithic notebook kept
live in the Jupyter *kernel namespace* and that is therefore shared by more than
one analysis step (the "data contract" of the notebook).  Step-specific design
knobs (ring-radius search ranges, spectrometer parameters, coupler-gap ranges,
aqueous-index ranges, ...) deliberately stay inside their own step modules, so
that each script remains a self-documenting, single place to tune that step.

Units convention (NEVER broken):
    * Everything passed to Lumerical is SI (metres, Hz).
    * nm / um are used only for human-readable display and for sweep arrays
      whose names end in `_UM` / `_NM`.

Physical platform (SiN strip waveguide, lead/RI evanescent biosensor):
    core      : Si3N4,  n = 1.99      , height 400 nm
    lower clad: SiO2 ,  n = 1.4469    (thermal BOX / substrate)
    upper clad: aqueous n = 1.33 (sensor)  OR  SiO2 n = 1.4469 (spectrometer)
    width     : ~1000 nm nominal ; polarisation TE0 ; lambda0 = 1550 nm.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Lumerical installation
#   Set LUMERICAL_VERSION to match the folder name under
#   C:\Program Files\Lumerical\   (Windows)  or  /opt/lumerical/  (Linux).
# ─────────────────────────────────────────────────────────────────────────────
LUMERICAL_VERSION = "v202"

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("SiN_FDE")

# ─────────────────────────────────────────────────────────────────────────────
# I/O — output directory and HDF5 cache paths
# ─────────────────────────────────────────────────────────────────────────────
VERSION_NAME = "LUM_SiN_STRp_400nm_wdth_sweep_V1"
PROJECT_DIR = Path.cwd()
DATA_DIR = PROJECT_DIR / "data_STRp_SiN_mode_analysis_LUM"
DATA_DIR.mkdir(parents=True, exist_ok=True)
HDF5_PATH = DATA_DIR / f"{VERSION_NAME}.h5"

# SiO2-clad (symmetric stack) width sweep — same engine, different background
VERSION_NAME_SIO2 = "LUM_SiN_STRp_400nm_wdth_sweep_SiO2clad_V1"
HDF5_PATH_SIO2 = DATA_DIR / f"{VERSION_NAME_SIO2}.h5"

# Where final figures are exported (PNG + PDF)
FIGURES_DIR = DATA_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Waveguide geometry (um unless stated)
# ─────────────────────────────────────────────────────────────────────────────
CORE_THICKNESS_UM = 0.400        # SiN core height  [um]
N_MODES_REQUEST = 6              # trial modes requested from the FDE solver

SIM_Y_MARGIN_UM = 2.5            # lateral cladding margin each side [um]
SIM_Z_BELOW_UM = 2.0             # SiO2 substrate depth below core   [um]
SIM_Z_ABOVE_UM = 2.0             # upper-cladding depth above core    [um]

# ─────────────────────────────────────────────────────────────────────────────
# Fixed refractive indices (evaluated once at 1550 nm via Sellmeier, then held
# constant across the whole width / wavelength sweep).
# ─────────────────────────────────────────────────────────────────────────────
N_SIN_FIXED = 1.99               # SiN  @ 1550 nm  (FDE index + mesh sizing)
N_SIO2_FIXED = 1.4469            # SiO2 @ 1550 nm  (FDE index + guided-mode cutoff)
N_UPPER_CLADDING = 1.33          # aqueous medium (sensor upper cladding)
N_UPPER_CLADDING_SIO2 = N_SIO2_FIXED  # symmetric SiO2 stack (spectrometer)

# ─────────────────────────────────────────────────────────────────────────────
# Width sweep : 600 nm -> 1500 nm, 100 uniformly spaced points
# ─────────────────────────────────────────────────────────────────────────────
SWEEP_WIDTHS_UM = np.linspace(0.600, 1.500, 100)     # [um]

# ─────────────────────────────────────────────────────────────────────────────
# Wavelength sweep : lambda0 = 1550 nm + n * (10/13) nm, 13 points
# (exact rational step so the 13 spectrometer resonances tile one 10 nm FSR)
# ─────────────────────────────────────────────────────────────────────────────
LAMBDA_START_NM = 1550.0
DELTA_LAMBDA_NM = 10.0 / 13.0                         # ~0.769231 nm
N_WAVELENGTHS = 13
SWEEP_WL_NM = LAMBDA_START_NM + np.arange(N_WAVELENGTHS) * DELTA_LAMBDA_NM
SWEEP_WL_UM = SWEEP_WL_NM * 1e-3                      # [um]
SWEEP_WL_M = SWEEP_WL_UM * 1e-6                       # [m] — Lumerical SI

# ─────────────────────────────────────────────────────────────────────────────
# Derived simulation-domain dimensions
# ─────────────────────────────────────────────────────────────────────────────
SIM_Y_SPAN_UM = SWEEP_WIDTHS_UM.max() + 2.0 * SIM_Y_MARGIN_UM      # 6.5 um
SIM_Z_SPAN_UM = SIM_Z_BELOW_UM + CORE_THICKNESS_UM + SIM_Z_ABOVE_UM  # 4.4 um

# ─────────────────────────────────────────────────────────────────────────────
# Mesh sizing : ~10 cells per wavelength inside the SiN core
# ─────────────────────────────────────────────────────────────────────────────
MESH_CELLS_PER_WVL = 10
_mesh_step_um = SWEEP_WL_UM.max() / (N_SIN_FIXED * MESH_CELLS_PER_WVL)
MESH_CELLS_Y = int(np.ceil(SIM_Y_SPAN_UM / _mesh_step_um))         # ~83
MESH_CELLS_Z = int(np.ceil(SIM_Z_SPAN_UM / _mesh_step_um))         # ~56

# Convenience: all names that `from config import *` should export.
__all__ = [name for name in dir() if not name.startswith("_")]
