"""
config.py — Single, user-facing configuration for the SiN FDE mode-analysis
package.

This is the ONE place to set up an application run. Everything that can be
decided *before* any simulation lives here as a plain input: the operating
wavelength, the target FSR, the per-ring FWHM, the number of spectrometer rings
and therefore the ring resonance wavelengths. The package then produces the
geometry the application needs (waveguide width, ring radii, coupler gaps, ...).

The ONE geometric quantity that cannot be fixed a priori is the single-mode
waveguide WIDTH: it depends on where the guide goes multimode, which is only
known after the width sweep + modal analysis. Instead of hard-coding it, this
file specifies the RULE for choosing it — the working width is taken a fixed
margin below the TE multimode-cutoff width (see SINGLE_MODE_MARGIN_NM and
select_single_mode_width_nm()). The modal-analysis step applies that rule to the
measured cutoff and hands the resulting width to the ring steps through `state`.

Units convention (NEVER broken):
    * Everything passed to Lumerical is SI (metres, Hz).
    * nm / um are used only for human-readable display and for sweep arrays
      whose names end in `_UM` / `_NM`.

Physical platform (SiN strip waveguide, evanescent RI biosensor):
    core      : Si3N4,  n = 1.99      , height 400 nm
    lower clad: SiO2 ,  n = 1.4469    (thermal BOX / substrate)
    upper clad: aqueous n = 1.33 (sensor)  OR  SiO2 n = 1.4469 (spectrometer)
    width     : auto-selected single-mode ; polarisation TE0.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

# ╔═════════════════════════════════════════════════════════════════════════════
# ║  USER INPUTS — set the whole application here (no prior simulation needed)  ║
# ╚═════════════════════════════════════════════════════════════════════════════

# ── Operating / design wavelength ────────────────────────────────────────────
LAMBDA0_NM = 780.0              # central design wavelength (sensor ring)   [nm]

# ── Spectral design of the ring cascade ──────────────────────────────────────
TARGET_FSR_NM  = 10.0            # target free spectral range, every ring    [nm]
N_SPEC_RINGS   = 13              # number of SiO2 spectrometer rings
SENSOR_FWHM_NM = 0.5             # target FWHM of the aqueous sensor ring     [nm]
SPEC_FWHM_NM   = 0.5             # target FWHM of each spectrometer ring      [nm]

# The N_SPEC_RINGS spectrometer resonances tile exactly one FSR above LAMBDA0:
#     lambda_n = LAMBDA0_NM + n * (TARGET_FSR_NM / N_SPEC_RINGS),  n = 0 .. N-1
SPEC_DELTA_LAMBDA_NM = TARGET_FSR_NM / N_SPEC_RINGS          # ~0.769231 nm
RING_RESONANCES_NM   = LAMBDA0_NM + np.arange(N_SPEC_RINGS) * SPEC_DELTA_LAMBDA_NM

# ── Single-mode WIDTH selection (needs the width sweep + modal analysis) ──────
#   The working width cannot be fixed up front. It is chosen automatically as
#       width = (TE multimode-cutoff width)  −  SINGLE_MODE_MARGIN_NM
#   which keeps the guide comfortably single-mode. Tune the backoff here.
#   WG_WIDTH_OVERRIDE_NM forces a width and bypasses the automatic selection;
#   WG_WIDTH_FALLBACK_NM is only used if a step runs without the modal step
#   having provided a width (and no override is set).
SINGLE_MODE_MARGIN_NM = 30.0    # backoff below the TE MM cutoff width       [nm]
WG_WIDTH_OVERRIDE_NM  = None     # None = auto-select ; or e.g. 1000.0 to force [nm]
WG_WIDTH_FALLBACK_NM  = 1000.0   # used only when neither auto nor override available


def select_single_mode_width_nm(cutoff_nm):
    """
    Choose the single-mode working width from a measured TE multimode cutoff.

    Inputs
    ------
    cutoff_nm : float | None
        TE multimode-onset width [nm] measured by the modal-analysis step,
        or None if the guide is single-mode across the whole sweep.

    Outputs
    -------
    float | None
        WG_WIDTH_OVERRIDE_NM if it is set; otherwise
        cutoff_nm − SINGLE_MODE_MARGIN_NM; or None if no cutoff is available
        and no override is set.
    """
    if WG_WIDTH_OVERRIDE_NM is not None:
        return float(WG_WIDTH_OVERRIDE_NM)
    if cutoff_nm is None:
        return None
    return float(cutoff_nm) - float(SINGLE_MODE_MARGIN_NM)


# ╔═════════════════════════════════════════════════════════════════════════════
# ║  SIMULATION / SOLVER SETTINGS (rarely changed)                             ║
# ╚═════════════════════════════════════════════════════════════════════════════

# ── Lumerical installation ───────────────────────────────────────────────────
#   Set LUMERICAL_VERSION to match the folder name under
#   C:\Program Files\Lumerical\   (Windows)  or  /opt/lumerical/  (Linux).
LUMERICAL_VERSION = "v202"

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("SiN_FDE")

# ── I/O — output directory and HDF5 cache paths ──────────────────────────────
VERSION_NAME = "LUM_SiN_STRp_220nm_wdth_sweep_V1"
PROJECT_DIR = Path.cwd()
DATA_DIR = PROJECT_DIR / "data_STRp_SiN_mode_analysis_LUM"
DATA_DIR.mkdir(parents=True, exist_ok=True)
HDF5_PATH = DATA_DIR / f"{VERSION_NAME}.h5"

# SiO2-clad (symmetric stack) width sweep — same engine, different background
VERSION_NAME_SIO2 = "LUM_SiN_STRp_220nm_wdth_sweep_SiO2clad_V1"
HDF5_PATH_SIO2 = DATA_DIR / f"{VERSION_NAME_SIO2}.h5"

# ── One dedicated HDF5 file PER SWEEP ─────────────────────────────────────────
# Previously every post-width sweep wrote groups into the width-sweep file,
# coupling unrelated simulations into one container. Each sweep now owns its
# own file (still resumable independently).
#   step1  -> HDF5_PATH               (width sweep, aqueous clad)
#   step1b -> HDF5_PATH_SIO2          (width sweep, SiO2 clad)
#   step3  -> HDF5_PATH_RING_RADIUS   (sensor ring radius sweep)
#   step5  -> HDF5_PATH_SPECTROMETER  (13 spectrometer ring radius sweeps)
#   step7  -> HDF5_PATH_COUPLER       (coupler-gap sweep; sensor + spectrometer)
#   step9  -> HDF5_PATH_AQUEOUS_SWEEP (aqueous index sweep)
HDF5_PATH_RING_RADIUS   = DATA_DIR / f"{VERSION_NAME}_ring_radius.h5"
HDF5_PATH_SPECTROMETER  = DATA_DIR / f"{VERSION_NAME_SIO2}_spectrometer.h5"
HDF5_PATH_COUPLER       = DATA_DIR / f"{VERSION_NAME}_coupler_gap.h5"
HDF5_PATH_AQUEOUS_SWEEP = DATA_DIR / f"{VERSION_NAME}_aqueous_index_sweep.h5"

# Where final figures are exported (PNG + PDF)
FIGURES_DIR = DATA_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Waveguide geometry (um unless stated) ────────────────────────────────────
CORE_THICKNESS_UM = 0.220        # SiN core height  [um]
N_MODES_REQUEST = 6              # trial modes requested from the FDE solver

SIM_Y_MARGIN_UM = 2.5            # lateral cladding margin each side [um]
SIM_Z_BELOW_UM = 2.0             # SiO2 substrate depth below core   [um]
SIM_Z_ABOVE_UM = 2.0             # upper-cladding depth above core    [um]

# ── Material refractive indices (wavelength-dependent) ───────────────────────
#   These are evaluated ONCE at LAMBDA0_NM from published Sellmeier models, then
#   held constant across the sweep. They are NOT the same at 1550 nm and 780 nm,
#   so they must track LAMBDA0_NM — that is exactly why they are computed here
#   instead of being hard-coded. Sources:
#     Si3N4 : Luke et al., Opt. Lett. 40, 4823 (2015)   (stoichiometric LPCVD)
#     SiO2  : Malitson,    JOSA  55, 1205 (1965)        (fused silica)
def n_si3n4(lambda_nm):
    """Stoichiometric Si3N4 index (Luke 2015). lambda in nm -> n (dimensionless)."""
    l2 = (lambda_nm * 1e-3) ** 2                       # um^2
    return float(np.sqrt(1.0 + 3.0249 * l2 / (l2 - 0.1353406 ** 2)
                              + 40314.0 * l2 / (l2 - 1239.842 ** 2)))


def n_sio2(lambda_nm):
    """Fused-silica index (Malitson 1965). lambda in nm -> n (dimensionless)."""
    l2 = (lambda_nm * 1e-3) ** 2                       # um^2
    return float(np.sqrt(1.0 + 0.6961663 * l2 / (l2 - 0.0684043 ** 2)
                              + 0.4079426 * l2 / (l2 - 0.1162414 ** 2)
                              + 0.8974794 * l2 / (l2 - 9.896161 ** 2)))


# Optional overrides: set to a number to PIN an index (e.g. to reproduce the
# validated C-band baseline n_SiN = 1.99, n_SiO2 = 1.4469). None = use Sellmeier.
N_SIN_OVERRIDE  = None
N_SIO2_OVERRIDE = None

N_SIN_FIXED  = float(N_SIN_OVERRIDE)  if N_SIN_OVERRIDE  is not None else n_si3n4(LAMBDA0_NM)
N_SIO2_FIXED = float(N_SIO2_OVERRIDE) if N_SIO2_OVERRIDE is not None else n_sio2(LAMBDA0_NM)
N_UPPER_CLADDING = 1.33          # aqueous analyte (sensor upper cladding)  [set by user]
N_UPPER_CLADDING_SIO2 = N_SIO2_FIXED  # symmetric SiO2 stack (spectrometer)


def max_single_mode_thickness_um(lambda_nm=None, n_core=None, n_clad=None):
    """
    Symmetric-slab ceiling on core THICKNESS for vertical single-mode operation
    (only the fundamental vertical TE mode guided): t < lambda / (2 * NA), with
    NA = sqrt(n_core^2 - n_clad^2). Returns micrometres.
    """
    lambda_nm = LAMBDA0_NM if lambda_nm is None else lambda_nm
    n_core    = N_SIN_FIXED if n_core is None else n_core
    n_clad    = N_SIO2_FIXED if n_clad is None else n_clad
    na = float(np.sqrt(n_core ** 2 - n_clad ** 2))
    return (lambda_nm / (2.0 * na)) * 1e-3            # nm -> um

# Vertical single-mode self-check: the SiN core HEIGHT, like the width, must be
# chosen for the operating wavelength. A height good at 1550 nm is too thick at
# 780 nm. Warn (do not silently "fix") if CORE_THICKNESS_UM exceeds the ceiling.
_t_sm_max_um = max_single_mode_thickness_um()
if CORE_THICKNESS_UM > _t_sm_max_um:
    log.warning(
        f"CORE_THICKNESS_UM = {CORE_THICKNESS_UM*1e3:.0f} nm exceeds the vertical "
        f"single-mode ceiling {_t_sm_max_um*1e3:.0f} nm at lambda0 = {LAMBDA0_NM:.0f} nm "
        f"(n_SiN={N_SIN_FIXED:.3f}, n_SiO2={N_SIO2_FIXED:.3f}). The core supports "
        f">1 vertical mode, so NO width can make it single-mode and the symmetric "
        f"SiO2-clad (spectrometer) rings will be multimode across the whole width "
        f"sweep. Reduce CORE_THICKNESS_UM at this wavelength."
    )

# ── Width sweep : 600 nm -> 1500 nm, 100 uniformly spaced points ─────────────
#   This is the sweep that locates the multimode cutoff and hence sets the
#   single-mode working width (above).
SWEEP_WIDTHS_UM = np.linspace(0.350, 0.760, 100)     # [um]

# ── Wavelength sweep (DERIVED from the user inputs above) ─────────────────────
#   lambda0 + n*(FSR/N_SPEC_RINGS), one point per spectrometer ring, so the 13
#   resonances tile exactly one FSR. Kept under the original names that the
#   width-sweep / modal steps already consume.
LAMBDA_START_NM = LAMBDA0_NM
N_WAVELENGTHS = N_SPEC_RINGS
DELTA_LAMBDA_NM = SPEC_DELTA_LAMBDA_NM
SWEEP_WL_NM = RING_RESONANCES_NM
SWEEP_WL_UM = SWEEP_WL_NM * 1e-3                      # [um]
SWEEP_WL_M = SWEEP_WL_UM * 1e-6                       # [m] — Lumerical SI

# ── Derived simulation-domain dimensions ─────────────────────────────────────
SIM_Y_SPAN_UM = SWEEP_WIDTHS_UM.max() + 2.0 * SIM_Y_MARGIN_UM      # 6.5 um
SIM_Z_SPAN_UM = SIM_Z_BELOW_UM + CORE_THICKNESS_UM + SIM_Z_ABOVE_UM  # 4.4 um

# ── Mesh sizing : ~10 cells per wavelength inside the SiN core ───────────────
MESH_CELLS_PER_WVL = 10
_mesh_step_um = SWEEP_WL_UM.max() / (N_SIN_FIXED * MESH_CELLS_PER_WVL)
MESH_CELLS_Y = int(np.ceil(SIM_Y_SPAN_UM / _mesh_step_um))         # ~83
MESH_CELLS_Z = int(np.ceil(SIM_Z_SPAN_UM / _mesh_step_um))         # ~56

# Convenience: all names that `from config import *` should export.
__all__ = [name for name in dir() if not name.startswith("_")]
