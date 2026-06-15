"""
step9_aqueous_sweep.py — sensor-ring aqueous-index sweep (the sensor model).

Central sensor characterisation: bent-waveguide FDE sweep of the aqueous
cladding index 1.33->1.37 at fixed radius. Produces ais_neff/ais_ng and
the RI sensitivities that feed the INTERCONNECT circuit package.
"""
from __future__ import annotations

import math
import time
from datetime import datetime

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import cm, gridspec, colors
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.lines as mlines
import pandas as pd

from .config import *  # shared platform constants & paths
from .lumerical_session import import_lumapi
from .plotting import apply_style, save_fig, make_colorbar
from . import plotting as _plot

# ── Data-contract inputs (injected by main.py via `state`) ──────────
_exc = None
_l = None

apply_style()

# ─────────────────────────────────────────────────────────
#  Module constants / design knobs for this step
# ─────────────────────────────────────────────────────────
AIS_BEND_RADIUS_UM   = 19.0021        # [µm]  ← change this per experiment
AIS_N_AQ_START       = 1.3300      # lower bound  (pure water @ 1550 nm)
AIS_N_AQ_END         = 1.3700      # upper bound  (moderately contaminated water)
AIS_N_POINTS         = 200         # number of sweep points
AIS_LAM0_NM          = 1550.0      # [nm]  centre wavelength
AIS_DELTA_LAM_NM     = 5.0         # [nm]  half-span for central-difference ng
AIS_WG_WIDTH_NM      = 1000.0      # [nm]  core width  — WIDE face, along Y
AIS_WG_HEIGHT_NM     = 400.0       # [nm]  core height — THICKNESS, along Z
AIS_HIDE_GUI         = False       # True for headless / HPC runs
AIS_CSV_NAME         = f"AIS_R{AIS_BEND_RADIUS_UM:.2f}um_naq_sweep_{AIS_N_POINTS}pts"
_ais_lam0_m   = AIS_LAM0_NM   * 1e-9
_ais_lam_lo_m = (AIS_LAM0_NM - AIS_DELTA_LAM_NM) * 1e-9
_ais_lam_hi_m = (AIS_LAM0_NM + AIS_DELTA_LAM_NM) * 1e-9
_ais_dlam_m   = 2.0 * AIS_DELTA_LAM_NM * 1e-9   # denominator for central diff
_ais_n_aq_arr = np.linspace(AIS_N_AQ_START, AIS_N_AQ_END, AIS_N_POINTS)
_ais_N        = AIS_N_POINTS
_ais_R_m      = AIS_BEND_RADIUS_UM * 1e-6
_ais_wg_w_m   = AIS_WG_WIDTH_NM  * 1e-9    # 1000 nm → 1e-6 m
_ais_wg_h_m   = AIS_WG_HEIGHT_NM * 1e-9    # 400  nm → 4e-7 m
_ais_core_t_um   = AIS_WG_HEIGHT_NM * 1e-3        # 0.400 µm
_ais_half_t_um   = _ais_core_t_um / 2.0           # 0.200 µm
_ais_z_below_um  = SIM_Z_BELOW_UM                  # 2.000 µm  (from Cell 1)
_ais_z_above_um  = SIM_Z_ABOVE_UM                  # 2.000 µm
_ais_z_span_um   = _ais_z_below_um + _ais_core_t_um + _ais_z_above_um   # 4.400 µm
_ais_sio2_z_ctr  = -(_ais_half_t_um + _ais_z_below_um / 2.0)            # -1.100 µm
_ais_sio2_z_span = _ais_z_below_um                                        # 2.000 µm
_ais_z_ctr       = (_ais_z_above_um - _ais_z_below_um) / 2.0             # 0.000 µm
_ais_y_margin_um = 2.5
_ais_y_span_um   = AIS_WG_WIDTH_NM * 1e-3 + 2.0 * _ais_y_margin_um   # 6.0 µm
_ais_mesh_step_um = 6.5 / MESH_CELLS_Y          # physical mesh step [µm]
_ais_mesh_y       = int(np.ceil(_ais_y_span_um  / _ais_mesh_step_um))
_ais_mesh_z       = MESH_CELLS_Z                  # Z stack unchanged
_AIS_GROUP_KEY = (
    f"ais_R{AIS_BEND_RADIUS_UM:.4f}um"
    f"_w{AIS_WG_WIDTH_NM:.0f}nm"
    f"_h{AIS_WG_HEIGHT_NM:.0f}nm"
    f"_{_ais_N}pts"
    f"_naq{AIS_N_AQ_START:.4f}-{AIS_N_AQ_END:.4f}"
    f"_lam{AIS_LAM0_NM:.0f}nm"
)
_AIS_HDF5_GROUP = f"aqueous_index_sweep/{_AIS_GROUP_KEY}"

def _ais_build_fde(mode, n_aq: float, wavelength_m: float) -> None:
    """
    Build the bent-waveguide FDE cross-section for one (n_aq, λ) point.

    Stack (bottom → top, Z axis):
        SiO₂ substrate  →  SiN core 400 nm  →  Aqueous cladding n_aq

    Cross-section (Y-Z plane):
        Width  (Y axis) : AIS_WG_WIDTH_NM  = 1000 nm   (WIDE face)
        Height (Z axis) : AIS_WG_HEIGHT_NM = 400  nm   (THICKNESS)

    Bend geometry:
        bend_orientation = 0  →  curvature centre along +Y
        Ring lies in X-Y plane.
        Radius (pointing along +Y) is PARALLEL to the 1000 nm wide face.
        Radius is PERPENDICULAR to the 400 nm thickness sides (along Z).

    Structure override order (higher index wins over lower in overlap regions):
        1st (lowest priority) : AIS_bg        — aqueous background, full domain
        2nd                   : AIS_lower_clad — SiO₂ substrate slab
        3rd (highest priority): AIS_core       — SiN core rectangle
    """
    m = mode
    m.switchtolayout()
    m.selectall()
    m.delete()

    # ── FDE solver region ────────────────────────────────────────────────────
    m.addfde()
    m.set("solver type",            "2D X normal")
    m.set("x",                      0.0)
    m.set("y",                      0.0)
    m.set("z",                      _ais_z_ctr      * 1e-6)
    m.set("y span",                 _ais_y_span_um  * 1e-6)
    m.set("z span",                 _ais_z_span_um  * 1e-6)
    m.set("wavelength",             wavelength_m)
    m.set("number of trial modes",  N_MODES_REQUEST)
    m.set("mesh cells y",           _ais_mesh_y)
    m.set("mesh cells z",           _ais_mesh_z)
    # Bent waveguide settings:
    #   bend_orientation = 0  →  curvature centre in +Y direction
    #   This places the ring in the X-Y plane, with the radius vector
    #   running along +Y (parallel to the 1000 nm wide face of the waveguide).
    m.set("bent waveguide",         1)
    m.set("bend radius",            _ais_R_m)
    m.set("bend orientation",       0)

    # ── 1st structure : Aqueous background (full domain, lowest priority) ────
    m.addrect()
    m.set("name",    "AIS_bg")
    m.set("x",       0.0);  m.set("x span", 1.0e-6)
    m.set("y",       0.0);  m.set("y span", _ais_y_span_um  * 1e-6)
    m.set("z",       _ais_z_ctr     * 1e-6)
    m.set("z span",  _ais_z_span_um * 1e-6)
    m.set("material", "<Object defined dielectric>")
    m.set("index",    float(n_aq))          # ← varies with each sweep point

    # ── 2nd structure : SiO₂ lower substrate slab ────────────────────────────
    m.addrect()
    m.set("name",    "AIS_lower_clad")
    m.set("x",       0.0);  m.set("x span", 1.0e-6)
    m.set("y",       0.0);  m.set("y span", _ais_y_span_um   * 1e-6)
    m.set("z",       _ais_sio2_z_ctr  * 1e-6)
    m.set("z span",  _ais_sio2_z_span * 1e-6)
    m.set("material", "<Object defined dielectric>")
    m.set("index",    N_SIO2_FIXED)         # 1.4469, constant

    # ── 3rd structure : SiN waveguide core (highest priority) ────────────────
    m.addrect()
    m.set("name",    "AIS_core")
    m.set("x",       0.0);  m.set("x span", 1.0e-6)
    m.set("y",       0.0);  m.set("y span", _ais_wg_w_m)     # 1000 nm
    m.set("z",       0.0);  m.set("z span", _ais_wg_h_m)     # 400 nm
    m.set("material", "<Object defined dielectric>")
    m.set("index",    N_SIN_FIXED)           # 1.99, constant


def _ais_solve_neff(mode, n_aq: float, wavelength_m: float):
    """
    Single FDE solve at (n_aq, wavelength_m).

    Returns
    -------
    neff_re : float   Re(neff) of mode 1
    te_frac : float   TE polarisation fraction of mode 1

    Raises the lumapi exception upward so the caller can log and skip.
    """
    _ais_build_fde(mode, n_aq, wavelength_m)
    mode.run()
    mode.findmodes()
    raw_neff = mode.getdata("FDE::data::mode1", "neff")
    raw_te   = mode.getdata("FDE::data::mode1", "TE polarization fraction")
    neff_c   = complex(np.asarray(raw_neff).flat[0])
    te_v     = float(np.asarray(raw_te).flat[0])
    return neff_c.real, te_v


def _ais_compute_neff_ng(mode, n_aq: float):
    """
    Three-point central-difference group index at λ₀ for a given n_aq.

    Formula (Bogaerts Eq. 10):
        ng = neff(λ₀) − λ₀ · [neff(λ₀+Δλ) − neff(λ₀−Δλ)] / (2·Δλ)

    Returns
    -------
    neff_0  : float   neff at λ₀
    ng      : float   group index at λ₀
    te_frac : float   TE fraction at λ₀
    neff_lo : float   neff at λ₀ − Δλ  (raw stencil data)
    neff_hi : float   neff at λ₀ + Δλ
    """
    neff_lo, _      = _ais_solve_neff(mode, n_aq, _ais_lam_lo_m)
    neff_0,  te_frac = _ais_solve_neff(mode, n_aq, _ais_lam0_m)
    neff_hi, _      = _ais_solve_neff(mode, n_aq, _ais_lam_hi_m)
    dneff_dlam      = (neff_hi - neff_lo) / _ais_dlam_m
    ng              = neff_0 - _ais_lam0_m * dneff_dlam
    return neff_0, ng, te_frac, neff_lo, neff_hi


def _ais_init_hdf5(hf) -> None:
    """
    Pre-allocate all datasets for this sweep inside an already-open h5py file
    (opened in 'a' or 'r+' mode).

    Structure mirrors the ring_radius_sweep group from Cell 4 exactly:
      aqueous_index_sweep/<group_key>/
        metadata/
          attrs : sweep parameters, material constants, timestamps
          datasets : n_aq_array [N],  lam_stencil_nm [3]
        results/
          neff     [N]  float64   Re(neff) at λ₀
          ng       [N]  float64   group index at λ₀  (central difference)
          te_frac  [N]  float64   TE fraction at λ₀
          neff_lo  [N]  float64   neff at λ₀ − Δλ  (stencil raw)
          neff_hi  [N]  float64   neff at λ₀ + Δλ  (stencil raw)
        flags/
          computed [N]  bool
    """
    g  = hf.require_group(_AIS_HDF5_GROUP)
    mg = g.require_group("metadata")

    # Attributes — all required for scientific reproducibility
    for k, v in {
        "bend_radius_um":    AIS_BEND_RADIUS_UM,
        "bend_orientation":  0,               # curvature centre along +Y
        "wg_width_nm":       AIS_WG_WIDTH_NM,
        "wg_height_nm":      AIS_WG_HEIGHT_NM,
        "n_aq_start":        AIS_N_AQ_START,
        "n_aq_end":          AIS_N_AQ_END,
        "n_points":          _ais_N,
        "lam0_nm":           AIS_LAM0_NM,
        "delta_lam_nm":      AIS_DELTA_LAM_NM,
        "n_SiN":             N_SIN_FIXED,
        "n_SiO2":            N_SIO2_FIXED,
        "guided_mode_cutoff":N_SIO2_FIXED,    # Re(neff) > N_SIO2_FIXED → guided
        "version_name":      VERSION_NAME,
        "timestamp_start":   datetime.now().isoformat(),
    }.items():
        mg.attrs.setdefault(k, v)

    # Coordinate datasets
    if "n_aq_array" not in mg:
        mg.create_dataset("n_aq_array", data=_ais_n_aq_arr)
    if "lam_stencil_nm" not in mg:
        mg.create_dataset("lam_stencil_nm",
                          data=np.array([AIS_LAM0_NM - AIS_DELTA_LAM_NM,
                                         AIS_LAM0_NM,
                                         AIS_LAM0_NM + AIS_DELTA_LAM_NM]))

    # Result datasets — NaN-initialised, one chunk = full array
    _nan = np.full(_ais_N, np.nan, dtype=np.float64)
    rg = g.require_group("results")
    for ds_name in ("neff", "ng", "te_frac", "neff_lo", "neff_hi"):
        if ds_name not in rg:
            rg.create_dataset(ds_name, data=_nan.copy(), chunks=(_ais_N,))

    # Progress flag
    fg = g.require_group("flags")
    if "computed" not in fg:
        fg.create_dataset("computed",
                          data=np.zeros(_ais_N, dtype=bool),
                          chunks=(_ais_N,))


def _ais_load_cache(hf):
    """
    Read all result arrays for this sweep from an open h5py file.
    Returns (neff, ng, te_frac, neff_lo, neff_hi, computed) as np.ndarray.
    """
    g  = hf[_AIS_HDF5_GROUP]
    rg = g["results"]
    fg = g["flags"]
    return (
        rg["neff"]   [:].copy(),
        rg["ng"]     [:].copy(),
        rg["te_frac"][:].copy(),
        rg["neff_lo"][:].copy(),
        rg["neff_hi"][:].copy(),
        fg["computed"][:].copy(),
    )

def run(state=None):
    """Execute this step. `state` carries the shared namespace
    between steps (the notebook's old kernel globals + bridge).
    Returns the updated `state`."""
    state = {} if state is None else state
    global AIS_BEND_RADIUS_UM, AIS_CSV_NAME, AIS_DELTA_LAM_NM, AIS_HIDE_GUI, AIS_LAM0_NM, AIS_N_AQ_END, AIS_N_AQ_START, AIS_N_POINTS, AIS_WG_HEIGHT_NM, AIS_WG_WIDTH_NM, _AIS_GROUP_KEY, _AIS_HDF5_GROUP, _C_NEFF, _C_NG, _C_SENS, _C_TE, _SEP, _S_lam_mean, _S_neff_mean, _S_ng_mean, _ais_N, _ais_R_m, _ais_computed, _ais_core_t_um, _ais_csv_path, _ais_dlam_m, _ais_elapsed_total, _ais_half_t_um, _ais_hf, _ais_lam0_m, _ais_lam_hi_m, _ais_lam_lo_m, _ais_mesh_step_um, _ais_mesh_y, _ais_mesh_z, _ais_mode, _ais_n_aq_arr, _ais_n_cached, _ais_neff_arr, _ais_neff_hi_arr, _ais_neff_lo_arr, _ais_ng_arr, _ais_remaining, _ais_rg, _ais_runs_done, _ais_sio2_z_ctr, _ais_sio2_z_span, _ais_t0, _ais_te_arr, _ais_valid, _ais_wg_h_m, _ais_wg_w_m, _ais_y_margin_um, _ais_y_span_um, _ais_z_above_um, _ais_z_below_um, _ais_z_ctr, _ais_z_span_um, _ax00_ann, _ax01_ann, _elapsed, _eta, _ext, _fig1_stem, _fig2_stem, _hdr, _i, _labels, _lines, _ln_neff, _ln_ng, _n_aq, _neff_v, _ng_v, _nhi_v, _nlo_v, _rate, _te_v, _v_S_lam_pm_riu, _v_S_neff, _v_S_ng, _v_n_aq, _v_neff, _v_ng, _v_te, _Δn_aq, _Δneff, _Δng, ais_S_lam_pm_RIU, ais_S_neff, ais_S_ng, ais_n_aq_valid, ais_neff, ais_ng, ais_results_df, ais_te_frac, ax00, ax01, ax10, ax11, ax_neff, ax_ng, axes1, fig1, fig2
    globals()['lumapi'] = import_lumapi()
    globals().update(state)

    print("=" * 68)
    print("  Sensor Ring — Aqueous Index Sweep (Toxic Agent Sensing)")
    print("=" * 68)
    print(f"  Bend radius     : {AIS_BEND_RADIUS_UM:.4f} µm  (orientation: +Y, ring in X-Y plane)")
    print(f"  Width × height  : {AIS_WG_WIDTH_NM:.0f} nm × {AIS_WG_HEIGHT_NM:.0f} nm  (Y × Z)")
    print(f"  n_SiN           : {N_SIN_FIXED}   n_SiO₂   = {N_SIO2_FIXED}  (fixed)")
    print(f"  n_aq sweep      : {AIS_N_AQ_START:.4f} → {AIS_N_AQ_END:.4f}  ({_ais_N} pts,  "
          f"Δn = {(AIS_N_AQ_END - AIS_N_AQ_START) / (_ais_N - 1):.6f}/step)")
    print(f"  λ₀              : {AIS_LAM0_NM:.1f} nm   ng stencil ±{AIS_DELTA_LAM_NM:.1f} nm")
    print(f"  FDE domain      : y = {_ais_y_span_um:.2f} µm ({_ais_mesh_y} cells)  "
          f"z = {_ais_z_span_um:.2f} µm ({_ais_mesh_z} cells)")
    print(f"  Total FDE runs  : {_ais_N} × 3 = {_ais_N * 3}  (if uncached)")
    print(f"  HDF5 group      : {_AIS_HDF5_GROUP}")
    print(f"  HDF5 file       : {HDF5_PATH}")
    print(f"  CSV output      : {DATA_DIR / AIS_CSV_NAME}.csv")
    print("=" * 68)
    _ais_neff_arr    = np.full(_ais_N, np.nan, dtype=np.float64)
    _ais_ng_arr      = np.full(_ais_N, np.nan, dtype=np.float64)
    _ais_te_arr      = np.full(_ais_N, np.nan, dtype=np.float64)
    _ais_neff_lo_arr = np.full(_ais_N, np.nan, dtype=np.float64)
    _ais_neff_hi_arr = np.full(_ais_N, np.nan, dtype=np.float64)
    _ais_computed    = np.zeros(_ais_N, dtype=bool)
    _ais_hf = h5py.File(HDF5_PATH, "a")   # 'a' = read/write, create if absent
    if _AIS_HDF5_GROUP in _ais_hf:
        log.info(f"AIS cache found → {_AIS_HDF5_GROUP}")
        (
            _ais_neff_arr,
            _ais_ng_arr,
            _ais_te_arr,
            _ais_neff_lo_arr,
            _ais_neff_hi_arr,
            _ais_computed,
        ) = _ais_load_cache(_ais_hf)
        _ais_n_cached  = int(_ais_computed.sum())
        _ais_remaining = _ais_N - _ais_n_cached
        log.info(f"Cached: {_ais_n_cached}/{_ais_N}  |  Remaining: {_ais_remaining}")
    else:
        log.info(f"No AIS cache found — initialising group: {_AIS_HDF5_GROUP}")
        _ais_init_hdf5(_ais_hf)
        _ais_hf.flush()
        _ais_n_cached  = 0
        _ais_remaining = _ais_N
    _hdr = (f"{'n_aq':>10}  {'neff':>12}  {'ng':>12}  "
            f"{'TE frac':>9}  {'neff_lo':>12}  {'neff_hi':>12}  {'src':>6}")
    print(f"\n{_hdr}")
    print("─" * len(_hdr))
    for _i in range(_ais_N):
        if _ais_computed[_i]:
            print(f"  {_ais_n_aq_arr[_i]:.6f}  {_ais_neff_arr[_i]:.8f}  "
                  f"{_ais_ng_arr[_i]:.8f}  {_ais_te_arr[_i]:.6f}  "
                  f"{_ais_neff_lo_arr[_i]:.8f}  {_ais_neff_hi_arr[_i]:.8f}  cache")
    if _ais_remaining > 0:
        _ais_runs_done = 0
        _ais_t0 = time.time()
        log.info(f"Launching Lumerical MODE session  ({_ais_remaining} points remaining) …")
        _ais_mode = lumapi.MODE(hide=AIS_HIDE_GUI)

        try:
            for _i, _n_aq in enumerate(_ais_n_aq_arr):

                # ── Cache check ────────────────────────────────────────────────────
                if _ais_computed[_i]:
                    continue

                # ── Three-point stencil solve ──────────────────────────────────────
                try:
                    _neff_v, _ng_v, _te_v, _nlo_v, _nhi_v = \
                        _ais_compute_neff_ng(_ais_mode, _n_aq)
                except Exception as _exc:
                    log.warning(f"  n_aq = {_n_aq:.6f}  FAILED: {_exc}")
                    # Mark as done with NaN so a re-run skips this point cleanly
                    _ais_computed[_i] = True
                    _ais_hf[f"{_AIS_HDF5_GROUP}/flags/computed"][_i] = True
                    _ais_hf.flush()
                    continue

                # ── Store in memory ────────────────────────────────────────────────
                _ais_neff_arr   [_i] = _neff_v
                _ais_ng_arr     [_i] = _ng_v
                _ais_te_arr     [_i] = _te_v
                _ais_neff_lo_arr[_i] = _nlo_v
                _ais_neff_hi_arr[_i] = _nhi_v
                _ais_computed   [_i] = True

                # ── Write to HDF5 immediately — fault-safe incremental storage ─────
                _ais_rg = _ais_hf[f"{_AIS_HDF5_GROUP}/results"]
                _ais_rg["neff"]   [_i] = _neff_v
                _ais_rg["ng"]     [_i] = _ng_v
                _ais_rg["te_frac"][_i] = _te_v
                _ais_rg["neff_lo"][_i] = _nlo_v
                _ais_rg["neff_hi"][_i] = _nhi_v
                _ais_hf[f"{_AIS_HDF5_GROUP}/flags/computed"][_i] = True
                _ais_hf.flush()   # guarantees data survives a crash

                _ais_runs_done += 1

                # ── Live table row ─────────────────────────────────────────────────
                print(f"  {_n_aq:.6f}  {_neff_v:.8f}  {_ng_v:.8f}  "
                      f"{_te_v:.6f}  {_nlo_v:.8f}  {_nhi_v:.8f}  FDE")

                # ── Progress log every 10 new solves ──────────────────────────────
                if _ais_runs_done % 10 == 0 or _ais_runs_done == _ais_remaining:
                    _elapsed = time.time() - _ais_t0
                    _rate    = _ais_runs_done / _elapsed if _elapsed > 0 else 1e-9
                    _eta     = (_ais_remaining - _ais_runs_done) / _rate
                    log.info(
                        f"  [{_ais_runs_done:3d}/{_ais_remaining}]  "
                        f"n_aq = {_n_aq:.6f}  "
                        f"neff = {_neff_v:.6f}  "
                        f"ng = {_ng_v:.6f}  "
                        f"{_rate/3:.2f} n_aq pts/s  "    # ÷3 because 3 FDE/point
                        f"ETA {_eta:.0f} s"
                    )

        finally:
            _ais_mode.close()
            _ais_elapsed_total = time.time() - _ais_t0
            log.info(
                f"MODE session closed  "
                f"({_ais_runs_done} new points in {_ais_elapsed_total:.1f} s,  "
                f"avg {_ais_elapsed_total / max(_ais_runs_done * 3, 1):.2f} s/FDE)"
            )
    _ais_hf[_AIS_HDF5_GROUP]["metadata"].attrs["timestamp_end"]  = \
        datetime.now().isoformat()
    _ais_hf[_AIS_HDF5_GROUP]["metadata"].attrs["runs_completed"] = \
        int(_ais_computed.sum())
    _ais_hf.flush()
    _ais_hf.close()
    log.info(f"HDF5 closed → {HDF5_PATH}")
    _ais_valid = ~np.isnan(_ais_neff_arr)
    if not np.any(_ais_valid):
        raise RuntimeError(
            "All FDE runs failed. Check WARNING lines above for the specific error. "
            "Common causes: incorrect lumapi path, Lumerical license not available, "
            "or bend radius outside the physical guided-mode regime."
        )
    _v_n_aq  = _ais_n_aq_arr[_ais_valid]   # valid aqueous index values
    _v_neff  = _ais_neff_arr[_ais_valid]   # corresponding neff
    _v_ng    = _ais_ng_arr  [_ais_valid]   # corresponding ng
    _v_te    = _ais_te_arr  [_ais_valid]   # TE fraction
    _v_S_neff = np.gradient(_v_neff, _v_n_aq)   # dneff / dn_aq   [RIU⁻¹]
    _v_S_ng   = np.gradient(_v_ng,   _v_n_aq)   # dng   / dn_aq   [RIU⁻¹]
    _v_S_lam_pm_riu = (_ais_lam0_m * 1e9 / _v_ng) * _v_S_neff * 1e3   # [pm/RIU]
    ais_results_df = pd.DataFrame({
        "n_aq":           _v_n_aq,
        "neff":           _v_neff,
        "ng":             _v_ng,
        "te_frac":        _v_te,
        "S_neff_RIU":     _v_S_neff,     # dneff/dn_aq  [RIU⁻¹]
        "S_ng_RIU":       _v_S_ng,       # dng  /dn_aq  [RIU⁻¹]
        "S_lam_pm_RIU":   _v_S_lam_pm_riu,  # resonance shift sensitivity [pm/RIU]
    })
    _ais_csv_path = DATA_DIR / f"{AIS_CSV_NAME}.csv"
    ais_results_df.to_csv(str(_ais_csv_path), index=False, float_format="%.10f")
    log.info(f"CSV saved → {_ais_csv_path}")
    _Δn_aq    = AIS_N_AQ_END - AIS_N_AQ_START
    _Δneff    = float(_v_neff[-1] - _v_neff[0])
    _Δng      = float(_v_ng[-1]   - _v_ng[0])
    _S_neff_mean = float(np.mean(_v_S_neff))
    _S_ng_mean   = float(np.mean(_v_S_ng))
    _S_lam_mean  = float(np.mean(_v_S_lam_pm_riu))
    print("\n" + "=" * 70)
    print("  AQUEOUS INDEX SWEEP — SUMMARY")
    print("=" * 70)
    print(f"  Bend radius       : R = {AIS_BEND_RADIUS_UM:.4f} µm")
    print(f"  Waveguide         : {AIS_WG_WIDTH_NM:.0f} nm × {AIS_WG_HEIGHT_NM:.0f} nm  (w × h)")
    print(f"  n_aq range        : {AIS_N_AQ_START:.4f} → {AIS_N_AQ_END:.4f}  (Δ = {_Δn_aq:.4f} RIU)")
    print(f"  Points computed   : {int(_ais_valid.sum())} / {_ais_N}")
    print(f"  neff range        : {float(_v_neff.min()):.8f} → {float(_v_neff.max()):.8f}")
    print(f"    Δneff total      : {_Δneff:+.8f}")
    print(f"    mean dneff/dn_aq : {_S_neff_mean:.6f}  RIU⁻¹")
    print(f"  ng range          : {float(_v_ng.min()):.8f} → {float(_v_ng.max()):.8f}")
    print(f"    Δng total        : {_Δng:+.8f}")
    print(f"    mean dng/dn_aq   : {_S_ng_mean:.6f}  RIU⁻¹")
    print(f"  Resonance shift   : mean Δλ/Δn = {_S_lam_mean:.1f} pm/RIU  @ λ₀={AIS_LAM0_NM:.0f} nm")
    print("=" * 70)
    _SEP = "─" * 108
    print(f"\n  {'#':>5}  {'n_aq':>10}  {'neff':>14}  {'ng':>14}  "
          f"{'TE frac':>9}  {'S_neff [1/RIU]':>15}  {'S_ng [1/RIU]':>14}  "
          f"{'S_λ [pm/RIU]':>14}")
    print(f"  {_SEP}")
    for _i in range(len(_v_n_aq)):
        print(
            f"  {_i+1:5d}  "
            f"{_v_n_aq[_i]:10.6f}  "
            f"{_v_neff[_i]:14.10f}  "
            f"{_v_ng[_i]:14.10f}  "
            f"{_v_te[_i]:9.6f}  "
            f"{_v_S_neff[_i]:15.8f}  "
            f"{_v_S_ng[_i]:14.8f}  "
            f"{_v_S_lam_pm_riu[_i]:14.4f}"
        )
    print(f"  {_SEP}")
    print(f"\n  CSV saved to : {_ais_csv_path}")
    plt.rcParams.update({
        "figure.dpi":        150,
        "font.family":       "DejaVu Sans",
        "axes.titlesize":    12,
        "axes.labelsize":    11,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
        "lines.linewidth":   2.0,
        "lines.markersize":  4,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.30,
        "grid.linestyle":    "--",
    })
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
    fig1.suptitle(
        f"Aqueous Index Sweep — Sensor Ring Bend Mode Analysis\n"
        f"SiN {AIS_WG_HEIGHT_NM:.0f} nm × {AIS_WG_WIDTH_NM:.0f} nm  │  "
        f"R = {AIS_BEND_RADIUS_UM:.4f} µm  │  "
        f"bend_orientation = 0  (radius ‖ wide face, Y axis)  │  "
        f"λ₀ = {AIS_LAM0_NM:.0f} nm",
        fontsize=11, fontweight="bold", y=1.01,
    )
    _C_NEFF = "#0072B2"    # blue
    _C_NG   = "#D55E00"    # vermilion
    _C_SENS = "#009E73"    # green
    _C_TE   = "#CC79A7"    # pink
    ax00 = axes1[0, 0]
    ax00.plot(_v_n_aq, _v_neff, color=_C_NEFF, lw=2.2, label=r"$n_\mathrm{eff}$")
    ax00.set_xlabel(r"Aqueous refractive index  $n_\mathrm{aq}$")
    ax00.set_ylabel(r"Effective index  $n_\mathrm{eff}$")
    ax00.set_title(r"Effective Index vs $n_\mathrm{aq}$")
    ax00.legend(loc="upper left")
    ax00.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax00.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    _ax00_ann = (
        f"Δneff = {_Δneff:+.6f}\n"
        f"over Δn_aq = {_Δn_aq:.4f} RIU"
    )
    ax00.text(0.97, 0.06, _ax00_ann, transform=ax00.transAxes,
              ha="right", va="bottom", fontsize=8.5,
              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=_C_NEFF, alpha=0.90, lw=0.8))
    ax01 = axes1[0, 1]
    ax01.plot(_v_n_aq, _v_ng, color=_C_NG, lw=2.2, label=r"$n_g$")
    ax01.set_xlabel(r"Aqueous refractive index  $n_\mathrm{aq}$")
    ax01.set_ylabel(r"Group index  $n_g$")
    ax01.set_title(r"Group Index vs $n_\mathrm{aq}$")
    ax01.legend(loc="upper left")
    ax01.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax01.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    _ax01_ann = (
        f"Δng = {_Δng:+.6f}\n"
        f"over Δn_aq = {_Δn_aq:.4f} RIU"
    )
    ax01.text(0.97, 0.06, _ax01_ann, transform=ax01.transAxes,
              ha="right", va="bottom", fontsize=8.5,
              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=_C_NG, alpha=0.90, lw=0.8))
    ax10 = axes1[1, 0]
    ax10.plot(_v_n_aq, _v_S_neff, color=_C_NEFF, lw=2.0,
              label=r"$\partial n_\mathrm{eff}/\partial n_\mathrm{aq}$")
    ax10.plot(_v_n_aq, _v_S_ng,   color=_C_NG,   lw=2.0, ls="--",
              label=r"$\partial n_g/\partial n_\mathrm{aq}$")
    ax10.axhline(0, color="gray", lw=0.8, ls=":", alpha=0.6)
    ax10.set_xlabel(r"Aqueous refractive index  $n_\mathrm{aq}$")
    ax10.set_ylabel(r"Sensitivity  [RIU$^{-1}$]")
    ax10.set_title(r"Modal Sensitivity  $\partial n/\partial n_\mathrm{aq}$")
    ax10.legend(loc="upper left")
    ax10.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax10.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax11 = axes1[1, 1]
    ax11.plot(_v_n_aq, _v_S_lam_pm_riu, color=_C_SENS, lw=2.2,
              label=r"$\partial\lambda_\mathrm{res}/\partial n_\mathrm{aq}$")
    ax11.axhline(_S_lam_mean, color=_C_SENS, lw=1.2, ls="--", alpha=0.6,
                 label=f"mean = {_S_lam_mean:.1f} pm/RIU")
    ax11.set_xlabel(r"Aqueous refractive index  $n_\mathrm{aq}$")
    ax11.set_ylabel(r"Resonance shift sensitivity  [pm/RIU]")
    ax11.set_title(r"Sensing Sensitivity  $\partial\lambda_\mathrm{res}/\partial n_\mathrm{aq}$  @ $\lambda_0$ = "
                   f"{AIS_LAM0_NM:.0f} nm")
    ax11.legend(loc="upper left")
    ax11.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax11.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig1.tight_layout()
    _fig1_stem = DATA_DIR / f"{VERSION_NAME}_{AIS_CSV_NAME}_overview"
    for _ext in (".png", ".pdf"):
        fig1.savefig(str(_fig1_stem) + _ext, dpi=150, bbox_inches="tight")
        print(f"  Saved → {str(_fig1_stem) + _ext}")
    plt.show()
    fig2, ax_neff = plt.subplots(figsize=(9, 5.5))
    ax_ng = ax_neff.twinx()
    _ln_neff, = ax_neff.plot(_v_n_aq, _v_neff, color=_C_NEFF, lw=2.4,
                              label=r"$n_\mathrm{eff}$  (left axis)")
    _ln_ng,   = ax_ng.plot  (_v_n_aq, _v_ng,   color=_C_NG,   lw=2.4, ls="--",
                              label=r"$n_g$  (right axis)")
    ax_neff.set_xlabel(r"Aqueous cladding refractive index  $n_\mathrm{aq}$",
                       fontsize=11)
    ax_neff.set_ylabel(r"Effective index  $n_\mathrm{eff}$",
                       color=_C_NEFF, fontsize=11)
    ax_ng.set_ylabel(r"Group index  $n_g$",
                     color=_C_NG, fontsize=11)
    ax_neff.tick_params(axis="y", colors=_C_NEFF)
    ax_ng.tick_params(axis="y", colors=_C_NG)
    ax_neff.set_title(
        f"Effective and Group Index vs Aqueous Index\n"
        f"SiN {AIS_WG_HEIGHT_NM:.0f} nm × {AIS_WG_WIDTH_NM:.0f} nm  │  "
        f"R = {AIS_BEND_RADIUS_UM:.4f} µm  │  λ₀ = {AIS_LAM0_NM:.0f} nm",
        fontsize=11, fontweight="bold",
    )
    ax_neff.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax_neff.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax_ng.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    _lines  = [_ln_neff, _ln_ng]
    _labels = [_l.get_label() for _l in _lines]
    ax_neff.legend(_lines, _labels, loc="upper left", framealpha=0.90)
    ax_neff.grid(True, alpha=0.30, linestyle="--")
    fig2.tight_layout()
    _fig2_stem = DATA_DIR / f"{VERSION_NAME}_{AIS_CSV_NAME}_neff_ng_combined"
    for _ext in (".png", ".pdf"):
        fig2.savefig(str(_fig2_stem) + _ext, dpi=150, bbox_inches="tight")
        print(f"  Saved → {str(_fig2_stem) + _ext}")
    plt.show()
    ais_n_aq_valid      = _v_n_aq             # aqueous index values      [N_valid]
    ais_neff            = _v_neff             # effective index            [N_valid]
    ais_ng              = _v_ng               # group index                [N_valid]
    ais_te_frac         = _v_te               # TE polarisation fraction   [N_valid]
    ais_S_neff          = _v_S_neff           # dneff/dn_aq  [RIU⁻¹]       [N_valid]
    ais_S_ng            = _v_S_ng             # dng  /dn_aq  [RIU⁻¹]       [N_valid]
    ais_S_lam_pm_RIU    = _v_S_lam_pm_riu     # dλ_res/dn_aq [pm/RIU]      [N_valid]
    print("\n  Variables exported to the next cell:")
    print(f"    ais_n_aq_valid     shape = {ais_n_aq_valid.shape}  (valid aqueous index values)")
    print(f"    ais_neff           shape = {ais_neff.shape}  (effective index)")
    print(f"    ais_ng             shape = {ais_ng.shape}  (group index)")
    print(f"    ais_te_frac        shape = {ais_te_frac.shape}  (TE fraction)")
    print(f"    ais_S_neff         shape = {ais_S_neff.shape}  (dneff/dn_aq  [RIU⁻¹])")
    print(f"    ais_S_ng           shape = {ais_S_ng.shape}  (dng/dn_aq    [RIU⁻¹])")
    print(f"    ais_S_lam_pm_RIU   shape = {ais_S_lam_pm_RIU.shape}  (dλ_res/dn_aq [pm/RIU])")
    print(f"    ais_results_df     Pandas DataFrame ({len(ais_results_df)} rows × {len(ais_results_df.columns)} columns)")
    print(f"\n  CSV saved to : {_ais_csv_path}")
    print(f"  HDF5 group   : {_AIS_HDF5_GROUP}")
    print(f"  HDF5 file    : {HDF5_PATH}")

    state.update({k: globals().get(k) for k in [
        'AIS_BEND_RADIUS_UM', 'AIS_CSV_NAME', 'AIS_DELTA_LAM_NM', 'AIS_HIDE_GUI', 'AIS_LAM0_NM', 'AIS_N_AQ_END',
        'AIS_N_AQ_START', 'AIS_N_POINTS', 'AIS_WG_HEIGHT_NM', 'AIS_WG_WIDTH_NM', '_AIS_GROUP_KEY', '_AIS_HDF5_GROUP',
        '_C_NEFF', '_C_NG', '_C_SENS', '_C_TE', '_SEP', '_S_lam_mean',
        '_S_neff_mean', '_S_ng_mean', '_ais_N', '_ais_R_m', '_ais_computed', '_ais_core_t_um',
        '_ais_csv_path', '_ais_dlam_m', '_ais_elapsed_total', '_ais_half_t_um', '_ais_hf', '_ais_lam0_m',
        '_ais_lam_hi_m', '_ais_lam_lo_m', '_ais_mesh_step_um', '_ais_mesh_y', '_ais_mesh_z', '_ais_mode',
        '_ais_n_aq_arr', '_ais_n_cached', '_ais_neff_arr', '_ais_neff_hi_arr', '_ais_neff_lo_arr', '_ais_ng_arr',
        '_ais_remaining', '_ais_rg', '_ais_runs_done', '_ais_sio2_z_ctr', '_ais_sio2_z_span', '_ais_t0',
        '_ais_te_arr', '_ais_valid', '_ais_wg_h_m', '_ais_wg_w_m', '_ais_y_margin_um', '_ais_y_span_um',
        '_ais_z_above_um', '_ais_z_below_um', '_ais_z_ctr', '_ais_z_span_um', '_ax00_ann', '_ax01_ann',
        '_elapsed', '_eta', '_ext', '_fig1_stem', '_fig2_stem', '_hdr',
        '_i', '_labels', '_lines', '_ln_neff', '_ln_ng', '_n_aq',
        '_neff_v', '_ng_v', '_nhi_v', '_nlo_v', '_rate', '_te_v',
        '_v_S_lam_pm_riu', '_v_S_neff', '_v_S_ng', '_v_n_aq', '_v_neff', '_v_ng',
        '_v_te', '_Δn_aq', '_Δneff', '_Δng', 'ais_S_lam_pm_RIU', 'ais_S_neff',
        'ais_S_ng', 'ais_n_aq_valid', 'ais_neff', 'ais_ng', 'ais_results_df', 'ais_te_frac',
        'ax00', 'ax01', 'ax10', 'ax11', 'ax_neff', 'ax_ng',
        'axes1', 'fig1', 'fig2',
    ] if k in globals()})
    return state


if __name__ == "__main__":
    run()
