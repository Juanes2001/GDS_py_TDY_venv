"""
step3_ring_radius.py — sensor-ring radius sweep for FSR matching.

Bent-waveguide FDE radius sweep that finds the sensor-ring radius giving
the target 10 nm FSR. New HDF5 group inside the existing cache file.
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

from .config import *  # shared platform constants & paths
from .lumerical_session import import_lumapi
from .plotting import apply_style, save_fig, make_colorbar
from . import plotting as _plot

# ── Data-contract inputs (injected by main.py via `state`) ──────────
_exc = None

apply_style()

# ─────────────────────────────────────────────────────────
#  Module constants / design knobs for this step
# ─────────────────────────────────────────────────────────
FWHM_SENSOR_NM      = 0.5        # [nm]  sensor ring target FWHM
FWHM_SPEC_NM        = 0.5        # [nm]  spectrometer rings default FWHM
RR_FSR_NM      = 10.0      # [nm]  target free spectral range
RR_LAM0_NM     = 1550.0    # [nm]  resonance wavelength
RR_WG_WIDTH_NM = 1000.0    # [nm]  waveguide width  (height = CORE_THICKNESS_UM)
RR_R_MIN_UM  = 18.0    # [µm]
RR_R_MAX_UM  = 20.0   # [µm]
RR_N_RADII   = 100
_RR_DELTA_LAM_NM = 5.0   # [nm]  half-span for central difference
_RR_GROUP_KEY = (
    f"rr_{RR_FSR_NM:.0f}nm_{RR_LAM0_NM:.0f}nm_"
    f"{RR_WG_WIDTH_NM:.0f}nm_"
    f"{RR_R_MIN_UM:.1f}-{RR_R_MAX_UM:.1f}um_{RR_N_RADII}pts"
)
_RR_HDF5_GROUP = f"ring_radius_sweep/{_RR_GROUP_KEY}"
_lam0_m   = RR_LAM0_NM * 1e-9
_lam_lo_m = (RR_LAM0_NM - _RR_DELTA_LAM_NM) * 1e-9
_lam_hi_m = (RR_LAM0_NM + _RR_DELTA_LAM_NM) * 1e-9
_dlam_m   = 2.0 * _RR_DELTA_LAM_NM * 1e-9
_FSR_m    = RR_FSR_NM * 1e-9
_wg_w_m   = RR_WG_WIDTH_NM * 1e-9
_core_t_um = CORE_THICKNESS_UM
_target_ngL_m  = _lam0_m**2 / _FSR_m
_target_ngL_um = _target_ngL_m * 1e6
_sim_y_span_um  = SIM_Y_SPAN_UM
_sim_z_below_um = SIM_Z_BELOW_UM
_sim_z_above_um = SIM_Z_ABOVE_UM
_sim_z_span_um  = _sim_z_below_um + _core_t_um + _sim_z_above_um
_half_t_um      = _core_t_um / 2.0
_sio2_z_ctr_um  = -(_half_t_um + _sim_z_below_um / 2.0)
_sio2_z_span_um = _sim_z_below_um
_sim_z_ctr_um   = (_sim_z_above_um - _sim_z_below_um) / 2.0
_radii_um = np.linspace(RR_R_MIN_UM, RR_R_MAX_UM, RR_N_RADII)
_radii_m  = _radii_um * 1e-6
_N        = RR_N_RADII

def _rr_init_hdf5(hf) -> None:
    """
    Create ring_radius_sweep/<group_key>/ inside an already-open h5py file
    opened in 'r+' or 'a' mode.  Pre-allocates all datasets with NaN / False.
    Mirrors the structure of _init_hdf5 from Cell 2 exactly.
    """
    g = hf.require_group(_RR_HDF5_GROUP)

    # ── metadata ─────────────────────────────────────────────────────────────
    mg = g.require_group("metadata")
    mg.attrs["fsr_nm"]         = RR_FSR_NM
    mg.attrs["lam0_nm"]        = RR_LAM0_NM
    mg.attrs["wg_width_nm"]    = RR_WG_WIDTH_NM
    mg.attrs["wg_height_nm"]   = _core_t_um * 1e3
    mg.attrs["r_min_um"]       = RR_R_MIN_UM
    mg.attrs["r_max_um"]       = RR_R_MAX_UM
    mg.attrs["n_radii"]        = _N
    mg.attrs["delta_lam_nm"]   = _RR_DELTA_LAM_NM
    mg.attrs["n_SiN"]          = N_SIN_FIXED
    mg.attrs["n_SiO2"]         = N_SIO2_FIXED
    mg.attrs["n_upper_clad"]   = N_UPPER_CLADDING
    mg.attrs["version_name"]   = VERSION_NAME
    mg.attrs["timestamp_start"] = datetime.now().isoformat()

    # axis coordinate arrays
    if "radii_um" not in mg:
        mg.create_dataset("radii_um",       data=_radii_um)
    if "lam_stencil_nm" not in mg:
        mg.create_dataset("lam_stencil_nm",
                          data=np.array([RR_LAM0_NM - _RR_DELTA_LAM_NM,
                                         RR_LAM0_NM,
                                         RR_LAM0_NM + _RR_DELTA_LAM_NM]))

    # ── results — NaN-initialised, chunked per radius row ────────────────────
    rg = g.require_group("results")
    _nan = np.full(_N, np.nan, dtype=np.float64)
    for ds_name in ("neff", "ng", "te_frac", "ngL_um", "neff_lo", "neff_hi"):
        if ds_name not in rg:
            rg.create_dataset(ds_name, data=_nan.copy(), chunks=(_N,))

    # ── progress flag ─────────────────────────────────────────────────────────
    fg = g.require_group("flags")
    if "computed" not in fg:
        fg.create_dataset("computed",
                          data=np.zeros(_N, dtype=bool),
                          chunks=(_N,))


def _rr_load_cache(hf,
                   neff_arr, ng_arr, te_arr,
                   ngL_arr, neff_lo_arr, neff_hi_arr,
                   computed) -> int:
    """
    Read ring_radius_sweep/<group_key>/ from an open h5py file into the
    pre-allocated in-memory arrays.  Returns number of cached points.
    All arrays are modified in-place.
    """
    g  = hf[_RR_HDF5_GROUP]
    rg = g["results"]
    fg = g["flags"]

    neff_arr[:]    = rg["neff"][:]
    ng_arr[:]      = rg["ng"][:]
    te_arr[:]      = rg["te_frac"][:]
    ngL_arr[:]     = rg["ngL_um"][:] * 1e-6   # stored in µm, keep in m here
    neff_lo_arr[:] = rg["neff_lo"][:]
    neff_hi_arr[:] = rg["neff_hi"][:]
    computed[:]    = fg["computed"][:]

    return int(computed.sum())


def _rr_build_fde(mode, radius_m: float, wavelength_m: float) -> None:
    """Build bent waveguide cross-section + FDE solver for one (R, λ) point."""
    m = mode
    m.switchtolayout()
    m.selectall()
    m.delete()

    m.addfde()
    m.set("solver type",           "2D X normal")
    m.set("x",                     0.0)
    m.set("y",                     0.0)
    m.set("z",                     _sim_z_ctr_um  * 1e-6)
    m.set("y span",                _sim_y_span_um * 1e-6)
    m.set("z span",                _sim_z_span_um * 1e-6)
    m.set("wavelength",            wavelength_m)
    m.set("number of trial modes", N_MODES_REQUEST)
    m.set("mesh cells y",          MESH_CELLS_Y)
    m.set("mesh cells z",          MESH_CELLS_Z)
    m.set("bent waveguide",        1)
    m.set("bend radius",           radius_m)
    m.set("bend orientation",      0)

    m.addrect()
    m.set("name",    "RR_bg")
    m.set("x",       0.0);  m.set("x span", 1.0e-6)
    m.set("y",       0.0);  m.set("y span", _sim_y_span_um * 1e-6)
    m.set("z",       _sim_z_ctr_um  * 1e-6)
    m.set("z span",  _sim_z_span_um * 1e-6)
    m.set("material", "<Object defined dielectric>")
    m.set("index",    N_UPPER_CLADDING)

    m.addrect()
    m.set("name",    "RR_lower_clad")
    m.set("x",       0.0);  m.set("x span", 1.0e-6)
    m.set("y",       0.0);  m.set("y span", _sim_y_span_um  * 1e-6)
    m.set("z",       _sio2_z_ctr_um * 1e-6)
    m.set("z span",  _sio2_z_span_um * 1e-6)
    m.set("material", "<Object defined dielectric>")
    m.set("index",    N_SIO2_FIXED)

    m.addrect()
    m.set("name",    "RR_core")
    m.set("x",       0.0);  m.set("x span", 1.0e-6)
    m.set("y",       0.0);  m.set("y span", _wg_w_m)
    m.set("z",       0.0);  m.set("z span", _core_t_um * 1e-6)
    m.set("material", "<Object defined dielectric>")
    m.set("index",    N_SIN_FIXED)


def _rr_solve_neff(mode, radius_m: float, wavelength_m: float):
    """Single FDE solve → (Re(neff), TE fraction) for mode 1."""
    _rr_build_fde(mode, radius_m, wavelength_m)
    mode.run()
    mode.findmodes()
    raw_neff = mode.getdata("FDE::data::mode1", "neff")
    raw_te   = mode.getdata("FDE::data::mode1", "TE polarization fraction")
    neff_c   = complex(np.asarray(raw_neff).flat[0])
    te_v     = float(np.asarray(raw_te).flat[0])
    return neff_c.real, te_v


def _rr_neff_ng(mode, radius_m: float):
    """
    Three-point central-difference ng at lam0 for a given bend radius.
    Returns (neff_at_lam0, ng, te_frac, neff_lo, neff_hi).
    """
    neff_lo, _    = _rr_solve_neff(mode, radius_m, _lam_lo_m)
    neff_0,  te_v = _rr_solve_neff(mode, radius_m, _lam0_m)
    neff_hi, _    = _rr_solve_neff(mode, radius_m, _lam_hi_m)
    dneff_dlam    = (neff_hi - neff_lo) / _dlam_m
    ng            = neff_0 - _lam0_m * dneff_dlam
    return neff_0, ng, te_v, neff_lo, neff_hi

def run(state=None):
    """Execute this step. `state` carries the shared namespace
    between steps (the notebook's old kernel globals + bridge).
    Returns the updated `state`."""
    state = {} if state is None else state
    global FWHM_SENSOR_NM, FWHM_SPEC_NM, RR_FSR_NM, RR_LAM0_NM, RR_N_RADII, RR_R_MAX_UM, RR_R_MIN_UM, RR_WG_WIDTH_NM, _C_BEST, _C_NEFF, _C_NG, _C_NGL, _C_TGT, _FSR_m, _L_m, _N, _RR_DELTA_LAM_NM, _RR_GROUP_KEY, _RR_HDF5_GROUP, _R_m, _R_um, _R_v, _bi, _computed, _core_t_um, _delta, _dist, _dlam_m, _elapsed, _elapsed_total, _eta, _half_t_um, _hdr, _hf, _i, _lam0_m, _lam_hi_m, _lam_lo_m, _n_cached, _neff_arr, _neff_hi_arr, _neff_lo_arr, _neff_v, _ngL_arr, _ngL_m, _ngL_v, _ng_arr, _ng_v, _nhi, _nlo, _radii_m, _radii_um, _rate, _remaining, _rg, _rr_mode, _runs_done, _sim_y_span_um, _sim_z_above_um, _sim_z_below_um, _sim_z_ctr_um, _sim_z_span_um, _sio2_z_ctr_um, _sio2_z_span_um, _suffix, _t0, _target_ngL_m, _target_ngL_um, _te_arr, _te_v, _valid, _wg_w_m, ax1, ax2, ax3, fig1, fig2, fig3, rr_FSR_pred_nm, rr_best_L_um, rr_best_R_um, rr_best_neff, rr_best_ng, rr_best_ngL_um
    globals()['lumapi'] = import_lumapi()
    globals().update(state)

    print("=" * 65)
    print("  Ring Resonator — Radius Sweep for FSR Matching")
    print("=" * 65)
    print(f"  Target FSR           : {RR_FSR_NM:.2f} nm")
    print(f"  Resonance wavelength : {RR_LAM0_NM:.1f} nm")
    print(f"  Waveguide width      : {RR_WG_WIDTH_NM:.0f} nm")
    print(f"  Waveguide height     : {_core_t_um*1e3:.0f} nm")
    print(f"  n_core               : {N_SIN_FIXED}   (N_SIN_FIXED)")
    print(f"  n_lower_clad         : {N_SIO2_FIXED}  (N_SIO2_FIXED)")
    print(f"  n_upper_clad         : {N_UPPER_CLADDING}   (N_UPPER_CLADDING)")
    print(f"  Required ng·L        : {_target_ngL_um:.4f} µm")
    print(f"  ng stencil           : ±{_RR_DELTA_LAM_NM:.1f} nm  ({_N*3} FDE solves if uncached)")
    print(f"  Radius sweep         : {RR_R_MIN_UM:.1f} – {RR_R_MAX_UM:.1f} µm  ({_N} pts)")
    print(f"  HDF5 group           : {_RR_HDF5_GROUP}")
    print(f"  HDF5 file            : {HDF5_PATH}")
    print("=" * 65)
    _neff_arr    = np.full(_N, np.nan)
    _ng_arr      = np.full(_N, np.nan)
    _te_arr      = np.full(_N, np.nan)
    _ngL_arr     = np.full(_N, np.nan)   # [m]
    _neff_lo_arr = np.full(_N, np.nan)
    _neff_hi_arr = np.full(_N, np.nan)
    _computed    = np.zeros(_N, dtype=bool)
    _hf = h5py.File(HDF5_PATH, "a")   # 'a' = read/write, create if missing
    if _RR_HDF5_GROUP in _hf:
        log.info(f"Ring sweep cache found → {_RR_HDF5_GROUP}")
        _n_cached = _rr_load_cache(
            _hf,
            _neff_arr, _ng_arr, _te_arr,
            _ngL_arr, _neff_lo_arr, _neff_hi_arr,
            _computed,
        )
        _remaining = _N - _n_cached
        log.info(f"Cached: {_n_cached}/{_N}  |  Remaining: {_remaining}")
        if _remaining == 0:
            log.info("All radii already computed — skipping FDE entirely.")
    else:
        log.info(f"No ring sweep cache found — initialising group: {_RR_HDF5_GROUP}")
        _rr_init_hdf5(_hf)
        _hf.flush()
        _n_cached  = 0
        _remaining = _N
    _hdr = (f"{'Radius (µm)':>12}  {'neff':>10}  {'ng':>10}  "
            f"{'TE frac':>8}  {'ng·L (µm)':>12}  {'Δ/target':>10}  {'source':>8}")
    print(f"\n{_hdr}")
    print("-" * len(_hdr))
    for _i in range(_N):
        if _computed[_i]:
            _ngL_m = _ngL_arr[_i]
            _delta  = (_ngL_m - _target_ngL_m) / _target_ngL_m * 100.0
            print(f"  {_radii_um[_i]:>10.2f}  {_neff_arr[_i]:>10.4f}  "
                  f"{_ng_arr[_i]:>10.4f}  {_te_arr[_i]:>8.3f}  "
                  f"{_ngL_m*1e6:>12.4f}  {_delta:>+9.2f}%  {'cache':>8}")
    if _remaining > 0:
        _runs_done = 0
        _t0 = time.time()
        log.info(f"Launching MODE session  ({_remaining} radii to compute) …")
        _rr_mode = lumapi.MODE(hide=False)

        try:
            for _i, (_R_um, _R_m) in enumerate(zip(_radii_um, _radii_m)):
                if _computed[_i]:
                    continue   # already in cache — skip

                try:
                    _neff_v, _ng_v, _te_v, _nlo, _nhi = _rr_neff_ng(_rr_mode, _R_m)
                except Exception as _exc:
                    log.warning(f"  R = {_R_um:6.2f} µm  FAILED: {_exc}")
                    # Mark as done with NaN so a re-run doesn't retry a broken radius
                    _computed[_i] = True
                    _hf[f"{_RR_HDF5_GROUP}/flags/computed"][_i] = True
                    _hf.flush()
                    continue

                _L_m   = 2.0 * np.pi * _R_m
                _ngL_m = _ng_v * _L_m

                # ── store in memory ───────────────────────────────────────────────
                _neff_arr[_i]    = _neff_v
                _ng_arr[_i]      = _ng_v
                _te_arr[_i]      = _te_v
                _ngL_arr[_i]     = _ngL_m
                _neff_lo_arr[_i] = _nlo
                _neff_hi_arr[_i] = _nhi
                _computed[_i]    = True

                # ── write to HDF5 immediately + flush (fault-safe) ────────────────
                _rg = _hf[f"{_RR_HDF5_GROUP}/results"]
                _rg["neff"]   [_i] = _neff_v
                _rg["ng"]     [_i] = _ng_v
                _rg["te_frac"][_i] = _te_v
                _rg["ngL_um"] [_i] = _ngL_m * 1e6        # stored in µm
                _rg["neff_lo"][_i] = _nlo
                _rg["neff_hi"][_i] = _nhi
                _hf[f"{_RR_HDF5_GROUP}/flags/computed"][_i] = True
                _hf.flush()

                _runs_done += 1
                _delta = (_ngL_m - _target_ngL_m) / _target_ngL_m * 100.0
                print(f"  {_R_um:>10.2f}  {_neff_v:>10.4f}  {_ng_v:>10.4f}  "
                      f"{_te_v:>8.3f}  {_ngL_m*1e6:>12.4f}  {_delta:>+9.2f}%  "
                      f"{'FDE':>8}")

                # progress every 5 new solves
                if _runs_done % 5 == 0 or _runs_done == _remaining:
                    _elapsed = time.time() - _t0
                    _rate    = _runs_done / _elapsed if _elapsed > 0 else 1e-9
                    _eta     = (_remaining - _runs_done) / _rate
                    log.info(f"  [{_runs_done:3d}/{_remaining}]  "
                             f"R = {_R_um:.1f} µm  |  "
                             f"{_rate:.2f} radii/s  |  ETA {_eta:.0f} s")

        finally:
            _rr_mode.close()
            _elapsed_total = time.time() - _t0
            log.info(f"MODE session closed  "
                     f"({_runs_done} new solves in {_elapsed_total:.1f} s, "
                     f"avg {_elapsed_total/max(_runs_done*3,1):.2f} s/FDE)")
    _hf[_RR_HDF5_GROUP]["metadata"].attrs["timestamp_end"]  = datetime.now().isoformat()
    _hf[_RR_HDF5_GROUP]["metadata"].attrs["runs_completed"] = int(_computed.sum())
    _hf.flush()
    _hf.close()
    log.info(f"HDF5 closed  →  {HDF5_PATH}")
    _valid = ~np.isnan(_ng_arr)
    if not np.any(_valid):
        raise RuntimeError(
            "All radius sweep FDE runs failed.  "
            "Check the WARNING lines above for the specific error."
        )
    _R_v    = _radii_um[_valid]
    _neff_v = _neff_arr[_valid]
    _ng_v   = _ng_arr[_valid]
    _te_v   = _te_arr[_valid]
    _ngL_v  = _ngL_arr[_valid]   # [m]
    _dist = np.abs(_ngL_v - _target_ngL_m)
    _bi   = int(np.argmin(_dist))
    rr_best_R_um   = _R_v[_bi]
    rr_best_L_um   = 2.0 * np.pi * rr_best_R_um
    rr_best_neff   = _neff_v[_bi]
    rr_best_ng     = _ng_v[_bi]
    rr_best_ngL_um = _ngL_v[_bi] * 1e6
    rr_FSR_pred_nm = (_lam0_m**2 / _ngL_v[_bi]) * 1e9
    print("\n" + "=" * 65)
    print("  BEST MATCH")
    print("=" * 65)
    print(f"  Bend radius  R = {rr_best_R_um:.4f} µm")
    print(f"  Ring length  L = {rr_best_L_um:.4f} µm  (= 2π × R)")
    print(f"  neff           = {rr_best_neff:.6f}")
    print(f"  ng             = {rr_best_ng:.6f}")
    print(f"  ng·L achieved  = {rr_best_ngL_um:.4f} µm")
    print(f"  ng·L target    = {_target_ngL_um:.4f} µm")
    print(f"  Residual       = {_dist[_bi]*1e6:.5f} µm  "
          f"({_dist[_bi]/_target_ngL_m*100:.3f} %)")
    print(f"  FSR predicted  = {rr_FSR_pred_nm:.4f} nm  (target {RR_FSR_NM:.2f} nm)")
    print("=" * 65)
    plt.rcParams.update({
        "figure.dpi":        150,
        "font.family":       "DejaVu Sans",
        "axes.titlesize":    12,
        "axes.labelsize":    11,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
        "lines.linewidth":   2.0,
        "lines.markersize":  6,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "grid.linestyle":    "--",
    })
    _C_NG   = "#0072B2"
    _C_NEFF = "#D55E00"
    _C_NGL  = "#009E73"
    _C_BEST = "#CC79A7"
    _C_TGT  = "#E69F00"
    _suffix = (f"w={RR_WG_WIDTH_NM:.0f} nm, h={_core_t_um*1e3:.0f} nm, "
               f"$\\lambda_0$={RR_LAM0_NM:.0f} nm")
    fig1, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(_R_v, _ng_v, "o-", color=_C_NG, label="$n_g$")
    ax1.axvline(rr_best_R_um, color=_C_BEST, ls=":", lw=1.6,
                label=f"Best  R = {rr_best_R_um:.2f} µm")
    ax1.set_xlabel("Bend radius  $R$  (µm)")
    ax1.set_ylabel("Group index  $n_g$")
    ax1.set_title(f"Group index vs. Bend Radius\n({_suffix})")
    ax1.legend()
    ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig1.tight_layout();  plt.show()
    fig1.tight_layout();  plt.show()
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    ax2.plot(_R_v, _neff_v, "s-", color=_C_NEFF, label="$n_{eff}$")
    ax2.axvline(rr_best_R_um, color=_C_BEST, ls=":", lw=1.6,
                label=f"Best  R = {rr_best_R_um:.2f} µm")
    ax2.set_xlabel("Bend radius  $R$  (µm)")
    ax2.set_ylabel("Effective index  $n_{eff}$")
    ax2.set_title(f"Effective index vs. Bend Radius\n({_suffix})")
    ax2.legend()
    ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig2.tight_layout();  plt.show()
    fig2.tight_layout();  plt.show()
    fig3, ax3 = plt.subplots(figsize=(7, 4.5))
    ax3.plot(_R_v, _ngL_v * 1e6, "D-", color=_C_NGL,
             label="$n_g \\cdot L$  (FDE result)")
    ax3.axhline(_target_ngL_um, color=_C_TGT, ls="--", lw=1.8,
                label=f"Target  {_target_ngL_um:.2f} µm  (FSR = {RR_FSR_NM:.1f} nm)")
    ax3.axvline(rr_best_R_um, color=_C_BEST, ls=":", lw=1.6,
                label=f"Best  R = {rr_best_R_um:.2f} µm → FSR ≈ {rr_FSR_pred_nm:.3f} nm")
    ax3.scatter([rr_best_R_um], [rr_best_ngL_um],
                s=90, zorder=5, color=_C_BEST, edgecolors="k", linewidths=0.8)
    ax3.set_xlabel("Bend radius  $R$  (µm)")
    ax3.set_ylabel("$n_g \\cdot L$  (µm)")
    ax3.set_title(f"$n_g \\cdot L$ vs. Bend Radius — FSR Matching\n({_suffix})")
    ax3.legend(fontsize=8)
    ax3.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax3.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig3.tight_layout();  plt.show()
    fig3.tight_layout();  plt.show()
    print(f"\n  Cached in : {HDF5_PATH}")
    print(f"  HDF5 group: {_RR_HDF5_GROUP}")
    print(f"\n  Variables exported to the next cell:")
    print(f"    rr_best_R_um   = {rr_best_R_um:.4f}   # µm")
    print(f"    rr_best_L_um   = {rr_best_L_um:.4f}   # µm")
    print(f"    rr_best_neff   = {rr_best_neff:.6f}")
    print(f"    rr_best_ng     = {rr_best_ng:.6f}")
    print(f"    rr_FSR_pred_nm = {rr_FSR_pred_nm:.4f}   # nm")

    state.update({k: globals().get(k) for k in [
        'FWHM_SENSOR_NM', 'FWHM_SPEC_NM', 'RR_FSR_NM', 'RR_LAM0_NM', 'RR_N_RADII', 'RR_R_MAX_UM',
        'RR_R_MIN_UM', 'RR_WG_WIDTH_NM', '_C_BEST', '_C_NEFF', '_C_NG', '_C_NGL',
        '_C_TGT', '_FSR_m', '_L_m', '_N', '_RR_DELTA_LAM_NM', '_RR_GROUP_KEY',
        '_RR_HDF5_GROUP', '_R_m', '_R_um', '_R_v', '_bi', '_computed',
        '_core_t_um', '_delta', '_dist', '_dlam_m', '_elapsed', '_elapsed_total',
        '_eta', '_half_t_um', '_hdr', '_hf', '_i', '_lam0_m',
        '_lam_hi_m', '_lam_lo_m', '_n_cached', '_neff_arr', '_neff_hi_arr', '_neff_lo_arr',
        '_neff_v', '_ngL_arr', '_ngL_m', '_ngL_v', '_ng_arr', '_ng_v',
        '_nhi', '_nlo', '_radii_m', '_radii_um', '_rate', '_remaining',
        '_rg', '_rr_mode', '_runs_done', '_sim_y_span_um', '_sim_z_above_um', '_sim_z_below_um',
        '_sim_z_ctr_um', '_sim_z_span_um', '_sio2_z_ctr_um', '_sio2_z_span_um', '_suffix', '_t0',
        '_target_ngL_m', '_target_ngL_um', '_te_arr', '_te_v', '_valid', '_wg_w_m',
        'ax1', 'ax2', 'ax3', 'fig1', 'fig2', 'fig3',
        'rr_FSR_pred_nm', 'rr_best_L_um', 'rr_best_R_um', 'rr_best_neff', 'rr_best_ng', 'rr_best_ngL_um',
    ] if k in globals()})
    return state


if __name__ == "__main__":
    run()
