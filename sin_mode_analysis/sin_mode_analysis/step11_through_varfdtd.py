"""
step11_through_varfdtd.py — analytic through-port model vs varFDTD.

Compact analytic through-port model of the sensor ring, compared against
an external varFDTD dataset for validation.
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
import warnings
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.signal import find_peaks

from .config import *  # shared platform constants & paths
from .plotting import apply_style, save_fig, make_colorbar
from . import plotting as _plot

# ── Data-contract inputs (injected by main.py via `state`) ──────────
FWHM_SENSOR_NM = None
RR_LAM0_NM = None
_os = None
d = None
i = None
rr_FSR_pred_nm = None
rr_best_R_um = None
rr_best_neff = None
rr_best_ng = None

apply_style()

# ─────────────────────────────────────────────────────────
#  Module constants / design knobs for this step
# ─────────────────────────────────────────────────────────


def _neff_lam(lam_arr):
    return _neff0_res + (_ng0 - _neff0_res) * (_lres0 - lam_arr) / _lres0


def _T_through_model(lam_arr, k1sq, k2sq):
    t1 = np.sqrt(1.0 - k1sq);  t2 = np.sqrt(1.0 - k2sq)
    a_eff = _a_field * t2
    phi   = 2.0 * np.pi * _neff_lam(lam_arr) * _L0 / lam_arr
    return np.abs((t1 - a_eff*np.exp(1j*phi)) / (1.0 - a_eff*t1*np.exp(1j*phi))) ** 2


def _measure_dip(wl, T_lin, idx, search=500):
    Tbase = np.percentile(T_lin, 97)
    Tmin  = T_lin[idx]
    half  = (Tbase + Tmin) / 2.0
    lo, hi = max(0, idx-search), min(len(T_lin)-1, idx+search)
    li = ri = idx
    for ii in range(idx, lo, -1):
        if T_lin[ii] >= half: li = ii; break
    for ii in range(idx, hi):
        if T_lin[ii] >= half: ri = ii; break
    return dict(lres=wl[idx], Tmin=Tmin, fwhm=wl[ri]-wl[li],
                wl_L=wl[li], wl_R=wl[ri], half=half,
                er=-10.0*math.log10(max(Tmin, 1e-30)))


def _y(v):
    return 10.0*math.log10(max(float(v), 1e-15)) if PLOT_DB_SCALE else float(v)

def run(state=None):
    """Execute this step. `state` carries the shared namespace
    between steps (the notebook's old kernel globals + bridge).
    Returns the updated `state`."""
    state = {} if state is None else state
    global DERIVE_KAPPA_FROM_FWHM, FDTD_DATA_FILE, PLOT_DB_SCALE, RING_KAPPA_DROP_SQ_MANUAL, RING_KAPPA_INPUT_SQ_MANUAL, RING_LOSS_DB_PER_M_CONFIG, Y_LIM_DB, Y_LIM_LINEAR, _, _ANN_DB, _BLACK, _BLUE, _ER_dB, _F, _FSR_est_nm, _FSR_nm, _F_target, _GREY, _L0, _R0, _T_anal, _T_anal_lin, _T_f, _T_f_lin, _T_min, _X0, _X1, _a_eff, _a_field, _alpha_Npm, _at, _at_tgt, _ax, _d0, _d0f, _d1, _d1f, _dd, _dip_data, _dips_f, _fig, _fsr_fdtd, _fsr_fdtd_mean, _fv, _fwhm_nm, _ha_L, _half_anal, _ii, _k1sq, _k2sq, _kappa_src, _lam_next_nm, _lam_plot_nm, _lam_res_nm, _loss0, _lres0, _m_order, _msk, _neff0, _neff0_res, _ng0, _raw, _sc, _stem, _t1, _t2, _t_sym, _u, _wl_f, _x_L, _ya, _ya_er_tip, _ylim_hi, _ylim_lo, _yspan, er_teo, fsr_dy, fsr_fdtd, fsr_teo, lres_teo, lres_vf0, lres_vf1
    globals().update(state)

    warnings.filterwarnings("ignore")
    PLOT_DB_SCALE = True        # True → dB   |   False → lineal [0-1]
    Y_LIM_DB     = (-32.0,  8.0)   # [dB]  — margen superior amplio para flechas FSR
    Y_LIM_LINEAR = ( -0.07, 1.12)  # [u.a.]
    RING_LOSS_DB_PER_M_CONFIG = 101.0
    DERIVE_KAPPA_FROM_FWHM     = True
    RING_KAPPA_INPUT_SQ_MANUAL = 0.1458
    RING_KAPPA_DROP_SQ_MANUAL  = 0.1434
    FDTD_DATA_FILE = str(DATA_DIR) + "/Through_varFDTD_18-85_3.txt"
    _X0, _X1 = 1545.0, 1565.5
    _ANN_DB = dict(
        fsr_teo    =  5.0,    # nivel de la flecha FSR teórico
        fsr_fdtd   =  2.5,    # nivel de la flecha FSR varFDTD
        fsr_dy     =  0.8,    # dB encima de la flecha → texto FSR
        fwhm_teo   = -4.5,    # caja "FWHM = …" teórico  (→ flecha al bracket en -3 dB)
        fwhm_vf0   = -9.5,    # caja FWHM+ER varFDTD dip-0  (→ flecha al bracket)
        lres_teo   = -15.0,   # caja λ_res teórico
        lres_vf0   = -22.0,   # caja λ_res varFDTD dip-0
        er_teo     = -27.5,   # caja ER teórico
        lres_vf1   = -15.0,   # caja λ_res varFDTD dip-1
        fwhm_vf1   =  -9.5,   # caja FWHM+ER varFDTD dip-1  (→ flecha al bracket)
    )
    _R0    = rr_best_R_um * 1e-6
    _L0    = _R0 * 2.0 * math.pi
    _neff0 = rr_best_neff
    _ng0   = rr_best_ng
    _lres0 = RR_LAM0_NM * 1e-9
    _loss0 = RING_LOSS_DB_PER_M_CONFIG
    _alpha_Npm = _loss0 * math.log(10) / 20.0
    _a_field   = math.exp(-_alpha_Npm * _L0)
    _FSR_est_nm = (RR_LAM0_NM ** 2) / (_ng0 * _L0 * 1e9)
    if DERIVE_KAPPA_FROM_FWHM:
        _F_target = _FSR_est_nm / FWHM_SENSOR_NM
        _u        = (-math.pi + math.sqrt(math.pi**2 + 4.0*_F_target**2)) / (2.0*_F_target)
        _at_tgt   = _u ** 2
        if _at_tgt / _a_field > 1.0:
            warnings.warn(f"[CELL D] FWHM={FWHM_SENSOR_NM:.3f} nm incompatible con pérdidas. Usando κ² manuales.")
            _k1sq = RING_KAPPA_INPUT_SQ_MANUAL
            _k2sq = RING_KAPPA_DROP_SQ_MANUAL
            _kappa_src = "manual (fallback)"
        else:
            _t_sym = min(math.sqrt(_at_tgt / _a_field), 1.0 - 1e-9)
            _k1sq  = 1.0 - _t_sym ** 2
            _k2sq  = _k1sq
            _kappa_src = f"derivado (FWHM_SENSOR_NM = {FWHM_SENSOR_NM:.3f} nm)"
    else:
        _k1sq = RING_KAPPA_INPUT_SQ_MANUAL
        _k2sq = RING_KAPPA_DROP_SQ_MANUAL
        _kappa_src = "manual"
    _m_order   = round(_neff0 * _L0 / _lres0)
    _neff0_res = _m_order * _lres0 / _L0        # resonancia exacta en RR_LAM0_NM
    _t1 = math.sqrt(1.0 - _k1sq);  _t2 = math.sqrt(1.0 - _k2sq)
    _t1 = math.sqrt(1.0 - _k1sq);  _t2 = math.sqrt(1.0 - _k2sq)
    _a_eff   = _a_field * _t2
    _at      = _a_eff * _t1
    _F       = math.pi * math.sqrt(_at) / (1.0 - _at)
    _FSR_nm  = (_lres0 * 1e9)**2 / (_ng0 * _L0 * 1e9)
    _fwhm_nm = _FSR_nm / _F
    _T_min   = ((_t1 - _a_eff) / (1.0 - _a_eff * _t1))**2
    _ER_dB   = -10.0 * math.log10(max(_T_min, 1e-30))
    _lam_res_nm  = RR_LAM0_NM          # exactamente el diseño, sin búsqueda
    _lam_next_nm = _lam_res_nm + _FSR_nm
    _lam_plot_nm = np.linspace(_X0, _X1, 200_000)
    _T_anal_lin  = _T_through_model(_lam_plot_nm * 1e-9, _k1sq, _k2sq)
    _T_anal = (10.0 * np.log10(np.clip(_T_anal_lin, 1e-15, 1.0))
               if PLOT_DB_SCALE else _T_anal_lin)
    if not _os.path.isfile(FDTD_DATA_FILE):
        raise FileNotFoundError(f"varFDTD no encontrado:\n  {FDTD_DATA_FILE}")
    _raw     = np.genfromtxt(FDTD_DATA_FILE, delimiter=',', skip_header=3, invalid_raise=False)
    _raw     = _raw[~np.isnan(_raw).any(axis=1)]
    _raw     = _raw[np.argsort(_raw[:, 0])]
    _wl_f    = _raw[:, 0] * 1e9
    _T_f_lin = _raw[:, 1]
    _msk     = (_wl_f >= _X0) & (_wl_f <= _X1)
    _wl_f    = _wl_f[_msk];  _T_f_lin = _T_f_lin[_msk]
    _wl_f    = _wl_f[_msk];  _T_f_lin = _T_f_lin[_msk]
    _T_f = (10.0 * np.log10(np.clip(_T_f_lin, 1e-15, 1.0))
            if PLOT_DB_SCALE else _T_f_lin)
    _dips_f, _ = find_peaks(-_T_f_lin, prominence=0.25, distance=40)
    _dip_data      = [_measure_dip(_wl_f, _T_f_lin, d) for d in _dips_f]
    _fsr_fdtd      = [_dip_data[i+1]["lres"]-_dip_data[i]["lres"] for i in range(len(_dip_data)-1)]
    _fsr_fdtd_mean = float(np.mean(_fsr_fdtd)) if _fsr_fdtd else None
    print("=" * 65)
    print("  PARÁMETROS ANALÍTICOS  (Cell D v4-final)")
    print("=" * 65)
    print(f"  R          = {_R0*1e6:.4f} µm  [← rr_best_R_um]")
    print(f"  L          = {_L0*1e6:.3f} µm")
    print(f"  neff (FDE) = {_neff0:.6f}  [← rr_best_neff]")
    print(f"  neff (adj) = {_neff0_res:.6f}  (resonancia exacta en {_lam_res_nm:.2f} nm)")
    print(f"  ng         = {_ng0:.6f}  [← rr_best_ng]")
    print(f"  λ_res      = {_lam_res_nm:.4f} nm  (= RR_LAM0_NM, exacta)")
    print(f"  FSR        = {_FSR_nm:.4f} nm  (rr_FSR_pred = {rr_FSR_pred_nm:.4f} nm)")
    print(f"  κ²         = {_k1sq:.4f}  [{_kappa_src}]")
    print(f"  Finesse    = {_F:.2f}")
    print(f"  FWHM       = {_fwhm_nm:.4f} nm  (objetivo: {FWHM_SENSOR_NM:.2f} nm)")
    print(f"  ER         = {_ER_dB:.1f} dB  |  a_field = {_a_field:.5f}")
    print(f"  Escala     = {'dB  [10·log₁₀(T)]' if PLOT_DB_SCALE else 'lineal'}")
    print("=" * 65)
    for _dd in _dip_data:
        print(f"  varFDTD  λ={_dd['lres']:.4f} nm  FWHM={_dd['fwhm']:.4f} nm  ER={_dd['er']:.2f} dB")
    for _ii, _fv in enumerate(_fsr_fdtd):
        print(f"  FSR varFDTD [{_ii+1}→{_ii+2}] = {_fv:.4f} nm")
    _BLUE  = "#1565C0"
    _BLACK = "#212121"
    _GREY  = "#757575"
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif" : ["DejaVu Serif", "Georgia", "Times New Roman"],
        "font.size"  : 14, "axes.labelsize": 17, "axes.titlesize": 14,
        "legend.fontsize": 11, "xtick.labelsize": 14, "ytick.labelsize": 14,
        "axes.linewidth": 1.2, "axes.grid": True,
        "grid.alpha": 0.18, "grid.linewidth": 0.5,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "xtick.major.size": 6.0, "ytick.major.size": 6.0,
        "xtick.minor.size": 3.0, "ytick.minor.size": 3.0,
        "figure.dpi": 150,
    })
    _fig, _ax = plt.subplots(figsize=(14, 7.5))
    _ylim_lo, _ylim_hi = Y_LIM_DB if PLOT_DB_SCALE else Y_LIM_LINEAR
    _yspan = _ylim_hi - _ylim_lo
    if PLOT_DB_SCALE:
        _ya = _ANN_DB                           # alias para legibilidad
        _ya_er_tip = _ylim_lo + 0.04 * _yspan  # fondo visible (dip teórico → -∞)
        _x_L   = _X0 + 0.8                     # x base LEFT  (ha="left")
        _ha_L  = "left"
    else:
        # Posiciones fraccionales para modo lineal
        class _ya:
            fsr_teo  = _ylim_lo + 0.916 * _yspan
            fsr_fdtd = _ylim_lo + 0.822 * _yspan
            fsr_dy   = 0.022 * _yspan
            lres_teo = _ylim_lo + 0.310 * _yspan
            lres_vf0 = _ylim_lo + 0.175 * _yspan
            er_teo   = _ylim_lo + 0.095 * _yspan
            lres_vf1 = _ylim_lo + 0.390 * _yspan
        _ya_er_tip = 0.0
        _x_L  = _lam_res_nm - 4.2
        _ha_L = "center"
    _ax.plot(_lam_plot_nm, _T_anal,
             color=_BLUE, lw=2.4, ls="-", zorder=3, alpha=0.90,
             label="Teórico  (κ_orig)")
    _ax.plot(_wl_f, _T_f,
             color=_BLACK, lw=1.8, alpha=0.88, zorder=4,
             label="varFDTD")
    _ax.annotate("",
                 xy=(_lam_res_nm, _ya["fsr_teo"]),
                 xytext=(_lam_next_nm, _ya["fsr_teo"]),
                 arrowprops=dict(arrowstyle="<->", color=_BLUE, lw=1.7), zorder=5)
    _ax.text((_lam_res_nm+_lam_next_nm)/2.0, _ya["fsr_teo"]+_ya["fsr_dy"],
             f"FSR = {_FSR_nm:.2f} nm  (teórico)",
             ha="center", va="bottom", fontsize=11.5, color=_BLUE,
             bbox=dict(boxstyle="round,pad=0.30", fc="white", ec=_BLUE, alpha=0.95, lw=0.9),
             zorder=5)
    if len(_dip_data) >= 2:
        for _ii in range(len(_dip_data)-1):
            _d0f, _d1f = _dip_data[_ii], _dip_data[_ii+1]
            _ax.annotate("",
                         xy=(_d0f["lres"], _ya["fsr_fdtd"]),
                         xytext=(_d1f["lres"], _ya["fsr_fdtd"]),
                         arrowprops=dict(arrowstyle="<->", color=_BLACK, lw=1.7), zorder=5)
            _ax.text((_d0f["lres"]+_d1f["lres"])/2.0, _ya["fsr_fdtd"]+_ya["fsr_dy"],
                     f"FSR = {_fsr_fdtd[_ii]:.2f} nm  (varFDTD)",
                     ha="center", va="bottom", fontsize=11.5, color=_BLACK,
                     bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="#555555", alpha=0.95, lw=0.8),
                     zorder=5)
    _half_anal = (1.0 + _T_min) / 2.0
    _ax.annotate("",
                 xy=(_lam_res_nm - _fwhm_nm/2.0, _y(_half_anal)),
                 xytext=(_lam_res_nm + _fwhm_nm/2.0, _y(_half_anal)),
                 arrowprops=dict(arrowstyle="<->", color=_BLUE, lw=1.6), zorder=7)
    _ax.annotate(
        f"$\\lambda_{{res}}$ = {_lam_res_nm:.2f} nm\n(teórico, acopl. crítico)",
        xy=(_lam_res_nm, _y(_T_min + 0.008)),
        xytext=(_x_L, _ya["lres_teo"]),
        fontsize=11, color=_BLUE, ha=_ha_L, va="center",
        arrowprops=dict(arrowstyle="->", color=_BLUE, lw=1.2,
                        connectionstyle="arc3,rad=-0.22"),
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=_BLUE, alpha=0.96, lw=0.9),
        zorder=8)
    if PLOT_DB_SCALE:
        _ax.annotate(
            f"FWHM = {_fwhm_nm:.2f} nm",
            xy=(_lam_res_nm, _y(_half_anal)),          # centro del bracket
            xytext=(_x_L, _ya["fwhm_teo"]),
            fontsize=11, color=_BLUE, ha="left", va="top",
            arrowprops=dict(arrowstyle="->", color=_BLUE, lw=1.0,
                            connectionstyle="arc3,rad=+0.10"),
            bbox=dict(boxstyle="round,pad=0.30", fc="white", ec=_BLUE, alpha=0.95, lw=0.9),
            zorder=7)
    else:
        _ax.text(_lam_res_nm - 1.0, _y(_half_anal) + 0.030*_yspan,
                 f"FWHM = {_fwhm_nm:.2f} nm",
                 ha="center", va="bottom", fontsize=11, color=_BLUE,
                 bbox=dict(boxstyle="round,pad=0.30", fc="white", ec=_BLUE, alpha=0.95, lw=0.9),
                 zorder=7)
    _ax.annotate(
        f"ER aprox {_ER_dB:.0f} dB",
        xy=(_lam_res_nm, _ya_er_tip if PLOT_DB_SCALE else _y(_T_min)),
        xytext=(_x_L + (0.7 if PLOT_DB_SCALE else 0.0), _ya["er_teo"]),
        fontsize=11, color=_BLUE, ha=_ha_L, va="center",
        arrowprops=dict(arrowstyle="->", color=_BLUE, lw=1.1,
                        connectionstyle="arc3,rad=-0.22"),
        bbox=dict(boxstyle="round,pad=0.30", fc="white", ec=_BLUE, alpha=0.95, lw=0.9),
        zorder=8)
    if len(_dip_data) >= 1:
        _d0 = _dip_data[0]

        # λ_res varFDTD dip-0
        _ax.annotate(
            f"$\\lambda_{{res}}$ = {_d0['lres']:.2f} nm  (varFDTD)",
            xy=(_d0["lres"], _y(_d0["Tmin"] + 0.008)),
            xytext=(_x_L, _ya["lres_vf0"] if PLOT_DB_SCALE else _ylim_lo + 0.175*_yspan),
            fontsize=10.5, color=_BLACK, ha=_ha_L, va="center",
            arrowprops=dict(arrowstyle="->", color=_GREY, lw=1.1,
                            connectionstyle="arc3,rad=-0.22"),
            bbox=dict(boxstyle="round,pad=0.32", fc="#FAFAFA", ec=_GREY, alpha=0.95, lw=0.8),
            zorder=8)

        # Bracket FWHM varFDTD dip-0
        _ax.annotate("",
                     xy=(_d0["wl_L"], _y(_d0["half"])),
                     xytext=(_d0["wl_R"], _y(_d0["half"])),
                     arrowprops=dict(arrowstyle="<->", color=_GREY, lw=1.3), zorder=7)

        # Caja FWHM+ER dip-0 con flecha al bracket (dB) o texto al lado (lineal)
        if PLOT_DB_SCALE:
            _ax.annotate(
                f"FWHM aprox {_d0['fwhm']:.2f} nm\nER = {_d0['er']:.1f} dB",
                xy=(_d0["wl_L"], _y(_d0["half"])),        # borde izq. del bracket
                xytext=(_x_L, _ya["fwhm_vf0"]),
                fontsize=10, color="#333333", ha="left", va="top",
                arrowprops=dict(arrowstyle="->", color=_GREY, lw=1.0,
                                connectionstyle="arc3,rad=-0.10"),
                bbox=dict(boxstyle="round,pad=0.26", fc="#FAFAFA",
                          ec="#9E9E9E", alpha=0.94, lw=0.7), zorder=7)
        else:
            _ax.text(_d0["lres"] - 2.0, _y(_d0["half"]) - 0.072*_yspan,
                     f"FWHM aprox {_d0['fwhm']:.2f} nm\nER = {_d0['er']:.1f} dB",
                     ha="center", va="top", fontsize=10, color="#333333",
                     bbox=dict(boxstyle="round,pad=0.26", fc="#FAFAFA",
                               ec="#9E9E9E", alpha=0.94, lw=0.7), zorder=7)
    if len(_dip_data) >= 2:
        _d1 = _dip_data[1]

        # λ_res varFDTD dip-1
        _ax.annotate(
            f"$\\lambda_{{res}}$ = {_d1['lres']:.2f} nm  (varFDTD)",
            xy=(_d1["lres"], _y(_d1["Tmin"] + 0.008)),
            xytext=(_d1["lres"] + 0.5, _ya["lres_vf1"] if PLOT_DB_SCALE else _ylim_lo + 0.390*_yspan),
            fontsize=10.5, color=_BLACK, ha="left", va="center",
            arrowprops=dict(arrowstyle="->", color=_GREY, lw=1.1,
                            connectionstyle="arc3,rad=+0.22"),
            bbox=dict(boxstyle="round,pad=0.32", fc="#FAFAFA", ec=_GREY, alpha=0.95, lw=0.8),
            zorder=8)

        # Bracket FWHM varFDTD dip-1
        _ax.annotate("",
                     xy=(_d1["wl_L"], _y(_d1["half"])),
                     xytext=(_d1["wl_R"], _y(_d1["half"])),
                     arrowprops=dict(arrowstyle="<->", color=_GREY, lw=1.3), zorder=7)

        # Caja FWHM+ER dip-1 → derecha, flecha al bracket
        if PLOT_DB_SCALE:
            _ax.annotate(
                f"FWHM aprox {_d1['fwhm']:.2f} nm\nER = {_d1['er']:.1f} dB",
                xy=(_d1["wl_R"], _y(_d1["half"])),        # borde der. del bracket
                xytext=(_d1["lres"] + 0.5, _ya["fwhm_vf1"]),
                fontsize=10, color="#333333", ha="left", va="top",
                arrowprops=dict(arrowstyle="->", color=_GREY, lw=1.0,
                                connectionstyle="arc3,rad=+0.10"),
                bbox=dict(boxstyle="round,pad=0.26", fc="#FAFAFA",
                          ec="#9E9E9E", alpha=0.94, lw=0.7), zorder=7)
        else:
            _ax.text(_d1["lres"] + 2.0, _y(_d1["half"]) - 0.072*_yspan,
                     f"FWHM aprox {_d1['fwhm']:.2f} nm\nER = {_d1['er']:.1f} dB",
                     ha="center", va="top", fontsize=10, color="#333333",
                     bbox=dict(boxstyle="round,pad=0.26", fc="#FAFAFA",
                               ec="#9E9E9E", alpha=0.94, lw=0.7), zorder=7)
    _ax.set_xlabel(r"$\lambda$  (nm)", fontsize=17, labelpad=10)
    if PLOT_DB_SCALE:
        _ax.set_ylabel(r"Transmitancia  $T_\mathrm{through}$  (dB)", fontsize=17, labelpad=10)
        _ax.set_ylim(*Y_LIM_DB)
        _ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
        _ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    else:
        _ax.set_ylabel(r"Transmitancia  $T_\mathrm{through}$  (u.a.)", fontsize=17, labelpad=10)
        _ax.set_ylim(*Y_LIM_LINEAR)
        _ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        _ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    _ax.set_xlim(_X0, _X1)
    _ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    _ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    _ax.set_title(
        r"Transmitancia teórica (acoplamiento crítico,  $\kappa_\mathrm{orig}$)"
        r"  vs  simulación varFDTD" + "\n"
        f"Anillo sensor SiN  —  "
        f"$R$ = {_R0*1e6:.2f} µm  |  $L$ = {_L0*1e6:.2f} µm  |  "
        f"$n_{{eff}}$ = {_neff0:.4f}  |  $n_g$ = {_ng0:.4f}  |  "
        f"Pérdidas = {_loss0:.0f} dB/m",
        fontsize=13, pad=12)
    _ax.legend(framealpha=0.96, fontsize=11, loc="upper left",
               edgecolor="#BDBDBD", handlelength=2.0, handleheight=1.2)
    _fig.tight_layout(pad=1.5)
    _sc   = "dB" if PLOT_DB_SCALE else "lineal"
    _stem = f"theoretical_kappa_orig_vs_varFDTD_v4_{_sc}"
    _fig.savefig(DATA_DIR / (_stem + ".png"), dpi=300, bbox_inches="tight")
    _fig.savefig(DATA_DIR / (_stem + ".pdf"),           bbox_inches="tight")
    log.info(f"Guardado → {_stem}.png/.pdf  en  {DATA_DIR}")
    plt.show()
    print(f"  → {str(DATA_DIR / (_stem + '.png'))}")

    state.update({k: globals().get(k) for k in [
        'DERIVE_KAPPA_FROM_FWHM', 'FDTD_DATA_FILE', 'PLOT_DB_SCALE', 'RING_KAPPA_DROP_SQ_MANUAL', 'RING_KAPPA_INPUT_SQ_MANUAL', 'RING_LOSS_DB_PER_M_CONFIG',
        'Y_LIM_DB', 'Y_LIM_LINEAR', '_', '_ANN_DB', '_BLACK', '_BLUE',
        '_ER_dB', '_F', '_FSR_est_nm', '_FSR_nm', '_F_target', '_GREY',
        '_L0', '_R0', '_T_anal', '_T_anal_lin', '_T_f', '_T_f_lin',
        '_T_min', '_X0', '_X1', '_a_eff', '_a_field', '_alpha_Npm',
        '_at', '_at_tgt', '_ax', '_d0', '_d0f', '_d1',
        '_d1f', '_dd', '_dip_data', '_dips_f', '_fig', '_fsr_fdtd',
        '_fsr_fdtd_mean', '_fv', '_fwhm_nm', '_ha_L', '_half_anal', '_ii',
        '_k1sq', '_k2sq', '_kappa_src', '_lam_next_nm', '_lam_plot_nm', '_lam_res_nm',
        '_loss0', '_lres0', '_m_order', '_msk', '_neff0', '_neff0_res',
        '_ng0', '_raw', '_sc', '_stem', '_t1', '_t2',
        '_t_sym', '_u', '_wl_f', '_x_L', '_ya', '_ya_er_tip',
        '_ylim_hi', '_ylim_lo', '_yspan', 'er_teo', 'fsr_dy', 'fsr_fdtd',
        'fsr_teo', 'lres_teo', 'lres_vf0', 'lres_vf1',
    ] if k in globals()})
    return state


if __name__ == "__main__":
    run()
