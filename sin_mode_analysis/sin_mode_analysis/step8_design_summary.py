"""
step8_design_summary.py — platform design summary tables and spectra.

Collects every simulated parameter and renders the geometry/optics table,
the coupling table and the 14-ring through-port transmission curves.
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
from .plotting import apply_style, save_fig, make_colorbar
from . import plotting as _plot

# ── Data-contract inputs (injected by main.py via `state`) ──────────
ALPHA_PROP_DB_CM = None
FSR_pm_nm = None
FWHM_SENSOR_NM = None
FWHM_SPEC_NM = None
GridSpec = None
RR_WG_WIDTH_NM = None
_cc_results = None
_r = None
cc_FWHM_nm = None
cc_L_pm_um = None
cc_Lc_a_um = None
cc_Q_target = None
cc_R_pm_um = None
cc_a = None
cc_k1 = None
cc_k2 = None
cc_labels = None
cc_lam_res_nm = None
cc_r1 = None
cc_r2 = None
dc_dn_input_sim = None
dc_dn_output_sim = None
dc_gap_input_nm = None
dc_gap_output_nm = None
dc_k1_achieved = None
dc_k2_achieved = None
spec_FSR_pm_nm = None

apply_style()

# ─────────────────────────────────────────────────────────
#  Module constants / design knobs for this step
# ─────────────────────────────────────────────────────────


def _adf_through(phi, a, r1, r2):
    """Add-drop through-port intensity transmission  Tp  (Bogaerts Eq. 5)."""
    num = r1**2 * a**2 - 2*r1*r2*a*np.cos(phi) + r2**2
    den = 1.0 - 2*r1*r2*a*np.cos(phi) + (r1*r2*a)**2
    return num / den


def _adf_drop(phi, a, r1, r2):
    """Add-drop drop-port intensity transmission  Td  (Bogaerts Eq. 6)."""
    num = (1 - r1**2) * (1 - r2**2) * a
    den = 1.0 - 2*r1*r2*a*np.cos(phi) + (r1*r2*a)**2
    return num / den

def run(state=None):
    """Execute this step. `state` carries the shared namespace
    between steps (the notebook's old kernel globals + bridge).
    Returns the updated `state`."""
    state = {} if state is None else state
    global _L, _LN10, _Lc, _N_PTS, _N_RINGS, _Q, _R, _SEP, _Td, _Tp, _a, _alpha, _ax, _ax00, _ax01, _ax10, _ax11, _ax_r, _ax_sensor, _ax_t1, _ax_t2, _axes, _axes_ov, _b, _bars00, _bars01, _bars10, _base_loss, _base_neff, _base_ng, _bw, _cbar, _cell, _clad, _clad_str, _cmap, _cnorm, _col, _colors, _dn_i, _dn_o, _er_db, _face, _fig_all, _fig_ov, _fig_r, _fig_t1, _fig_t2, _fsr, _fsr_all_nm, _fwhm, _fwhm_meas, _fwhm_pm, _gap_i, _gap_o, _gs, _half_p, _half_power, _i, _info, _j, _k1a, _k1sq, _k2a, _k2sq, _l_idx, _lam, _lam0, _lam_arr, _lam_l, _lam_r, _lam_span, _left_idx, _loss_col_idx, _neff, _neff_col_idx, _ng, _ng_col_idx, _path_ov, _path_ri, _path_t1, _path_t2, _path_tx, _phi, _r1, _r2, _r_idx, _right_idx, _ring_labels, _ring_types, _row, _sm, _t1_cols, _t1_rows, _t2_cols, _t2_rows, _table1, _table2, _type, _v, _w, _x, _xin, _xout, _y_fsr, _y_fwhm, _y_mid, cc_alpha_field, cc_loss_dB_m, cc_neff_pm, cc_ng, i
    globals().update(state)

    plt.rcParams.update({
        "figure.dpi":        150,
        "font.family":       "DejaVu Sans",
        "axes.titlesize":    11,
        "axes.labelsize":    10,
        "xtick.labelsize":   8.5,
        "ytick.labelsize":   8.5,
        "legend.fontsize":   8,
        "lines.linewidth":   1.6,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.25,
        "grid.linestyle":    "--",
    })
    _N_RINGS = 14
    _cmap    = cm.viridis
    _cnorm   = Normalize(vmin=0, vmax=_N_RINGS - 1)
    _colors  = [_cmap(_cnorm(i)) for i in range(_N_RINGS)]
    _LN10 = np.log(10)   # ≈ 2.302585
    cc_neff_pm    = np.array([float(_r["neff_pm"])     for _r in _cc_results])  # (14,)
    cc_ng         = np.array([float(_r["ng"])          for _r in _cc_results])  # (14,)
    cc_alpha_field= np.array([float(_r["alpha_total"]) for _r in _cc_results])  # (14,) [1/m]
    cc_loss_dB_m  = cc_alpha_field * 20.0 / _LN10                               # (14,) [dB/m]
    _fsr_all_nm = np.empty(_N_RINGS)
    _fsr_all_nm[0]   = FSR_pm_nm        # sensor ring  (Cell 9)
    _fsr_all_nm[1:]  = spec_FSR_pm_nm   # spectrometer rings (Cell 10)
    _N_PTS = 4001
    _ring_types = ["Sensor"] + [f"Spec {i:02d}" for i in range(1, 14)]
    _t1_rows = []
    for i in range(_N_RINGS):
        _lam   = float(cc_lam_res_nm[i])
        _R     = float(cc_R_pm_um[i])
        _L     = float(cc_L_pm_um[i])
        _fsr   = float(_fsr_all_nm[i])
        _fwhm  = float(cc_FWHM_nm[i])
        _Q     = float(cc_Q_target[i])
        _a     = float(cc_a[i])
        _r1    = float(cc_r1[i])
        _r2    = float(cc_r2[i])
        _neff  = float(cc_neff_pm[i])
        _ng    = float(cc_ng[i])
        _alpha = float(cc_loss_dB_m[i])   # ← NEW
        _t1_rows.append([
            cc_labels[i],
            _ring_types[i],
            f"{_lam:.4f}",
            f"{_R:.4f}",
            f"{_L:.4f}",
            f"{_fsr:.4f}",
            f"{_fwhm:.4f}",
            f"{_Q:.0f}",
            f"{_a:.6f}",
            f"{_r1:.6f}",
            f"{_r2:.6f}",
            f"{_neff:.6f}",
            f"{_ng:.6f}",
            f"{_alpha:.4f}",   # ← NEW
        ])
    _t1_cols = [
        "Ring", "Type",
        "λ_res\n(nm)", "R\n(µm)", "L\n(µm)",
        "FSR\n(nm)", "FWHM\n(nm)", "Q-factor",
        "a\n(round-trip)", "r₁\n(input)", "r₂\n(output)",
        "neff\n(@ λ_res)",
        "ng\n(@ λ_res)",
        "α_loss\n(dB/m)",   # ← NEW
    ]
    _fig_t1, _ax_t1 = plt.subplots(figsize=(27, 7))   # widened again for the extra column
    _ax_t1.axis("off")
    _table1 = _ax_t1.table(
        cellText   = _t1_rows,
        colLabels  = _t1_cols,
        loc        = "center",
        cellLoc    = "center",
    )
    _table1.auto_set_font_size(False)
    _table1.set_fontsize(8.5)
    _table1.scale(1.0, 1.55)
    for _j in range(len(_t1_cols)):
        _cell = _table1[0, _j]
        _cell.set_facecolor("#1A1A2E")
        _cell.set_text_props(color="white", fontweight="bold")
    for _i in range(_N_RINGS):
        _face = "#EAECF0" if _i % 2 == 0 else "#FFFFFF"
        if _i == 0:
            _face = "#FFF3CD"   # amber for sensor ring
        for _j in range(len(_t1_cols)):
            _table1[_i + 1, _j].set_facecolor(_face)
    _neff_col_idx  = _t1_cols.index("neff\n(@ λ_res)")
    _ng_col_idx    = _t1_cols.index("ng\n(@ λ_res)")
    _loss_col_idx  = _t1_cols.index("α_loss\n(dB/m)")   # ← NEW
    for _i in range(_N_RINGS):
        _base_neff = "#D6EAF8" if _i % 2 == 0 else "#EBF5FB"
        _base_ng   = "#D5F5E3" if _i % 2 == 0 else "#EAFAF1"
        _base_loss = "#FDEDEC" if _i % 2 == 0 else "#FEF9E7"   # ← light red/orange tint
        if _i == 0:
            _base_neff = "#FDEBD0"
            _base_ng   = "#FDEBD0"
            _base_loss = "#FDEBD0"
        _table1[_i + 1, _neff_col_idx ].set_facecolor(_base_neff)
        _table1[_i + 1, _ng_col_idx   ].set_facecolor(_base_ng)
        _table1[_i + 1, _loss_col_idx ].set_facecolor(_base_loss)   # ← NEW
    _table1[0, _neff_col_idx ].set_facecolor("#154360")   # dark blue
    _table1[0, _ng_col_idx   ].set_facecolor("#145A32")   # dark green
    _table1[0, _loss_col_idx ].set_facecolor("#7B241C")   # dark red  ← NEW
    _table1.auto_set_column_width(list(range(len(_t1_cols))))
    _fig_t1.suptitle(
        f"TABLE 1 — Geometrical & Optical Parameters  │  "
        f"SiN {CORE_THICKNESS_UM*1e3:.0f} nm × {RR_WG_WIDTH_NM:.0f} nm  │  "
        f"α_prop = {ALPHA_PROP_DB_CM:.1f} dB/cm  │  14 rings (1 sensor + 13 spectrometer)  │  "
        f"neff & ng: bent FDE @ λ_res  │  α_loss = α_prop + α_bend  [dB/m]",
        fontsize=11, fontweight="bold", y=0.97,
    )
    _fig_t1.tight_layout()
    _path_t1 = DATA_DIR / f"{VERSION_NAME}_TABLE1_optical_params.png"
    _fig_t1.savefig(str(_path_t1), dpi=200, bbox_inches="tight",
                    facecolor="white")
    print(f"  Saved → {_path_t1}")
    plt.show()
    _t2_rows = []
    for i in range(_N_RINGS):
        _k1sq  = float(cc_k1[i])**2
        _k2sq  = float(cc_k2[i])**2
        _gap_i = float(dc_gap_input_nm[i])
        _gap_o = float(dc_gap_output_nm[i])
        _Lc    = float(cc_Lc_a_um[i])
        _dn_i  = float(dc_dn_input_sim[i])
        _dn_o  = float(dc_dn_output_sim[i])
        _k1a   = float(dc_k1_achieved[i])
        _k2a   = float(dc_k2_achieved[i])
        _t2_rows.append([
            cc_labels[i],
            _ring_types[i],
            f"{float(cc_k1[i]):.6f}",
            f"{float(cc_k2[i]):.6f}",
            f"{_k1sq:.6f}",
            f"{_k2sq:.6f}",
            f"{_gap_i:.1f}",
            f"{_gap_o:.1f}",
            f"{_Lc:.4f}",
            f"{_dn_i:.6f}",
            f"{_dn_o:.6f}",
            f"{_k1a:.6f}",
            f"{_k2a:.6f}",
        ])
    _t2_cols = [
        "Ring", "Type",
        "k₁\n(target)", "k₂\n(target)",
        "k₁²\n(power)", "k₂²\n(power)",
        "Gap in\n(nm)", "Gap out\n(nm)",
        "Lc\n(µm)",
        "Δn_in\n(sim)", "Δn_out\n(sim)",
        "k₁\n(achieved)", "k₂\n(achieved)",
    ]
    _fig_t2, _ax_t2 = plt.subplots(figsize=(24, 7))
    _ax_t2.axis("off")
    _table2 = _ax_t2.table(
        cellText   = _t2_rows,
        colLabels  = _t2_cols,
        loc        = "center",
        cellLoc    = "center",
    )
    _table2.auto_set_font_size(False)
    _table2.set_fontsize(8.0)
    _table2.scale(1.0, 1.55)
    for _j in range(len(_t2_cols)):
        _cell = _table2[0, _j]
        _cell.set_facecolor("#1B2A4A")
        _cell.set_text_props(color="white", fontweight="bold")
    for _i in range(_N_RINGS):
        _face = "#E8F4FD" if _i % 2 == 0 else "#FFFFFF"
        if _i == 0:
            _face = "#FFF3CD"
        for _j in range(len(_t2_cols)):
            _table2[_i + 1, _j].set_facecolor(_face)
    _fig_t2.suptitle(
        f"TABLE 2 — Coupling Coefficients & Gap Parameters  │  "
        f"Critical coupling  (asymmetric r₁ ≠ r₂)  │  "
        f"DC selector: Lc = R_pm / 3",
        fontsize=11, fontweight="bold", y=0.97,
    )
    _fig_t2.tight_layout()
    _path_t2 = DATA_DIR / f"{VERSION_NAME}_TABLE2_coupling_params.png"
    _fig_t2.savefig(str(_path_t2), dpi=200, bbox_inches="tight",
                    facecolor="white")
    print(f"  Saved → {_path_t2}")
    plt.show()
    _fig_all = plt.figure(figsize=(26, 16))
    _fig_all.suptitle(
        f"Through-Port & Drop-Port Transmission — All 14 Rings\n"
        f"SiN {CORE_THICKNESS_UM*1e3:.0f} nm × {RR_WG_WIDTH_NM:.0f} nm  │  "
        f"α = {ALPHA_PROP_DB_CM:.1f} dB/cm  │  "
        f"FWHM_sensor = {FWHM_SENSOR_NM:.2f} nm  │  "
        f"FWHM_spec = {FWHM_SPEC_NM:.2f} nm  │  "
        f"Target FSR ≈ 10 nm",
        fontsize=12, fontweight="bold", y=1.002,
    )
    _gs = GridSpec(3, 5, figure=_fig_all, hspace=0.55, wspace=0.40)
    _axes = []
    _ax_sensor = _fig_all.add_subplot(_gs[0, 0:2])
    _axes.append(_ax_sensor)
    for _row in range(3):
        for _col in range(5):
            if _row == 0 and _col < 2:
                continue
            if len(_axes) >= _N_RINGS:
                break
            _axes.append(_fig_all.add_subplot(_gs[_row, _col]))
        if len(_axes) >= _N_RINGS:
            break
    for _i in range(_N_RINGS):
        _ax    = _axes[_i]
        _lam0  = float(cc_lam_res_nm[_i])
        _fsr   = float(_fsr_all_nm[_i])
        _fwhm  = float(cc_FWHM_nm[_i])
        _Q     = float(cc_Q_target[_i])
        _a     = float(cc_a[_i])
        _r1    = float(cc_r1[_i])
        _r2    = float(cc_r2[_i])
        _neff  = float(cc_neff_pm[_i])
        _ng    = float(cc_ng[_i])
        _alpha = float(cc_loss_dB_m[_i])
        _col   = _colors[_i]

        _lam_span = 2.2 * _fsr
        _lam_arr  = np.linspace(_lam0 - _lam_span/2, _lam0 + _lam_span/2, _N_PTS)
        _phi      = 2.0 * np.pi * (_lam_arr - _lam0) / _fsr

        _Tp = _adf_through(_phi, _a, _r1, _r2)
        _Td = _adf_drop(   _phi, _a, _r1, _r2)

        _ax.plot(_lam_arr, _Tp, color=_col, lw=1.5, label="Through")
        _ax.plot(_lam_arr, _Td, color=_col, lw=1.2, ls="--", alpha=0.65, label="Drop")
        _ax.axvline(_lam0, color="gray", lw=0.8, ls=":", alpha=0.7)

        # FSR bracket
        _y_fsr = 0.92
        _ax.annotate(
            "", xy=(_lam0 + _fsr, _y_fsr), xytext=(_lam0, _y_fsr),
            xycoords=("data", "axes fraction"),
            textcoords=("data", "axes fraction"),
            arrowprops=dict(arrowstyle="<->", color="dimgray", lw=0.9),
        )
        _ax.text(
            _lam0 + _fsr/2, _y_fsr + 0.04,
            f"FSR={_fsr:.3f} nm",
            transform=_ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=6.5, color="dimgray",
        )

        # FWHM annotation
        _half_power = (_Tp.min() + 1.0) / 2.0
        try:
            _left_idx  = np.where((_lam_arr < _lam0) & (_Tp < _half_power))[0][-1]
            _right_idx = np.where((_lam_arr > _lam0) & (_Tp < _half_power))[0][0]
            _lam_l     = _lam_arr[_left_idx]
            _lam_r     = _lam_arr[_right_idx]
            _fwhm_meas = _lam_r - _lam_l
            _y_fwhm    = 0.25
            _ax.annotate(
                "", xy=(_lam_r, _y_fwhm), xytext=(_lam_l, _y_fwhm),
                xycoords=("data", "axes fraction"),
                textcoords=("data", "axes fraction"),
                arrowprops=dict(arrowstyle="<->", color=_col, lw=1.0),
            )
            _ax.text(
                _lam0, _y_fwhm - 0.10,
                f"FWHM≈{_fwhm_meas*1e3:.0f} pm",
                transform=_ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=6.5, color=_col,
            )
        except IndexError:
            pass

        # ER annotation
        _er_db = -10 * np.log10(max(_Tp.min(), 1e-12))
        _ax.text(0.97, 0.10,
                 f"ER={_er_db:.1f} dB",
                 transform=_ax.transAxes, ha="right", va="bottom",
                 fontsize=6.5, color=_col,
                 bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7,
                           ec=_col, lw=0.5))

        _clad_str = "aqueous" if _i == 0 else "SiO₂"
        _ax.set_title(
            f"{cc_labels[_i]}  ({_clad_str})\n"
            f"λ_res={_lam0:.3f} nm   Q={_Q:.0f}   "
            f"neff={_neff:.4f}   ng={_ng:.4f}   α={_alpha:.1f} dB/m",
            fontsize=7.5, pad=3,
        )
        _ax.set_xlabel("Wavelength (nm)", fontsize=7.5)
        _ax.set_ylabel("Transmission", fontsize=7.5)
        _ax.set_ylim(-0.05, 1.10)
        _ax.set_xlim(_lam_arr[0], _lam_arr[-1])
        _ax.tick_params(labelsize=6.5)
        _ax.xaxis.set_major_locator(ticker.MaxNLocator(4, prune="both"))
        _ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))

        if _i == 0:
            _ax.legend(loc="upper right", fontsize=7, framealpha=0.8)
    _sm   = cm.ScalarMappable(cmap=_cmap, norm=_cnorm)
    _sm.set_array([])
    _cbar = _fig_all.colorbar(
        _sm, ax=_axes,
        orientation="vertical", fraction=0.008, pad=0.01, shrink=0.75,
    )
    _cbar.set_label("Ring index", fontsize=9)
    _cbar.set_ticks(range(_N_RINGS))
    _cbar.ax.tick_params(labelsize=7)
    _path_tx = DATA_DIR / f"{VERSION_NAME}_transmission_all_rings.png"
    _fig_all.savefig(str(_path_tx), dpi=180, bbox_inches="tight",
                     facecolor="white")
    print(f"  Saved → {_path_tx}")
    plt.show()
    for _i in range(_N_RINGS):
        _lam0  = float(cc_lam_res_nm[_i])
        _fsr   = float(_fsr_all_nm[_i])
        _fwhm  = float(cc_FWHM_nm[_i])
        _Q     = float(cc_Q_target[_i])
        _a     = float(cc_a[_i])
        _r1    = float(cc_r1[_i])
        _r2    = float(cc_r2[_i])
        _neff  = float(cc_neff_pm[_i])
        _ng    = float(cc_ng[_i])
        _alpha = float(cc_loss_dB_m[_i])   # ← NEW
        _col   = _colors[_i]
        _clad  = "aqueous (n=1.33)" if _i == 0 else "SiO₂ (n=1.4469)"

        _lam_span = 2.5 * _fsr
        _lam_arr  = np.linspace(_lam0 - _lam_span/2, _lam0 + _lam_span/2, 6001)
        _phi      = 2.0 * np.pi * (_lam_arr - _lam0) / _fsr

        _Tp = _adf_through(_phi, _a, _r1, _r2)
        _Td = _adf_drop(   _phi, _a, _r1, _r2)

        _fig_r, _ax_r = plt.subplots(figsize=(10, 5))

        _ax_r.plot(_lam_arr, _Tp, color=_col, lw=2.0, label="Through port  $T_p$")
        _ax_r.plot(_lam_arr, _Td, color=_col, lw=1.5, ls="--", alpha=0.7,
                   label="Drop port  $T_d$")
        _ax_r.axvline(_lam0, color="gray", lw=1.0, ls=":", alpha=0.7,
                      label=f"$\\lambda_{{res}}$ = {_lam0:.4f} nm")

        _half_p = (_Tp.min() + 1.0) / 2.0
        try:
            _l_idx     = np.where((_lam_arr < _lam0) & (_Tp < _half_p))[0][-1]
            _r_idx     = np.where((_lam_arr > _lam0) & (_Tp < _half_p))[0][0]
            _lam_l     = _lam_arr[_l_idx]
            _lam_r     = _lam_arr[_r_idx]
            _fwhm_meas = _lam_r - _lam_l
            _y_mid     = (_Tp[_l_idx] + _Tp[_r_idx]) / 2.0
            _ax_r.annotate(
                f"FWHM = {_fwhm_meas*1e3:.1f} pm  (target {_fwhm*1e3:.0f} pm)",
                xy=(_lam0, _y_mid),
                xytext=(_lam0 + _fsr * 0.2, _y_mid - 0.15),
                fontsize=8.5, color=_col,
                arrowprops=dict(arrowstyle="-|>", color=_col, lw=1.0),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=_col,
                          alpha=0.85, lw=0.7),
            )
            _ax_r.hlines(_half_p, _lam_l, _lam_r,
                         colors=_col, lw=1.2, ls=":", alpha=0.6)
        except IndexError:
            pass

        _ax_r.annotate(
            "", xy=(_lam0 + _fsr, 0.95), xytext=(_lam0, 0.95),
            xycoords=("data", "axes fraction"),
            textcoords=("data", "axes fraction"),
            arrowprops=dict(arrowstyle="<->", color="dimgray", lw=1.2),
        )
        _ax_r.text(
            _lam0 + _fsr / 2, 0.98,
            f"FSR = {_fsr:.4f} nm",
            transform=_ax_r.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=8.5, color="dimgray",
        )

        _er_db = -10 * np.log10(max(_Tp.min(), 1e-12))

        # Parameter text box — α_loss added
        _info = (
            f"Ring: {cc_labels[_i]}\n"
            f"Cladding: {_clad}\n"
            f"λ_res  = {_lam0:.4f} nm\n"
            f"FSR    = {_fsr:.4f} nm\n"
            f"FWHM   = {_fwhm*1e3:.1f} pm  (design)\n"
            f"Q      = {_Q:.0f}\n"
            f"ER     = {_er_db:.2f} dB\n"
            f"neff   = {_neff:.6f}\n"
            f"ng     = {_ng:.6f}\n"
            f"α_loss = {_alpha:.4f} dB/m\n"   # ← NEW
            f"a      = {_a:.6f}\n"
            f"r₁     = {_r1:.6f}\n"
            f"r₂     = {_r2:.6f}\n"
            f"Gap_in  = {float(dc_gap_input_nm[_i]):.1f} nm\n"
            f"Gap_out = {float(dc_gap_output_nm[_i]):.1f} nm"
        )
        _ax_r.text(0.985, 0.98, _info,
                   transform=_ax_r.transAxes,
                   ha="right", va="top", fontsize=7.5,
                   family="monospace",
                   bbox=dict(boxstyle="round,pad=0.5", fc="white",
                             ec=_col, alpha=0.92, lw=0.8))

        _ax_r.set_title(
            f"Transmission — {cc_labels[_i]}  "
            f"({'Sensor' if _i == 0 else 'Spectrometer'})\n"
            f"R = {float(cc_R_pm_um[_i]):.4f} µm   "
            f"L = {float(cc_L_pm_um[_i]):.4f} µm   "
            f"Cladding: {_clad}   "
            f"neff = {_neff:.5f}   ng = {_ng:.5f}   α = {_alpha:.2f} dB/m",
            fontsize=10, fontweight="bold",
        )
        _ax_r.set_xlabel("Wavelength  λ  (nm)", fontsize=10)
        _ax_r.set_ylabel("Intensity transmission", fontsize=10)
        _ax_r.set_ylim(-0.05, 1.10)
        _ax_r.set_xlim(_lam_arr[0], _lam_arr[-1])
        _ax_r.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
        _ax_r.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        _ax_r.yaxis.set_minor_locator(ticker.AutoMinorLocator())

        _fig_r.tight_layout()
        _path_ri = DATA_DIR / f"{VERSION_NAME}_ring{_i:02d}_{cc_labels[_i]}_transmission.png"
        _fig_r.savefig(str(_path_ri), dpi=180, bbox_inches="tight",
                       facecolor="white")
        print(f"  Ring {_i:02d} saved → {_path_ri}")
        plt.show()
        plt.close(_fig_r)
    _x = np.arange(_N_RINGS)
    _w = 0.6
    _ring_labels = [str(cc_labels[i]) for i in range(_N_RINGS)]
    _fig_ov, _axes_ov = plt.subplots(2, 2, figsize=(18, 10))
    _fig_ov.suptitle(
        f"Photonic Platform Overview — 14-Ring Array\n"
        f"SiN {CORE_THICKNESS_UM*1e3:.0f} nm × {RR_WG_WIDTH_NM:.0f} nm  │  "
        f"α = {ALPHA_PROP_DB_CM:.1f} dB/cm  │  "
        f"Target FSR = 10 nm",
        fontsize=12, fontweight="bold", y=1.005,
    )
    _ax00 = _axes_ov[0, 0]
    _bars00 = _ax00.bar(_x, cc_lam_res_nm, width=_w,
                        color=_colors, edgecolor="k", lw=0.4, alpha=0.88)
    _ax00.set_xticks(_x)
    _ax00.set_xticklabels(_ring_labels, rotation=45, ha="right", fontsize=7.5)
    _ax00.set_ylabel("Resonance wavelength  λ_res  (nm)")
    _ax00.set_title("Resonance Wavelength per Ring")
    _ax00.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    for _b, _v in zip(_bars00, cc_lam_res_nm):
        _ax00.text(_b.get_x() + _b.get_width()/2, _v + 0.05,
                   f"{_v:.2f}", ha="center", va="bottom", fontsize=5.5, rotation=90)
    _ax01 = _axes_ov[0, 1]
    _bars01 = _ax01.bar(_x, _fsr_all_nm, width=_w,
                        color=_colors, edgecolor="k", lw=0.4, alpha=0.88)
    _ax01.set_xticks(_x)
    _ax01.set_xticklabels(_ring_labels, rotation=45, ha="right", fontsize=7.5)
    _ax01.set_ylabel("Free spectral range  FSR  (nm)")
    _ax01.set_title("FSR per Ring")
    _ax01.axhline(10.0, color="crimson", lw=1.2, ls="--", label="Target FSR = 10 nm")
    _ax01.legend(fontsize=8)
    _ax01.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    _ax10 = _axes_ov[1, 0]
    _fwhm_pm = cc_FWHM_nm * 1e3
    _bars10 = _ax10.bar(_x, _fwhm_pm, width=_w,
                        color=_colors, edgecolor="k", lw=0.4, alpha=0.88)
    _ax10.set_xticks(_x)
    _ax10.set_xticklabels(_ring_labels, rotation=45, ha="right", fontsize=7.5)
    _ax10.set_ylabel("FWHM  (pm)")
    _ax10.set_title("Linewidth (FWHM) per Ring")
    _ax10.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    for _b, _v in zip(_bars10, _fwhm_pm):
        _ax10.text(_b.get_x() + _b.get_width()/2, _v + 0.5,
                   f"{_v:.0f}", ha="center", va="bottom", fontsize=6.5)
    _ax11 = _axes_ov[1, 1]
    _bw   = 0.35
    _xin  = _x - _bw/2
    _xout = _x + _bw/2
    _ax11.bar(_xin,  dc_gap_input_nm,  width=_bw,
              color=_colors, edgecolor="k", lw=0.4, alpha=0.88, label="Input gap")
    _ax11.bar(_xout, dc_gap_output_nm, width=_bw,
              color=_colors, edgecolor="k", lw=0.4, alpha=0.55, hatch="//",
              label="Output gap")
    _ax11.set_xticks(_x)
    _ax11.set_xticklabels(_ring_labels, rotation=45, ha="right", fontsize=7.5)
    _ax11.set_ylabel("Directional coupler gap  (nm)")
    _ax11.set_title("Input & Output Coupling Gaps")
    _ax11.legend(fontsize=8.5)
    _ax11.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    _fig_ov.tight_layout()
    _path_ov = DATA_DIR / f"{VERSION_NAME}_platform_overview.png"
    _fig_ov.savefig(str(_path_ov), dpi=180, bbox_inches="tight",
                    facecolor="white")
    print(f"  Saved → {_path_ov}")
    plt.show()
    _SEP = "═" * 120
    print(f"\n{_SEP}")
    print("  PHOTONIC PLATFORM DESIGN SUMMARY  —  14-Ring Array")
    print(_SEP)
    print(f"  Waveguide  : SiN  {CORE_THICKNESS_UM*1e3:.0f} nm × {RR_WG_WIDTH_NM:.0f} nm")
    print(f"  Prop. loss : {ALPHA_PROP_DB_CM:.1f} dB/cm")
    print(f"  FWHM target: sensor = {FWHM_SENSOR_NM:.2f} nm   spec = {FWHM_SPEC_NM:.2f} nm")
    print(f"  Target FSR : 10.0 nm  (step = 10/13 ≈ 0.769 nm)")
    print(f"  α_loss     : α_prop + α_bend (field → power: ×20/ln10)  [dB/m]")
    print(f"\n  {'Ring':<12}{'Type':<12}{'λ_res (nm)':>13}{'R (µm)':>10}{'FSR (nm)':>11}"
          f"{'FWHM (pm)':>11}{'Q':>9}{'neff':>10}{'ng':>10}"
          f"{'α_loss':>10}{'Gap_in':>9}{'Gap_out':>10}")
    print(f"  {'':12}{'':12}{'':13}{'':10}{'':11}"
          f"{'':11}{'':9}{'':10}{'':10}"
          f"{'(dB/m)':>10}{'(nm)':>9}{'(nm)':>10}")
    print("  " + "─" * 116)
    for _i in range(_N_RINGS):
        _type = "Sensor" if _i == 0 else f"Spec {_i:02d}"
        print(
            f"  {cc_labels[_i]:<12}{_type:<12}"
            f"{cc_lam_res_nm[_i]:>13.4f}"
            f"{cc_R_pm_um[_i]:>10.4f}"
            f"{_fsr_all_nm[_i]:>11.4f}"
            f"{cc_FWHM_nm[_i]*1e3:>11.1f}"
            f"{cc_Q_target[_i]:>9.0f}"
            f"{cc_neff_pm[_i]:>10.6f}"
            f"{cc_ng[_i]:>10.6f}"
            f"{cc_loss_dB_m[_i]:>10.4f}"   # ← NEW
            f"{dc_gap_input_nm[_i]:>9.1f}"
            f"{dc_gap_output_nm[_i]:>10.1f}"
        )
    print(_SEP)
    print("\n  Figures saved:")
    print(f"    {_path_t1}")
    print(f"    {_path_t2}")
    print(f"    {_path_tx}")
    print(f"    {_path_ov}")
    print(f"    (+ 14 individual ring transmission PNGs)")
    print(_SEP)

    state.update({k: globals().get(k) for k in [
        '_L', '_LN10', '_Lc', '_N_PTS', '_N_RINGS', '_Q',
        '_R', '_SEP', '_Td', '_Tp', '_a', '_alpha',
        '_ax', '_ax00', '_ax01', '_ax10', '_ax11', '_ax_r',
        '_ax_sensor', '_ax_t1', '_ax_t2', '_axes', '_axes_ov', '_b',
        '_bars00', '_bars01', '_bars10', '_base_loss', '_base_neff', '_base_ng',
        '_bw', '_cbar', '_cell', '_clad', '_clad_str', '_cmap',
        '_cnorm', '_col', '_colors', '_dn_i', '_dn_o', '_er_db',
        '_face', '_fig_all', '_fig_ov', '_fig_r', '_fig_t1', '_fig_t2',
        '_fsr', '_fsr_all_nm', '_fwhm', '_fwhm_meas', '_fwhm_pm', '_gap_i',
        '_gap_o', '_gs', '_half_p', '_half_power', '_i', '_info',
        '_j', '_k1a', '_k1sq', '_k2a', '_k2sq', '_l_idx',
        '_lam', '_lam0', '_lam_arr', '_lam_l', '_lam_r', '_lam_span',
        '_left_idx', '_loss_col_idx', '_neff', '_neff_col_idx', '_ng', '_ng_col_idx',
        '_path_ov', '_path_ri', '_path_t1', '_path_t2', '_path_tx', '_phi',
        '_r1', '_r2', '_r_idx', '_right_idx', '_ring_labels', '_ring_types',
        '_row', '_sm', '_t1_cols', '_t1_rows', '_t2_cols', '_t2_rows',
        '_table1', '_table2', '_type', '_v', '_w', '_x',
        '_xin', '_xout', '_y_fsr', '_y_fwhm', '_y_mid', 'cc_alpha_field',
        'cc_loss_dB_m', 'cc_neff_pm', 'cc_ng', 'i',
    ] if k in globals()})
    return state


if __name__ == "__main__":
    run()
