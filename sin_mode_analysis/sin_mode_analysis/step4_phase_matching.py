"""
step4_phase_matching.py — phase-matching correction vs straight ng.

Analytic phase-matching (no new FDE): straight-waveguide ng by linear fit,
resonance order m, and the self-consistent phase-matched radius.
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
from scipy.interpolate import interp1d

from .config import *  # shared platform constants & paths
from .plotting import apply_style, save_fig, make_colorbar
from . import plotting as _plot

# ── Data-contract inputs (injected by main.py via `state`) ──────────
RR_FSR_NM = None
RR_LAM0_NM = None
RR_WG_WIDTH_NM = None
_lam0_m = None
_neff_arr = None
_ng_arr = None
_radii_um = None
_te_arr = None
neff_real = None
rr_FSR_pred_nm = None
rr_best_L_um = None
rr_best_R_um = None
rr_best_neff = None
rr_best_ng = None
rr_best_ngL_um = None
te_frac = None

apply_style()

# ─────────────────────────────────────────────────────────
#  Module constants / design knobs for this step
# ─────────────────────────────────────────────────────────


def _row(label, v1, v2, fmt=".4f"):
    print(f"  {label:<42}  {v1:>15{fmt}}  {v2:>20{fmt}}")

def run(state=None):
    """Execute this step. `state` carries the shared namespace
    between steps (the notebook's old kernel globals + bridge).
    Returns the updated `state`."""
    state = {} if state is None else state
    global FSR_pm_nm, L_pm_um, R_pm_0_m, R_pm_0_um, R_pm_um, _C_BEND, _C_BEST, _C_PM, _C_RES, _C_STR, _C_TGT, _FSR_pm_all, _L_pm_all_um, _L_v_m, _R_it, _R_pm_all_um, _R_pm_m, _R_pm_new, _R_v_m, _R_v_um_f, _SEP, _SEP2, _b, _bars, _bi_f, _colors, _correction_history, _dR, _delta_R, _delta_lam, _dlam_str_m, _dneff_2pt, _dneff_dlam_straight, _ext, _fig_stem, _interp_neff, _iter, _k, _labels, _lam0_idx, _lam_res_all, _lam_res_v, _m_cont, _m_int, _max_iter, _n_it, _n_iter, _neff_at_R, _neff_hi_str, _neff_lo_str, _neff_straight_lam0, _neff_v_b, _neff_vs_wl, _ng_max, _ng_min, _ng_shift_pct, _ng_straight_2pt, _ng_v_b, _ng_vals, _phase_resid_nm, _poly_coeffs, _resid_all_pm, _rr_w_um, _te_mean, _te_v_b, _te_vs_wl, _tol_m, _v, _valid_r, _valid_wl, _w_actual_nm, _w_diffs, _w_idx, _wl_m_arr, ax0, ax1, ax2, axes, fig, lam_res_pm_nm, m_best, neff_best, neff_pm, ng_best, ng_straight, phase_resid_pm_pm
    globals().update(state)

    _rr_w_um  = RR_WG_WIDTH_NM * 1e-3              # 1.000 µm
    _w_diffs  = np.abs(SWEEP_WIDTHS_UM - _rr_w_um)
    _w_idx    = int(np.argmin(_w_diffs))
    _w_actual_nm = SWEEP_WIDTHS_UM[_w_idx] * 1e3
    log.info(
        f"Straight ng extraction │ target width = {RR_WG_WIDTH_NM:.0f} nm  "
        f"→ closest sweep index = {_w_idx}  "
        f"({_w_actual_nm:.2f} nm)"
    )
    _neff_vs_wl = neff_real[_w_idx, :, 0].copy()   # shape (13,)
    _te_vs_wl   = te_frac  [_w_idx, :, 0].copy()
    _te_mean = float(np.nanmean(_te_vs_wl))
    if _te_mean < 0.5:
        log.warning(
            f"Mode 0 at w={_w_actual_nm:.0f} nm appears TM (mean TE frac={_te_mean:.3f}). "
            "Switching to mode index 1."
        )
        _neff_vs_wl = neff_real[_w_idx, :, 1].copy()
        _te_vs_wl   = te_frac  [_w_idx, :, 1].copy()
    _wl_m_arr = SWEEP_WL_NM * 1e-9              # shape (13,)   [m]
    _lam0_idx = int(np.argmin(np.abs(SWEEP_WL_NM - RR_LAM0_NM)))   # index of λ₀
    _valid_wl  = ~np.isnan(_neff_vs_wl)
    _poly_coeffs = np.polyfit(
        _wl_m_arr[_valid_wl],
        _neff_vs_wl[_valid_wl],
        deg=1,                  # linear: neff ≈ c0 + c1·λ
    )
    _dneff_dlam_straight = _poly_coeffs[0]          # c1  [1/m]
    _neff_straight_lam0  = float(np.polyval(_poly_coeffs, _lam0_m))
    ng_straight = _neff_straight_lam0 - _lam0_m * _dneff_dlam_straight
    _neff_lo_str = _neff_vs_wl[0]    # λ = 1550.000 nm
    _neff_hi_str = _neff_vs_wl[-1]   # λ = 1559.231 nm
    _dlam_str_m  = (_wl_m_arr[-1] - _wl_m_arr[0])
    _dneff_2pt   = (_neff_hi_str - _neff_lo_str) / _dlam_str_m
    _ng_straight_2pt = _neff_vs_wl[_lam0_idx] - _lam0_m * _dneff_2pt
    log.info(
        f"ng_straight (linear fit)  = {ng_straight:.6f}  |  "
        f"ng_straight (2-pt check)  = {_ng_straight_2pt:.6f}  |  "
        f"neff @ λ₀ = {_neff_straight_lam0:.6f}"
    )
    _valid_r   = ~np.isnan(_neff_arr)
    _R_v_m     = _radii_um[_valid_r] * 1e-6        # [m]
    _neff_v_b  = _neff_arr[_valid_r]               # bent neff
    _ng_v_b    = _ng_arr  [_valid_r]               # bent ng
    _te_v_b    = _te_arr  [_valid_r]
    _L_v_m     = 2.0 * np.pi * _R_v_m
    _m_cont    = _neff_v_b * _L_v_m / _lam0_m
    _m_int     = np.round(_m_cont).astype(int)
    _lam_res_v = (_neff_v_b * _L_v_m) / _m_int     # [m]
    _delta_lam = (_lam_res_v - _lam0_m) * 1e9       # [nm]  shift from λ₀
    _phase_resid_nm = np.abs(_delta_lam)             # [nm]
    _R_v_um_f  = _radii_um[_valid_r]
    _bi_f      = int(np.argmin(np.abs(_R_v_um_f - rr_best_R_um)))
    m_best     = int(_m_int[_bi_f])
    neff_best  = float(_neff_v_b[_bi_f])
    ng_best    = float(_ng_v_b[_bi_f])
    log.info(f"Best radius  R = {rr_best_R_um:.4f} µm  →  m = {m_best}")
    R_pm_0_m  = m_best * _lam0_m / (2.0 * np.pi * neff_best)
    R_pm_0_um = R_pm_0_m * 1e6
    _interp_neff = interp1d(
        _R_v_m, _neff_v_b,
        kind="linear",
        bounds_error=False,
        fill_value=(_neff_v_b[0], _neff_v_b[-1]),   # extrapolate flat
    )
    _tol_m    = 1e-13          # convergence: < 0.1 nm
    _R_pm_m   = R_pm_0_m       # initial guess
    _max_iter  = 20
    _n_iter    = 0
    _correction_history = [(_R_pm_m, neff_best)]
    for _iter in range(_max_iter):
        _neff_at_R   = float(_interp_neff(_R_pm_m))
        _R_pm_new    = m_best * _lam0_m / (2.0 * np.pi * _neff_at_R)
        _correction_history.append((_R_pm_new, _neff_at_R))
        _delta_R     = abs(_R_pm_new - _R_pm_m)
        _R_pm_m      = _R_pm_new
        _n_iter     += 1
        if _delta_R < _tol_m:
            break
    R_pm_um    = _R_pm_m * 1e6
    neff_pm    = float(_interp_neff(_R_pm_m))
    L_pm_um    = 2.0 * np.pi * R_pm_um
    FSR_pm_nm  = (_lam0_m**2 / (ng_best * L_pm_um * 1e-6)) * 1e9
    lam_res_pm_nm = (neff_pm * L_pm_um * 1e-6 / m_best) * 1e9
    phase_resid_pm_pm = abs(lam_res_pm_nm - RR_LAM0_NM) * 1e3   # [pm]
    log.info(
        f"Phase-matched radius  R_pm = {R_pm_um:.4f} µm  "
        f"(converged in {_n_iter} iterations,  "
        f"λ_res = {lam_res_pm_nm:.6f} nm,  "
        f"residual = {phase_resid_pm_pm:.4f} pm)"
    )
    _R_pm_all_um  = m_best * _lam0_m / (2.0 * np.pi * _neff_v_b) * 1e6
    _L_pm_all_um  = 2.0 * np.pi * _R_pm_all_um
    _FSR_pm_all   = (_lam0_m**2 / (_ng_v_b * _L_pm_all_um * 1e-6)) * 1e9
    _lam_res_all  = (_neff_v_b * 2.0 * np.pi * _R_v_um_f * 1e-6 / _m_int) * 1e9
    _resid_all_pm = np.abs(_lam_res_all - RR_LAM0_NM) * 1e3   # [pm]
    _SEP  = "─" * 82
    _SEP2 = "═" * 82
    print()
    print(_SEP2)
    print("  RING RESONATOR DESIGN SUMMARY  —  Phase Matching Correction")
    print(_SEP2)
    print(f"  Platform  : SiN  {CORE_THICKNESS_UM*1e3:.0f} nm × {RR_WG_WIDTH_NM:.0f} nm  │  "
          f"λ₀ = {RR_LAM0_NM:.1f} nm  │  Target FSR = {RR_FSR_NM:.1f} nm")
    print(_SEP)
    print()
    print("  GROUP INDEX COMPARISON")
    print(f"  {'Source':<42}  {'neff':>10}  {'ng':>10}  {'dneff/dλ [1/µm]':>18}")
    print("  " + "─" * 78)
    print(
        f"  {'Straight FDE (aqueous, linear fit)':<42}  "
        f"{_neff_straight_lam0:>10.6f}  "
        f"{ng_straight:>10.6f}  "
        f"{_dneff_dlam_straight*1e-6:>18.6f}"
    )
    print(
        f"  {'Straight FDE (2-point cross-check)':<42}  "
        f"{float(_neff_vs_wl[_lam0_idx]):>10.6f}  "
        f"{_ng_straight_2pt:>10.6f}  "
        f"{'—':>18}"
    )
    print(
        f"  {'Bent FDE @ R = ' + f'{rr_best_R_um:.4f} µm (FSR-matched)':<42}  "
        f"{rr_best_neff:>10.6f}  "
        f"{rr_best_ng:>10.6f}  "
        f"{'—':>18}"
    )
    print(
        f"  {'Bent FDE @ R_pm = ' + f'{R_pm_um:.4f} µm (phase-matched)':<42}  "
        f"{neff_pm:>10.6f}  "
        f"{ng_best:>10.6f}  "
        f"{'—':>18}"
    )
    _ng_shift_pct = (rr_best_ng - ng_straight) / ng_straight * 100.0
    print()
    print(f"  Δng (bend vs straight) = {rr_best_ng - ng_straight:+.6f}  "
          f"({_ng_shift_pct:+.3f} %)   ← chromatic confinement shift from bending")
    print()
    print(_SEP)
    print()
    print("  PHASE MATCHING CORRECTION")
    print(f"  {'Parameter':<42}  {'FSR-matched R':>15}  {'Phase-matched R_pm':>20}")
    print("  " + "─" * 78)
    _row("Radius  R  (µm)",               rr_best_R_um,    R_pm_um)
    _row("Ring length  L = 2πR  (µm)",    rr_best_L_um,    L_pm_um)
    _row("Effective index  neff",          rr_best_neff,    neff_pm,    fmt=".6f")
    _row("Group index  ng",                rr_best_ng,      ng_best,    fmt=".6f")
    _row("Resonance order  m",             float(m_best),   float(m_best), fmt=".0f")
    _row("λ_res = neff·L/m  (nm)",
         (rr_best_neff * rr_best_L_um * 1e-6 / m_best) * 1e9,
         lam_res_pm_nm)
    _row("Δλ from target λ₀  (pm)",
         abs(rr_best_neff * rr_best_L_um * 1e-6 / m_best - _lam0_m) * 1e12,
         phase_resid_pm_pm)
    _row("ng·L  (µm)",                     rr_best_ngL_um,  ng_best * L_pm_um)
    _row("FSR = λ₀² / (ng·L)  (nm)",       rr_FSR_pred_nm,  FSR_pm_nm)
    _row("FSR error vs target  (pm)",
         abs(rr_FSR_pred_nm - RR_FSR_NM) * 1e3,
         abs(FSR_pm_nm      - RR_FSR_NM) * 1e3)
    print()
    print(_SEP)
    print()
    print(f"  Phase-matching correction  ΔR = R_pm − R_best = "
          f"{(R_pm_um - rr_best_R_um) * 1e3:+.2f} nm  "
          f"({(R_pm_um - rr_best_R_um) / rr_best_R_um * 100:+.4f} %)")
    print(f"  Convergence               : {_n_iter} iterations  │  "
          f"λ_res residual = {phase_resid_pm_pm:.4f} pm")
    print(f"  Resonance order m = {m_best}  →  λ₀ = {m_best} × {RR_LAM0_NM / m_best * 1e3:.4f} pm")
    print()
    print(_SEP2)
    print()
    print("  ITERATIVE CONVERGENCE  (R_pm self-consistent refinement)")
    print(f"  {'Iter':>5}  {'R  (µm)':>14}  {'neff':>12}  {'ΔR  (nm)':>12}")
    print("  " + "─" * 50)
    for _k, (_R_it, _n_it) in enumerate(_correction_history):
        _dR = (_R_it - _correction_history[0][0]) * 1e9 if _k > 0 else 0.0
        print(f"  {_k:>5}  {_R_it*1e6:>14.6f}  {_n_it:>12.8f}  {_dR:>+12.4f}")
    print()
    print("  Variables exported:")
    print(f"    m_best          = {m_best}")
    print(f"    R_pm_um         = {R_pm_um:.6f}  # µm  (phase-matched)")
    print(f"    neff_pm         = {neff_pm:.6f}")
    print(f"    L_pm_um         = {L_pm_um:.6f}  # µm")
    print(f"    FSR_pm_nm       = {FSR_pm_nm:.6f}  # nm")
    print(f"    ng_straight     = {ng_straight:.6f}  (from straight 2D FDE)")
    print(f"    lam_res_pm_nm   = {lam_res_pm_nm:.6f}  # nm  (resonance @ R_pm)")
    print()
    plt.rcParams.update({
        "figure.dpi":        150,
        "font.family":       "DejaVu Sans",
        "axes.titlesize":    11,
        "axes.labelsize":    10,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "legend.fontsize":   8,
        "lines.linewidth":   2.0,
        "lines.markersize":  5,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.28,
        "grid.linestyle":    "--",
    })
    _C_STR  = "#0072B2"   # straight ng — blue
    _C_BEND = "#D55E00"   # bent neff/ng — vermilion
    _C_PM   = "#009E73"   # phase-matched — green
    _C_BEST = "#CC79A7"   # FSR-matched best — pink
    _C_TGT  = "#E69F00"   # target — amber
    _C_RES  = "#56B4E9"   # residual — sky
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        f"Ring Resonator Phase-Matching Correction\n"
        f"SiN {CORE_THICKNESS_UM*1e3:.0f} nm × {RR_WG_WIDTH_NM:.0f} nm  │  "
        f"λ₀ = {RR_LAM0_NM:.1f} nm  │  Target FSR = {RR_FSR_NM:.1f} nm",
        fontsize=12, fontweight="bold", y=1.02,
    )
    ax0 = axes[0]
    ax0.plot(
        SWEEP_WL_NM, _neff_vs_wl,
        "o-", color=_C_STR, label=f"Straight  w={_w_actual_nm:.0f} nm",
    )
    ax0.axhline(
        rr_best_neff, color=_C_BEND, ls="--", lw=1.6,
        label=f"Bent @ R={rr_best_R_um:.2f} µm  (FSR-match)",
    )
    ax0.axhline(
        neff_pm, color=_C_PM, ls=":", lw=1.8,
        label=f"Bent @ R_pm={R_pm_um:.4f} µm  (phase-match)",
    )
    ax0.axvline(
        RR_LAM0_NM, color=_C_TGT, ls=":", lw=1.3, alpha=0.7,
        label=f"λ₀ = {RR_LAM0_NM:.0f} nm",
    )
    ax0.set_xlabel("Wavelength  λ  (nm)")
    ax0.set_ylabel(r"Effective index  $n_\mathrm{eff}$")
    ax0.set_title(r"$n_\mathrm{eff}$  vs  Wavelength")
    ax0.legend(loc="lower right")
    ax0.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax0.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax1 = axes[1]
    _labels = [
        f"Straight\n(2D FDE\nlinear fit)",
        f"Straight\n(2D FDE\n2-point)",
        f"Bent FDE\nR={rr_best_R_um:.2f} µm\n(FSR-match)",
        f"Bent FDE\nR_pm={R_pm_um:.4f} µm\n(phase-match)",
    ]
    _ng_vals = [ng_straight, _ng_straight_2pt, rr_best_ng, ng_best]
    _colors  = [_C_STR, _C_STR, _C_BEND, _C_PM]
    _bars    = ax1.bar(
        range(4), _ng_vals,
        color=_colors, width=0.55, edgecolor="k", linewidth=0.6, alpha=0.88,
    )
    for _b, _v in zip(_bars, _ng_vals):
        ax1.text(
            _b.get_x() + _b.get_width() / 2,
            _v + 0.0003,
            f"{_v:.5f}",
            ha="center", va="bottom", fontsize=7.5,
        )
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(_labels, fontsize=7.5)
    ax1.set_ylabel(r"Group index  $n_g$")
    ax1.set_title(r"$n_g$ Comparison — Straight vs Bent")
    _ng_min = min(_ng_vals) - 0.003
    _ng_max = max(_ng_vals) + 0.006
    ax1.set_ylim(_ng_min, _ng_max)
    ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax1.grid(True, axis="y", alpha=0.28, linestyle="--")
    ax1.grid(False, axis="x")
    ax2 = axes[2]
    ax2.semilogy(
        _R_v_um_f, _resid_all_pm,
        "o-", color=_C_RES, markersize=3,
        label="|Δλ_res| = |neff·L/m − λ₀|",
    )
    ax2.axvline(
        rr_best_R_um, color=_C_BEST, ls="--", lw=1.6,
        label=f"FSR-matched  R = {rr_best_R_um:.2f} µm  "
              f"({_resid_all_pm[_bi_f]:.1f} pm)",
    )
    ax2.axvline(
        R_pm_um, color=_C_PM, ls=":", lw=1.8,
        label=f"Phase-matched  R_pm = {R_pm_um:.4f} µm  "
              f"({phase_resid_pm_pm:.4f} pm)",
    )
    ax2.scatter(
        [rr_best_R_um], [_resid_all_pm[_bi_f]],
        s=60, zorder=5, color=_C_BEST, edgecolors="k", linewidths=0.7,
    )
    ax2.scatter(
        [R_pm_um], [phase_resid_pm_pm],
        s=80, marker="*", zorder=6, color=_C_PM, edgecolors="k", linewidths=0.7,
    )
    ax2.set_xlabel("Bend radius  R  (µm)")
    ax2.set_ylabel("|Δλ_res|  (pm)  [log scale]")
    ax2.set_title("Phase-matching residual  |λ_res − λ₀|")
    ax2.legend(loc="upper right", fontsize=7.5)
    ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    plt.tight_layout()
    _fig_stem = DATA_DIR / f"{VERSION_NAME}_phase_match_correction"
    for _ext in (".png", ".pdf"):
        plt.savefig(str(_fig_stem) + _ext, dpi=150, bbox_inches="tight")
        print(f"  Saved → {str(_fig_stem) + _ext}")
    plt.show()

    state.update({k: globals().get(k) for k in [
        'FSR_pm_nm', 'L_pm_um', 'R_pm_0_m', 'R_pm_0_um', 'R_pm_um', '_C_BEND',
        '_C_BEST', '_C_PM', '_C_RES', '_C_STR', '_C_TGT', '_FSR_pm_all',
        '_L_pm_all_um', '_L_v_m', '_R_it', '_R_pm_all_um', '_R_pm_m', '_R_pm_new',
        '_R_v_m', '_R_v_um_f', '_SEP', '_SEP2', '_b', '_bars',
        '_bi_f', '_colors', '_correction_history', '_dR', '_delta_R', '_delta_lam',
        '_dlam_str_m', '_dneff_2pt', '_dneff_dlam_straight', '_ext', '_fig_stem', '_interp_neff',
        '_iter', '_k', '_labels', '_lam0_idx', '_lam_res_all', '_lam_res_v',
        '_m_cont', '_m_int', '_max_iter', '_n_it', '_n_iter', '_neff_at_R',
        '_neff_hi_str', '_neff_lo_str', '_neff_straight_lam0', '_neff_v_b', '_neff_vs_wl', '_ng_max',
        '_ng_min', '_ng_shift_pct', '_ng_straight_2pt', '_ng_v_b', '_ng_vals', '_phase_resid_nm',
        '_poly_coeffs', '_resid_all_pm', '_rr_w_um', '_te_mean', '_te_v_b', '_te_vs_wl',
        '_tol_m', '_v', '_valid_r', '_valid_wl', '_w_actual_nm', '_w_diffs',
        '_w_idx', '_wl_m_arr', 'ax0', 'ax1', 'ax2', 'axes',
        'fig', 'lam_res_pm_nm', 'm_best', 'neff_best', 'neff_pm', 'ng_best',
        'ng_straight', 'phase_resid_pm_pm',
    ] if k in globals()})
    return state


if __name__ == "__main__":
    run()
