"""
step6_critical_coupling.py — analytic critical-coupling engineering.

From a target FWHM (-> Q) computes round-trip amplitude a, self/cross
coupling r,k and the supermode index split for each of the 14 rings.
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
FWHM_SENSOR_NM = None
FWHM_SPEC_NM = None
L_pm_um = None
RR_LAM0_NM = None
RR_WG_WIDTH_NM = None
R_pm_um = None
k = None
neff_pm = None
r = None
rr_FSR_pred_nm = None
rr_best_ng = None
spec_FSR_pm_nm = None
spec_L_pm_um = None
spec_R_pm_um = None
spec_lam_res_pm_nm = None
spec_neff_pm = None
spec_ng_pm = None

apply_style()

# ─────────────────────────────────────────────────────────
#  Module constants / design knobs for this step
# ─────────────────────────────────────────────────────────
FWHM_SPEC_OVERRIDE  = {}         # example: {0: 0.15, 12: 0.25}
ALPHA_PROP_DB_CM    = 1.0        # [dB/cm]  bulk SiN propagation loss
FRAC_A              = 1.0 / 3.0  # first  Lc option: R_pm / 3
FRAC_B              = 1.0 / 2.0  # second Lc option: R_pm / 2
FWHM_TO_FSR_MIN     = 0.005      # FWHM ≥ 0.5 % of FSR
FWHM_TO_FSR_MAX     = 0.30       # FWHM ≤ 30 % of FSR
PRINT_LOSS_DETAIL   = True       # show bend + propagation loss breakdown
ALPHA_BEND_DB_CM_SENSOR = 0.01   # [dB/cm]  estimated bend radiation, sensor ring
ALPHA_BEND_DB_CM_SPEC   = 0.01   # [dB/cm]  estimated bend radiation, all spec rings
_ln10_over_10 = np.log(10) / 10.0          # ≈ 0.2303 Np/dB

def _db_cm_to_alpha_field(db_cm):
    """Convert dB/cm loss coefficient to field (amplitude) α [1/m]."""
    alpha_power_npm = db_cm * _ln10_over_10 * 100.0   # [Np/m]
    return alpha_power_npm / 2.0                        # field attenuation


def _compute_a(L_um, alpha_field_total):
    """
    Round-trip amplitude transmission.
      a = exp(−α_field_total × L)
    L_um in µm, alpha_field_total in 1/m.
    Returns a (dimensionless, 0 < a < 1).
    """
    L_m = L_um * 1e-6
    return float(np.exp(-alpha_field_total * L_m))


def _solve_critical_coupling(Q, lam_res_m, ng, L_um, a):
    """
    Solve for r1, r2, k1, k2 satisfying:
      - Target Q-factor for add-drop resonator
      - Asymmetric critical coupling: r1 = r2·a
      - Lossless coupler: k² + r² = 1

    Parameters
    ----------
    Q        : float  target loaded Q-factor
    lam_res_m: float  resonance wavelength [m]
    ng       : float  group index
    L_um     : float  round-trip length [µm]
    a        : float  round-trip amplitude transmission (0 < a < 1)

    Returns
    -------
    dict with keys: r1, r2, k1, k2, x, a, Q_achieved, FWHM_nm, valid
    """
    L_m  = L_um * 1e-6
    ngL  = ng * L_m                       # [m]

    # Quadratic: Q·λ·x² + π·ng·L·x − Q·λ = 0
    A    = Q * lam_res_m
    B    = np.pi * ngL
    disc = B**2 + 4.0 * A**2             # B² − 4AC = B² + 4Q²λ² > 0 always
    x    = (-B + np.sqrt(disc)) / (2.0 * A)   # x = r1

    r1   = float(np.clip(x, 0.0, 1.0 - 1e-9))
    r2   = float(np.clip(r1 / a, 0.0, 1.0 - 1e-9))
    k1   = float(np.sqrt(max(0.0, 1.0 - r1**2)))
    k2   = float(np.sqrt(max(0.0, 1.0 - r2**2)))

    # Verify achieved Q and FWHM
    prod     = r1 * r2 * a               # r1·r2·a
    if prod  < 1.0:
        Q_ach    = np.pi * ngL * np.sqrt(prod) / (lam_res_m * (1.0 - prod))
    else:
        Q_ach    = np.inf
    FWHM_nm  = lam_res_m / Q_ach * 1e9

    valid = (0.0 < r1 < 1.0) and (0.0 < r2 < 1.0) and (r2 >= r1)

    return {
        "r1":        r1,
        "r2":        r2,
        "k1":        k1,
        "k2":        k2,
        "x":         x,
        "a":         a,
        "Q_target":  Q,
        "Q_achieved":Q_ach,
        "FWHM_nm":   FWHM_nm,
        "valid":     valid,
    }


def _compute_delta_n(k, lam_m, Lc_m):
    """
    Required supermode index difference for coupling coefficient k
    in a directional coupler of length Lc_m at wavelength lam_m.

    Returns Δn (dimensionless) or NaN if k > 1 or Lc_m ≤ 0.
    """
    if Lc_m <= 0.0 or k < 0.0:
        return float("nan")
    if k > 1.0:
        return float("nan")
    arg = float(np.clip(k, 0.0, 1.0))
    return lam_m * np.arcsin(arg) / (np.pi * Lc_m)

def run(state=None):
    """Execute this step. `state` carries the shared namespace
    between steps (the notebook's old kernel globals + bridge).
    Returns the updated `state`."""
    state = {} if state is None else state
    global ALPHA_BEND_DB_CM_SENSOR, ALPHA_BEND_DB_CM_SPEC, ALPHA_PROP_DB_CM, FRAC_A, FRAC_B, FWHM_SPEC_OVERRIDE, FWHM_TO_FSR_MAX, FWHM_TO_FSR_MIN, Lc_a_m, Lc_a_um, Lc_b_m, Lc_b_um, PRINT_LOSS_DETAIL, Q_target, _H1, _H2, _H3, _H4, _L_um, _Lc, _R_pm_um, _SEP, _SEP2, _a, _a_v, _all_rings, _alpha_bend_sensor, _alpha_bend_spec, _alpha_prop_field, _alpha_total, _cc, _cc_results, _dn_in_a_v, _dn_in_b_v, _dn_out_a_v, _dn_out_b_v, _ext, _fig_stem, _fsr_nm, _fwhm, _fwhm_nm, _fwhm_ratio, _k1_v, _k2_v, _labels, _lam, _lam_res_m, _lam_v, _ln10_over_10, _n, _n_invalid, _nd_in, _nd_out, _ng, _r, _r1_v, _r2_v, _ratio_in, _ratio_out, _res, _ring, _sensor, _spec_rings, _vn, _x_idx, _x_labels, ax00, ax01, ax10, ax11, axes, cc_FWHM_nm, cc_L_pm_um, cc_Lc_a_um, cc_Lc_b_um, cc_Q_target, cc_R_pm_um, cc_a, cc_dn_in_a, cc_dn_in_b, cc_dn_out_a, cc_dn_out_b, cc_k1, cc_k2, cc_labels, cc_lam_res_nm, cc_r1, cc_r2, dn_in_a, dn_in_b, dn_out_a, dn_out_b, fig
    globals().update(state)

    _alpha_prop_field  = _db_cm_to_alpha_field(ALPHA_PROP_DB_CM)
    _alpha_bend_sensor = _db_cm_to_alpha_field(ALPHA_BEND_DB_CM_SENSOR)
    _alpha_bend_spec   = _db_cm_to_alpha_field(ALPHA_BEND_DB_CM_SPEC)
    log.info(
        f"Loss model │ α_prop = {ALPHA_PROP_DB_CM:.2f} dB/cm  "
        f"→ α_field = {_alpha_prop_field:.4f} 1/m  │  "
        f"α_bend_sensor = {ALPHA_BEND_DB_CM_SENSOR:.3f} dB/cm  "
        f"α_bend_spec = {ALPHA_BEND_DB_CM_SPEC:.3f} dB/cm"
    )
    _sensor = {
        "label":      "Sensor",
        "ring_type":  "sensor",
        "lam_res_nm": float(RR_LAM0_NM),
        "R_pm_um":    float(R_pm_um),
        "L_pm_um":    float(L_pm_um),
        "neff_pm":    float(neff_pm),
        "ng":         float(rr_best_ng),         # from Cell 8 bent FDE
        "FSR_nm":     float(rr_FSR_pred_nm),
        "fwhm_nm":    float(FWHM_SENSOR_NM),
        "alpha_bend": _alpha_bend_sensor,
    }
    _spec_rings = []
    for _n in range(N_SPEC_RINGS):
        _fwhm = FWHM_SPEC_OVERRIDE.get(_n, FWHM_SPEC_NM)
        _spec_rings.append({
            "label":      f"Spec_{_n:02d}",
            "ring_type":  "spectrometer",
            "ring_idx":   _n,
            "lam_res_nm": float(spec_lam_res_pm_nm[_n]),
            "R_pm_um":    float(spec_R_pm_um[_n]),
            "L_pm_um":    float(spec_L_pm_um[_n]),
            "neff_pm":    float(spec_neff_pm[_n]),
            "ng":         float(spec_ng_pm[_n]),
            "FSR_nm":     float(spec_FSR_pm_nm[_n]),
            "fwhm_nm":    float(_fwhm),
            "alpha_bend": _alpha_bend_spec,
        })
    _all_rings = [_sensor] + _spec_rings   # 14 entries total
    _cc_results = []    # list of result dicts, one per ring
    for _ring in _all_rings:
        _lam_res_m  = _ring["lam_res_nm"] * 1e-9
        _L_um       = _ring["L_pm_um"]
        _ng         = _ring["ng"]
        _fwhm_nm    = _ring["fwhm_nm"]
        _fsr_nm     = _ring["FSR_nm"]
        _R_pm_um    = _ring["R_pm_um"]

        # ── Validate FWHM against FSR guard rails ─────────────────────────────────
        _fwhm_ratio = _fwhm_nm / _fsr_nm
        if not (FWHM_TO_FSR_MIN <= _fwhm_ratio <= FWHM_TO_FSR_MAX):
            log.warning(
                f"  {_ring['label']:>10} │ FWHM/FSR = {_fwhm_ratio:.4f}  "
                f"(outside [{FWHM_TO_FSR_MIN}, {FWHM_TO_FSR_MAX}]) — proceeding anyway."
            )

        # ── Target Q ──────────────────────────────────────────────────────────────
        Q_target = _ring["lam_res_nm"] / _fwhm_nm

        # ── Round-trip loss ───────────────────────────────────────────────────────
        _alpha_total = _alpha_prop_field + _ring["alpha_bend"]
        _a           = _compute_a(_L_um, _alpha_total)

        # ── Solve for r1, r2 at critical coupling ─────────────────────────────────
        _cc = _solve_critical_coupling(Q_target, _lam_res_m, _ng, _L_um, _a)

        # ── Coupling section lengths ──────────────────────────────────────────────
        Lc_a_um = FRAC_A * _R_pm_um
        Lc_b_um = FRAC_B * _R_pm_um
        Lc_a_m  = Lc_a_um * 1e-6
        Lc_b_m  = Lc_b_um * 1e-6

        # ── Required Δn for each coupler (input and output) ───────────────────────
        # Input coupler (k1):
        dn_in_a  = _compute_delta_n(_cc["k1"], _lam_res_m, Lc_a_m)
        dn_in_b  = _compute_delta_n(_cc["k1"], _lam_res_m, Lc_b_m)
        # Output coupler (k2):
        dn_out_a = _compute_delta_n(_cc["k2"], _lam_res_m, Lc_a_m)
        dn_out_b = _compute_delta_n(_cc["k2"], _lam_res_m, Lc_b_m)

        # ── Assemble result ───────────────────────────────────────────────────────
        _res = {
            **_ring,
            "Q_target":    Q_target,
            "a":           _a,
            "a_dB":        -20.0 * np.log10(_a),           # round-trip amplitude loss [dB]
            "alpha_total": _alpha_total,
            "r1":          _cc["r1"],
            "r2":          _cc["r2"],
            "k1":          _cc["k1"],
            "k2":          _cc["k2"],
            "Q_achieved":  _cc["Q_achieved"],
            "FWHM_check_nm": _cc["FWHM_nm"],
            "valid":       _cc["valid"],
            # Coupling lengths [µm]
            "Lc_a_um":     Lc_a_um,
            "Lc_b_um":     Lc_b_um,
            # Required Δn (supermode index difference)
            "dn_in_a":     dn_in_a,          # input coupler,  Lc = R/3
            "dn_in_b":     dn_in_b,          # input coupler,  Lc = R/2
            "dn_out_a":    dn_out_a,         # output coupler, Lc = R/3
            "dn_out_b":    dn_out_b,         # output coupler, Lc = R/2
        }
        _cc_results.append(_res)

        if PRINT_LOSS_DETAIL:
            log.info(
                f"  {_ring['label']:>10} │ λ={_ring['lam_res_nm']:.4f} nm │ "
                f"R={_R_pm_um:.4f} µm │ L={_L_um:.4f} µm │ "
                f"α_tot={_alpha_total:.4f} 1/m │ a={_a:.6f} │ "
                f"Q={Q_target:.1f} │ r1={_cc['r1']:.6f} │ r2={_cc['r2']:.6f} │ "
                f"k1={_cc['k1']:.6f} │ k2={_cc['k2']:.6f} │ "
                f"valid={_cc['valid']}"
            )
    _SEP  = "─" * 148
    _SEP2 = "═" * 148
    print("\n\n")
    print(_SEP2)
    print("  CRITICAL COUPLING DESIGN — ANALYTICAL SUMMARY")
    print(f"  SiN platform  │  w={RR_WG_WIDTH_NM:.0f} nm  h={CORE_THICKNESS_UM*1e3:.0f} nm  │  "
          f"α_prop={ALPHA_PROP_DB_CM:.2f} dB/cm  │  "
          f"FWHM_sensor={FWHM_SENSOR_NM:.3f} nm  FWHM_spec={FWHM_SPEC_NM:.3f} nm")
    print(f"  Coupling fractions:  Lc_a = R_pm / {1/FRAC_A:.1f}  │  "
          f"Lc_b = R_pm / {1/FRAC_B:.1f}")
    print(_SEP2)
    print()
    print("  TABLE 1 — Round-trip loss and target Q-factor")
    print(_SEP)
    _H1 = (f"  {'Ring':>10}  {'λ_res (nm)':>13}  {'R_pm (µm)':>11}  "
           f"{'L_pm (µm)':>11}  {'α [1/m]':>9}  {'a':>10}  "
           f"{'a (dB)':>8}  {'FWHM (nm)':>11}  {'Q_target':>12}  {'Q_check':>12}")
    print(_H1)
    print("  " + "─" * 144)
    for _r in _cc_results:
        print(
            f"  {_r['label']:>10}  {_r['lam_res_nm']:>13.6f}  "
            f"{_r['R_pm_um']:>11.6f}  {_r['L_pm_um']:>11.6f}  "
            f"{_r['alpha_total']:>9.4f}  {_r['a']:>10.8f}  "
            f"{_r['a_dB']:>8.5f}  {_r['fwhm_nm']:>11.5f}  "
            f"{_r['Q_target']:>12.1f}  {_r['Q_achieved']:>12.1f}"
        )
    print()
    print("  TABLE 2 — Asymmetric critical coupling coefficients  (r2·a = r1)")
    print(_SEP)
    _H2 = (f"  {'Ring':>10}  {'λ_res (nm)':>13}  "
           f"{'a':>10}  {'r1':>10}  {'r2':>10}  "
           f"{'k1':>10}  {'k2':>10}  "
           f"{'k1²':>10}  {'k2²':>10}  {'valid':>6}")
    print(_H2)
    print("  " + "─" * 144)
    for _r in _cc_results:
        print(
            f"  {_r['label']:>10}  {_r['lam_res_nm']:>13.6f}  "
            f"{_r['a']:>10.8f}  {_r['r1']:>10.8f}  {_r['r2']:>10.8f}  "
            f"{_r['k1']:>10.8f}  {_r['k2']:>10.8f}  "
            f"{_r['k1']**2:>10.8f}  {_r['k2']**2:>10.8f}  "
            f"{'✓' if _r['valid'] else '✗':>6}"
        )
    print()
    print(f"  TABLE 3 — Required supermode Δn  (Lc = R_pm/{1/FRAC_A:.0f})")
    print(_SEP)
    _H3 = (f"  {'Ring':>10}  {'λ_res (nm)':>13}  "
           f"{'Lc_a (µm)':>11}  "
           f"{'Δn_in (k1)':>14}  {'Δn_out (k2)':>14}  "
           f"{'k1':>10}  {'k2':>10}  "
           f"{'Lc·Δn_in/λ':>13}  {'Lc·Δn_out/λ':>13}")
    print(_H3)
    print("  " + "─" * 144)
    for _r in _cc_results:
        _lam = _r["lam_res_nm"] * 1e-9
        _Lc  = _r["Lc_a_um"] * 1e-6
        _nd_in  = _r["dn_in_a"]
        _nd_out = _r["dn_out_a"]
        _ratio_in  = (_nd_in  * _Lc / _lam) if not np.isnan(_nd_in)  else float("nan")
        _ratio_out = (_nd_out * _Lc / _lam) if not np.isnan(_nd_out) else float("nan")
        print(
            f"  {_r['label']:>10}  {_r['lam_res_nm']:>13.6f}  "
            f"{_r['Lc_a_um']:>11.4f}  "
            f"{_nd_in:>14.8f}  {_nd_out:>14.8f}  "
            f"{_r['k1']:>10.8f}  {_r['k2']:>10.8f}  "
            f"{_ratio_in:>13.6f}  {_ratio_out:>13.6f}"
        )
    print()
    print(f"  TABLE 4 — Required supermode Δn  (Lc = R_pm/{1/FRAC_B:.0f})")
    print(_SEP)
    _H4 = (f"  {'Ring':>10}  {'λ_res (nm)':>13}  "
           f"{'Lc_b (µm)':>11}  "
           f"{'Δn_in (k1)':>14}  {'Δn_out (k2)':>14}  "
           f"{'k1':>10}  {'k2':>10}  "
           f"{'Lc·Δn_in/λ':>13}  {'Lc·Δn_out/λ':>13}")
    print(_H4)
    print("  " + "─" * 144)
    for _r in _cc_results:
        _lam = _r["lam_res_nm"] * 1e-9
        _Lc  = _r["Lc_b_um"] * 1e-6
        _nd_in  = _r["dn_in_b"]
        _nd_out = _r["dn_out_b"]
        _ratio_in  = (_nd_in  * _Lc / _lam) if not np.isnan(_nd_in)  else float("nan")
        _ratio_out = (_nd_out * _Lc / _lam) if not np.isnan(_nd_out) else float("nan")
        print(
            f"  {_r['label']:>10}  {_r['lam_res_nm']:>13.6f}  "
            f"{_r['Lc_b_um']:>11.4f}  "
            f"{_nd_in:>14.8f}  {_nd_out:>14.8f}  "
            f"{_r['k1']:>10.8f}  {_r['k2']:>10.8f}  "
            f"{_ratio_in:>13.6f}  {_ratio_out:>13.6f}"
        )
    print()
    print(_SEP2)
    _n_invalid = sum(1 for _r in _cc_results if not _r["valid"])
    if _n_invalid:
        log.warning(
            f"  {_n_invalid} ring(s) have invalid coupling solutions. "
            "Check FWHM targets — they may require r2 > 1 (too narrow for this loss level)."
        )
    else:
        log.info("  All 14 rings have valid critical-coupling solutions.")
    print()
    print("  PHYSICAL INTERPRETATION GUIDE")
    print("  " + "─" * 90)
    print(f"  r1 < r2  always (r2 = r1/a > r1 since a < 1) — asymmetric coupling required.")
    print(f"  k1 > k2  because r1 < r2  (input coupler stronger than output coupler).")
    print(f"  Δn_in > Δn_out  because k1 > k2 at the same Lc.")
    print(f"  Δn  is the even–odd supermode index splitting in the coupling section.")
    print(f"  Smaller gap between waveguides → larger Δn.")
    print(f"  Target range for realisable SiN gap couplers: 0.001 < Δn < 0.05.")
    print(f"  Values outside this range indicate Lc needs adjustment.")
    print()
    plt.rcParams.update({
        "figure.dpi":        150,
        "font.family":       "DejaVu Sans",
        "axes.titlesize":    11,
        "axes.labelsize":    10,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "legend.fontsize":   8,
        "lines.linewidth":   1.8,
        "lines.markersize":  6,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.28,
        "grid.linestyle":    "--",
    })
    _labels     = [_r["label"]      for _r in _cc_results]
    _lam_v      = [_r["lam_res_nm"] for _r in _cc_results]
    _r1_v       = [_r["r1"]         for _r in _cc_results]
    _r2_v       = [_r["r2"]         for _r in _cc_results]
    _k1_v       = [_r["k1"]         for _r in _cc_results]
    _k2_v       = [_r["k2"]         for _r in _cc_results]
    _a_v        = [_r["a"]          for _r in _cc_results]
    _dn_in_a_v  = [_r["dn_in_a"]    for _r in _cc_results]
    _dn_out_a_v = [_r["dn_out_a"]   for _r in _cc_results]
    _dn_in_b_v  = [_r["dn_in_b"]    for _r in _cc_results]
    _dn_out_b_v = [_r["dn_out_b"]   for _r in _cc_results]
    _x_idx      = np.arange(len(_cc_results))
    _x_labels   = [r["label"] for r in _cc_results]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f"Critical Coupling Design — SiN  │  "
        f"w={RR_WG_WIDTH_NM:.0f} nm  h={CORE_THICKNESS_UM*1e3:.0f} nm  │  "
        f"α={ALPHA_PROP_DB_CM:.1f} dB/cm  │  FWHM={FWHM_SENSOR_NM:.2f}/{FWHM_SPEC_NM:.2f} nm",
        fontsize=11, fontweight="bold", y=1.01,
    )
    ax00 = axes[0, 0]
    ax00.plot(_x_idx, _r1_v, "o-", color="#0072B2", lw=1.8, ms=5, label=r"$r_1$ (input)")
    ax00.plot(_x_idx, _r2_v, "s-", color="#D55E00", lw=1.8, ms=5, label=r"$r_2$ (output)")
    ax00.plot(_x_idx, _a_v,  "D-", color="#009E73", lw=1.4, ms=4, label=r"$a$ (roundtrip)")
    ax00.set_xticks(_x_idx)
    ax00.set_xticklabels(_x_labels, rotation=45, ha="right", fontsize=7)
    ax00.set_ylabel("Self-coupling coefficient")
    ax00.set_title(r"$r_1$, $r_2$, $a$ per ring  (critical coupling: $r_2 \cdot a = r_1$)")
    ax00.legend()
    ax00.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax01 = axes[0, 1]
    ax01.plot(_x_idx, _k1_v, "o-", color="#0072B2", lw=1.8, ms=5, label=r"$k_1$ (input)")
    ax01.plot(_x_idx, _k2_v, "s-", color="#D55E00", lw=1.8, ms=5, label=r"$k_2$ (output)")
    ax01.set_xticks(_x_idx)
    ax01.set_xticklabels(_x_labels, rotation=45, ha="right", fontsize=7)
    ax01.set_ylabel("Cross-coupling coefficient  k")
    ax01.set_title(r"Cross-coupling coefficients  $k_1$, $k_2$  ($k^2 = 1 - r^2$)")
    ax01.legend()
    ax01.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax10 = axes[1, 0]
    ax10.semilogy(_x_idx, _dn_in_a_v,  "o-", color="#0072B2", lw=1.8, ms=5,
                  label=rf"$\Delta n_\mathrm{{in}}$  $L_c = R/3$")
    ax10.semilogy(_x_idx, _dn_in_b_v,  "o--", color="#56B4E9", lw=1.6, ms=4,
                  label=rf"$\Delta n_\mathrm{{in}}$  $L_c = R/2$")
    ax10.semilogy(_x_idx, _dn_out_a_v, "s-", color="#D55E00", lw=1.8, ms=5,
                  label=rf"$\Delta n_\mathrm{{out}}$  $L_c = R/3$")
    ax10.semilogy(_x_idx, _dn_out_b_v, "s--", color="#E69F00", lw=1.6, ms=4,
                  label=rf"$\Delta n_\mathrm{{out}}$  $L_c = R/2$")
    ax10.axhspan(0.001, 0.05, alpha=0.10, color="#009E73",
                 label="Typical realisable Δn (0.001–0.05)")
    ax10.set_xticks(_x_idx)
    ax10.set_xticklabels(_x_labels, rotation=45, ha="right", fontsize=7)
    ax10.set_ylabel(r"Required supermode $\Delta n$")
    ax10.set_title(r"Required $\Delta n$ for input and output directional couplers")
    ax10.legend(fontsize=7, ncol=2)
    ax10.yaxis.set_minor_locator(ticker.LogLocator(subs=[2, 5]))
    ax11 = axes[1, 1]
    ax11.plot(_lam_v, [k**2 for k in _k1_v], "o-", color="#0072B2",
              lw=1.8, ms=5, label=r"$k_1^2$ (input)")
    ax11.plot(_lam_v, [k**2 for k in _k2_v], "s-", color="#D55E00",
              lw=1.8, ms=5, label=r"$k_2^2$ (output)")
    ax11.set_xlabel(r"Ring resonance wavelength  $\lambda_\mathrm{res}$  (nm)")
    ax11.set_ylabel("Power coupling ratio  $k^2$")
    ax11.set_title(r"Power coupling ratios $k_1^2$, $k_2^2$ vs wavelength")
    ax11.legend()
    ax11.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax11.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    plt.tight_layout()
    _fig_stem = DATA_DIR / f"{VERSION_NAME}_critical_coupling_design"
    for _ext in (".png", ".pdf"):
        plt.savefig(str(_fig_stem) + _ext, dpi=150, bbox_inches="tight")
        print(f"  Saved → {str(_fig_stem) + _ext}")
    plt.show()
    cc_labels      = [_r["label"]      for _r in _cc_results]
    cc_lam_res_nm  = np.array([_r["lam_res_nm"]  for _r in _cc_results])
    cc_R_pm_um     = np.array([_r["R_pm_um"]     for _r in _cc_results])
    cc_L_pm_um     = np.array([_r["L_pm_um"]     for _r in _cc_results])
    cc_a           = np.array([_r["a"]           for _r in _cc_results])
    cc_r1          = np.array([_r["r1"]          for _r in _cc_results])
    cc_r2          = np.array([_r["r2"]          for _r in _cc_results])
    cc_k1          = np.array([_r["k1"]          for _r in _cc_results])
    cc_k2          = np.array([_r["k2"]          for _r in _cc_results])
    cc_Q_target    = np.array([_r["Q_target"]    for _r in _cc_results])
    cc_FWHM_nm     = np.array([_r["fwhm_nm"]     for _r in _cc_results])
    cc_Lc_a_um     = np.array([_r["Lc_a_um"]     for _r in _cc_results])
    cc_Lc_b_um     = np.array([_r["Lc_b_um"]     for _r in _cc_results])
    cc_dn_in_a     = np.array([_r["dn_in_a"]     for _r in _cc_results])   # Δn input,  Lc=R/3
    cc_dn_in_b     = np.array([_r["dn_in_b"]     for _r in _cc_results])   # Δn input,  Lc=R/2
    cc_dn_out_a    = np.array([_r["dn_out_a"]    for _r in _cc_results])   # Δn output, Lc=R/3
    cc_dn_out_b    = np.array([_r["dn_out_b"]    for _r in _cc_results])   # Δn output, Lc=R/2
    print()
    print("  Exported arrays (shape = (14,), index 0 = sensor ring):")
    for _vn in [
        "cc_labels", "cc_lam_res_nm", "cc_R_pm_um", "cc_L_pm_um",
        "cc_a", "cc_r1", "cc_r2", "cc_k1", "cc_k2",
        "cc_Q_target", "cc_FWHM_nm",
        "cc_Lc_a_um", "cc_Lc_b_um",
        "cc_dn_in_a", "cc_dn_in_b", "cc_dn_out_a", "cc_dn_out_b",
    ]:
        print(f"    {_vn}")
    print()
    print("  Next step: gap sweep using FDE supermode analysis to find the")
    print("  waveguide separation that yields each target Δn at the coupling")
    print("  section wavelength (CELL 12).")

    state.update({k: globals().get(k) for k in [
        'ALPHA_BEND_DB_CM_SENSOR', 'ALPHA_BEND_DB_CM_SPEC', 'ALPHA_PROP_DB_CM', 'FRAC_A', 'FRAC_B', 'FWHM_SPEC_OVERRIDE',
        'FWHM_TO_FSR_MAX', 'FWHM_TO_FSR_MIN', 'Lc_a_m', 'Lc_a_um', 'Lc_b_m', 'Lc_b_um',
        'PRINT_LOSS_DETAIL', 'Q_target', '_H1', '_H2', '_H3', '_H4',
        '_L_um', '_Lc', '_R_pm_um', '_SEP', '_SEP2', '_a',
        '_a_v', '_all_rings', '_alpha_bend_sensor', '_alpha_bend_spec', '_alpha_prop_field', '_alpha_total',
        '_cc', '_cc_results', '_dn_in_a_v', '_dn_in_b_v', '_dn_out_a_v', '_dn_out_b_v',
        '_ext', '_fig_stem', '_fsr_nm', '_fwhm', '_fwhm_nm', '_fwhm_ratio',
        '_k1_v', '_k2_v', '_labels', '_lam', '_lam_res_m', '_lam_v',
        '_ln10_over_10', '_n', '_n_invalid', '_nd_in', '_nd_out', '_ng',
        '_r', '_r1_v', '_r2_v', '_ratio_in', '_ratio_out', '_res',
        '_ring', '_sensor', '_spec_rings', '_vn', '_x_idx', '_x_labels',
        'ax00', 'ax01', 'ax10', 'ax11', 'axes', 'cc_FWHM_nm',
        'cc_L_pm_um', 'cc_Lc_a_um', 'cc_Lc_b_um', 'cc_Q_target', 'cc_R_pm_um', 'cc_a',
        'cc_dn_in_a', 'cc_dn_in_b', 'cc_dn_out_a', 'cc_dn_out_b', 'cc_k1', 'cc_k2',
        'cc_labels', 'cc_lam_res_nm', 'cc_r1', 'cc_r2', 'dn_in_a', 'dn_in_b',
        'dn_out_a', 'dn_out_b', 'fig',
    ] if k in globals()})
    return state


if __name__ == "__main__":
    run()
