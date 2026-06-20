"""
step7_coupler_gap.py — directional-coupler gap sweep (supermodes).

Two-waveguide FDE supermode sweep that finds the physical gap giving the
target k1,k2 for each ring's input and drop couplers (brentq root-find).
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
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from .config import *  # shared platform constants & paths
from .lumerical_session import import_lumapi
from .plotting import apply_style, save_fig, make_colorbar
from . import plotting as _plot

# ── Data-contract inputs (injected by main.py via `state`) ──────────
FRAC_A = None
FRAC_B = None
_exc = None
cc_Lc_a_um = None
cc_Lc_b_um = None
cc_dn_in_a = None
cc_dn_in_b = None
cc_dn_out_a = None
cc_dn_out_b = None
cc_k1 = None
cc_k2 = None
cc_labels = None
cc_lam_res_nm = None
i = None
j = None
n = None
selected_width_nm = None

apply_style()

# ─────────────────────────────────────────────────────────
#  Module constants / design knobs for this step
# ─────────────────────────────────────────────────────────
_DC_N_RINGS    = 14         # 1 sensor (index 0) + 13 spectrometer (indices 1-13)
_DC_N_SPEC     = 13         # spectrometer rings only
DC_GAP_MIN_NM  = 150.0      # [nm]  minimum gap — fabrication lower bound
DC_GAP_MAX_NM  = 450.0      # [nm]  maximum gap — evanescent overlap negligible
DC_N_GAPS      = 100        # number of sweep points per coupler
DC_LC_SELECTOR = "a"
DC_N_MODES     = max(N_MODES_REQUEST, 6)
_DC_WG_W_NM    = (float(WG_WIDTH_OVERRIDE_NM) if WG_WIDTH_OVERRIDE_NM is not None
                  else float(WG_WIDTH_FALLBACK_NM))   # provisional; resolved in run()
_DC_WG_W_UM    = _DC_WG_W_NM * 1e-3        # 1.000 µm
_DC_WG_W_M     = _DC_WG_W_NM * 1e-9        # metres
_DC_MARGIN_UM  = 2.0        # [µm]  cladding margin each side beyond outer waveguide
DC_Y_SPAN_UM   = 2.0 * _DC_MARGIN_UM + 2.0 * _DC_WG_W_UM + DC_GAP_MAX_NM * 1e-3
DC_Y_SPAN_UM   = 8.0
_DC_SIM_Y_SPAN_ORIG_UM = 6.5       # SIM_Y_SPAN_UM from Cell 1 (known from output)
_DC_MESH_STEP_UM = _DC_SIM_Y_SPAN_ORIG_UM / MESH_CELLS_Y
DC_MESH_Y      = int(np.ceil(DC_Y_SPAN_UM / _DC_MESH_STEP_UM))
DC_MESH_Z      = MESH_CELLS_Z      # Z stack is identical to all previous cells
_DC_CORE_T_UM   = CORE_THICKNESS_UM         # 0.400 µm
_DC_HALF_T_UM   = _DC_CORE_T_UM / 2.0      # 0.200 µm
_DC_Z_BELOW_UM  = SIM_Z_BELOW_UM            # 2.0 µm
_DC_Z_ABOVE_UM  = SIM_Z_ABOVE_UM            # 2.0 µm
_DC_Z_SPAN_UM   = _DC_Z_BELOW_UM + _DC_CORE_T_UM + _DC_Z_ABOVE_UM   # 4.4 µm
_DC_SIO2_Z_CTR  = -(_DC_HALF_T_UM + _DC_Z_BELOW_UM / 2.0)            # -1.1 µm
_DC_SIO2_Z_SPAN = _DC_Z_BELOW_UM            # 2.0 µm
_DC_Z_CTR       = (_DC_Z_ABOVE_UM - _DC_Z_BELOW_UM) / 2.0            # 0.0 µm
_dc_gaps_nm = np.linspace(DC_GAP_MIN_NM, DC_GAP_MAX_NM, DC_N_GAPS)
_dc_gaps_m  = _dc_gaps_nm * 1e-9

def _dc_build_fde(mode, gap_m: float, wavelength_m: float,
                  n_upper_clad: float) -> None:
    """
    Build a 2D X-normal FDE for a symmetric side-by-side directional coupler.

    Parameters
    ----------
    mode          : lumapi.MODE session
    gap_m         : float  waveguide edge-to-edge gap [m]
    wavelength_m  : float  simulation wavelength [m]
    n_upper_clad  : float  background / upper cladding index
                           1.33 for sensor ring (aqueous)
                           1.4469 for spectrometer rings (SiO₂)
    """
    m = mode
    m.switchtolayout()
    m.selectall()
    m.delete()

    # Centre of each waveguide core (measured from y = 0)
    _y_ctr = _DC_WG_W_M / 2.0 + gap_m / 2.0

    # ── FDE solver (straight waveguide — no bent waveguide flag) ─────────────
    m.addfde()
    m.set("solver type",           "2D X normal")
    m.set("x",                     0.0)
    m.set("y",                     0.0)
    m.set("z",                     _DC_Z_CTR       * 1e-6)
    m.set("y span",                DC_Y_SPAN_UM    * 1e-6)
    m.set("z span",                _DC_Z_SPAN_UM   * 1e-6)
    m.set("wavelength",            wavelength_m)
    m.set("number of trial modes", DC_N_MODES)
    m.set("mesh cells y",          DC_MESH_Y)
    m.set("mesh cells z",          DC_MESH_Z)
    # NO "bent waveguide" — coupling section is straight

    # ── Background: upper cladding (fills entire domain, lowest priority) ─────
    m.addrect()
    m.set("name",     "DC_bg")
    m.set("x",        0.0);  m.set("x span", 1.0e-6)
    m.set("y",        0.0);  m.set("y span", DC_Y_SPAN_UM  * 1e-6)
    m.set("z",        _DC_Z_CTR       * 1e-6)
    m.set("z span",   _DC_Z_SPAN_UM   * 1e-6)
    m.set("material", "<Object defined dielectric>")
    m.set("index",    n_upper_clad)

    # ── SiO₂ lower substrate slab (overrides background in −Z region) ────────
    m.addrect()
    m.set("name",     "DC_lower_clad")
    m.set("x",        0.0);  m.set("x span", 1.0e-6)
    m.set("y",        0.0);  m.set("y span", DC_Y_SPAN_UM  * 1e-6)
    m.set("z",        _DC_SIO2_Z_CTR  * 1e-6)
    m.set("z span",   _DC_SIO2_Z_SPAN * 1e-6)
    m.set("material", "<Object defined dielectric>")
    m.set("index",    N_SIO2_FIXED)

    # ── Left SiN waveguide  (y < 0 side) ─────────────────────────────────────
    m.addrect()
    m.set("name",     "DC_wg_L")
    m.set("x",        0.0);  m.set("x span", 1.0e-6)
    m.set("y",       -_y_ctr)
    m.set("y span",   _DC_WG_W_M)
    m.set("z",        0.0);  m.set("z span", _DC_CORE_T_UM * 1e-6)
    m.set("material", "<Object defined dielectric>")
    m.set("index",    N_SIN_FIXED)

    # ── Right SiN waveguide (y > 0 side) ─────────────────────────────────────
    m.addrect()
    m.set("name",     "DC_wg_R")
    m.set("x",        0.0);  m.set("x span", 1.0e-6)
    m.set("y",       +_y_ctr)
    m.set("y span",   _DC_WG_W_M)
    m.set("z",        0.0);  m.set("z span", _DC_CORE_T_UM * 1e-6)
    m.set("material", "<Object defined dielectric>")
    m.set("index",    N_SIN_FIXED)


def _dc_solve_supermodes(mode, gap_m: float, wavelength_m: float,
                         n_upper_clad: float):
    """
    One FDE solve → (n_even, n_odd, Δn, te_even, te_odd).

    Returns (nan, nan, nan, nan, nan) on any failure.

    Mode tracking:
      1. Request DC_N_MODES modes.
      2. Filter: TE fraction > 0.5  AND  Re(neff) > N_SIO2_FIXED
         (guided TE modes only; N_SIO2_FIXED is the tighter cutoff
          because even the SiO₂-clad platform uses this as the substrate).
      3. Sort by Re(neff) descending.
      4. Take the top two:
           mode[0] = even supermode (Ey symmetric,  higher neff)
           mode[1] = odd  supermode (Ey antisymmetric, lower neff)
      5. Δn = n_even − n_odd  ≥ 0
    """
    _dc_build_fde(mode, gap_m, wavelength_m, n_upper_clad)
    mode.run()
    mode.findmodes()

    _te_modes = []
    for _mi in range(1, DC_N_MODES + 1):
        try:
            _mn    = f"FDE::data::mode{_mi}"
            _nraw  = mode.getdata(_mn, "neff")
            _teraw = mode.getdata(_mn, "TE polarization fraction")
            _nr    = complex(np.asarray(_nraw).flat[0]).real
            _te    = float(np.asarray(_teraw).flat[0])
            # Keep only guided TE-like modes
            if _te > 0.5 and _nr > N_SIO2_FIXED:
                _te_modes.append((_nr, _te, _mi))
        except Exception:
            continue

    if len(_te_modes) < 2:
        return (float("nan"),) * 5

    # Sort by neff descending → even = index 0, odd = index 1
    _te_modes.sort(key=lambda x: x[0], reverse=True)
    n_even  = float(_te_modes[0][0])
    te_even = float(_te_modes[0][1])
    n_odd   = float(_te_modes[1][0])
    te_odd  = float(_te_modes[1][1])
    delta_n = max(0.0, n_even - n_odd)

    return n_even, n_odd, delta_n, te_even, te_odd


def _dc_make_grp_path(ring_label, coupler_type, lam_nm):
    """Full HDF5 group path for one coupler sweep."""
    key = (
        f"dc_{ring_label}_{coupler_type}_{lam_nm:.4f}nm_"
        f"{DC_GAP_MIN_NM:.0f}-{DC_GAP_MAX_NM:.0f}nm_{DC_N_GAPS}pts"
    )
    return f"directional_couplers/{key}"


def _dc_init_group(hf, grp_path, ring_label, coupler_type, lam_nm,
                   n_upper_clad, k_target, target_dn_a, target_dn_b,
                   Lc_a_um, Lc_b_um):
    """Pre-allocate datasets for one coupler sweep inside an open h5py file."""
    N  = DC_N_GAPS
    g  = hf.require_group(grp_path)
    mg = g.require_group("metadata")

    mg.attrs.setdefault("ring_label",      ring_label)
    mg.attrs.setdefault("coupler_type",    coupler_type)
    mg.attrs.setdefault("lam_nm",          float(lam_nm))
    mg.attrs.setdefault("n_upper_clad",    float(n_upper_clad))
    mg.attrs.setdefault("wg_width_nm",     _DC_WG_W_NM)
    mg.attrs.setdefault("wg_height_nm",    _DC_CORE_T_UM * 1e3)
    mg.attrs.setdefault("gap_min_nm",      DC_GAP_MIN_NM)
    mg.attrs.setdefault("gap_max_nm",      DC_GAP_MAX_NM)
    mg.attrs.setdefault("n_gaps",          N)
    mg.attrs.setdefault("target_dn_a",     float(target_dn_a))
    mg.attrs.setdefault("target_dn_b",     float(target_dn_b))
    mg.attrs.setdefault("Lc_a_um",         float(Lc_a_um))
    mg.attrs.setdefault("Lc_b_um",         float(Lc_b_um))
    mg.attrs.setdefault("k_target",        float(k_target))
    mg.attrs.setdefault("timestamp_start", datetime.now().isoformat())

    if "gaps_nm" not in mg:
        mg.create_dataset("gaps_nm", data=_dc_gaps_nm)

    _nan = np.full(N, np.nan, dtype=np.float64)
    rg   = g.require_group("results")
    for _ds in ("n_even", "n_odd", "delta_n", "te_even", "te_odd"):
        if _ds not in rg:
            rg.create_dataset(_ds, data=_nan.copy(), chunks=(N,))

    fg = g.require_group("flags")
    if "computed" not in fg:
        fg.create_dataset("computed", data=np.zeros(N, dtype=bool), chunks=(N,))


def _dc_load_cache(hf, grp_path):
    """
    Load all result arrays for one coupler sweep from an open h5py file.
    Returns (n_even, n_odd, delta_n, te_even, te_odd, computed) as np arrays.
    """
    g  = hf[grp_path]
    rg = g["results"]
    fg = g["flags"]
    return (
        rg["n_even"][:].copy(),
        rg["n_odd"] [:].copy(),
        rg["delta_n"][:].copy(),
        rg["te_even"][:].copy(),
        rg["te_odd"] [:].copy(),
        fg["computed"][:].copy(),
    )


def _dc_find_gap(gaps_nm, delta_n_arr, target_dn, lam_nm, Lc_um):
    """
    Interpolate Δn(gap) and find the gap [nm] where Δn = target_dn.

    Returns a dict:
      gap_nm          optimal gap in nm
      delta_n_at_gap  Δn at that gap from the interpolant
      k_achieved      sin(π·Δn·Lc/λ) at the optimal gap
      error_pct       |Δn_sim − target| / target × 100
      in_range        True if target falls within simulated Δn range
    """
    _valid = ~np.isnan(delta_n_arr)
    if _valid.sum() < 3:
        return {"gap_nm": float("nan"), "delta_n_at_gap": float("nan"),
                "k_achieved": float("nan"), "error_pct": float("nan"),
                "in_range": False}

    _g   = gaps_nm[_valid]
    _dn  = delta_n_arr[_valid]

    # Cubic interpolant — Δn is monotonically decreasing with gap
    _interp = interp1d(_g, _dn, kind="cubic",
                       bounds_error=False,
                       fill_value=(_dn[0], _dn[-1]))

    _in_range = (_dn[-1] <= target_dn <= _dn[0])

    if _in_range:
        try:
            _gap_opt = float(brentq(
                lambda g: float(_interp(g)) - target_dn,
                float(_g[0]), float(_g[-1]),
                xtol=0.01,      # 0.01 nm precision
                maxiter=200,
            ))
        except Exception:
            # brentq failed — fall back to nearest discrete point
            _gap_opt = float(_g[int(np.argmin(np.abs(_dn - target_dn)))])
    else:
        # Target outside range: return the closest boundary
        _gap_opt = float(_g[0] if target_dn > _dn[0] else _g[-1])

    _dn_opt = float(_interp(_gap_opt))

    # k = sin(π · Δn · Lc / λ)
    _lam_m  = lam_nm * 1e-9
    _Lc_m   = Lc_um  * 1e-6
    _arg    = float(np.clip(np.pi * _dn_opt * _Lc_m / _lam_m, 0.0, 1.0))
    _k_ach  = float(np.sin(_arg))

    _err    = abs(_dn_opt - target_dn) / target_dn * 100.0 \
              if target_dn > 0 else float("nan")

    return {
        "gap_nm":         _gap_opt,
        "delta_n_at_gap": _dn_opt,
        "k_achieved":     _k_ach,
        "error_pct":      _err,
        "in_range":       _in_range,
        "_interp_fn":     _interp,    # kept for plotting
    }

def run(state=None):
    """Execute this step. `state` carries the shared namespace
    between steps (the notebook's old kernel globals + bridge).
    Returns the updated `state`."""
    state = {} if state is None else state
    global DC_GAP_MAX_NM, DC_GAP_MIN_NM, DC_LC_SELECTOR, DC_MESH_Y, DC_MESH_Z, DC_N_GAPS, DC_N_MODES, DC_Y_SPAN_UM, _C_IN, _C_OUT, _DC_CORE_T_UM, _DC_HALF_T_UM, _DC_MARGIN_UM, _DC_MESH_STEP_UM, _DC_N_RINGS, _DC_N_SPEC, _DC_SIM_Y_SPAN_ORIG_UM, _DC_SIO2_Z_CTR, _DC_SIO2_Z_SPAN, _DC_WG_W_M, _DC_WG_W_NM, _DC_WG_W_UM, _DC_Z_ABOVE_UM, _DC_Z_BELOW_UM, _DC_Z_CTR, _DC_Z_SPAN_UM, _HDR, _Lc_a, _Lc_b, _Lc_sel, _S, _S2, _any_uncached, _ax, _c, _comp_arr, _ctype, _dc_gaps_m, _dc_gaps_nm, _dc_jobs, _dc_mode, _dn, _dn_a, _dn_arr, _dn_b, _dn_sel, _el, _eta, _ext, _f1, _f2, _gap_in, _gap_m, _gap_out, _gi, _grp, _hdf5_path, _hdr, _hf, _hf_init, _hf_r, _hpath, _in_j, _in_jobs, _inj, _is_sensor, _j, _ji, _job, _job_n_done, _k1_ach, _k1_tgt, _k2_ach, _k2_tgt, _k_tgt, _label, _lam_m, _lam_nm, _m, _n_done, _n_fully_cached, _n_oor, _n_up, _ne, _ne_arr, _no, _no_arr, _ok, _ouj, _out_j, _out_jobs, _plt_lbl, _plt_s, _rate, _remain, _rg, _ri, _ri_arr, _runs, _t0, _tee, _tee_arr, _teo, _teo_arr, _vi, _vn, _vo, ax2a, ax2b, axes1, dc_dn_input_sim, dc_dn_output_sim, dc_err_input_pct, dc_err_output_pct, dc_gap_input_nm, dc_gap_output_nm, dc_k1_achieved, dc_k2_achieved, fig1, fig2
    globals()['lumapi'] = import_lumapi()
    globals().update(state)

    if WG_WIDTH_OVERRIDE_NM is not None:
        _DC_WG_W_NM = float(WG_WIDTH_OVERRIDE_NM)
    elif selected_width_nm is not None:
        _DC_WG_W_NM = float(selected_width_nm)
    else:
        _DC_WG_W_NM = float(WG_WIDTH_FALLBACK_NM)
        log.warning("step7: no single-mode width from the modal step and no "
                    f"WG_WIDTH_OVERRIDE_NM; using fallback {_DC_WG_W_NM:.0f} nm.")
    _DC_WG_W_UM  = _DC_WG_W_NM * 1e-3
    _DC_WG_W_M   = _DC_WG_W_NM * 1e-9
    DC_Y_SPAN_UM = 2.0 * _DC_MARGIN_UM + 2.0 * _DC_WG_W_UM + DC_GAP_MAX_NM * 1e-3
    log.info(f"step7 single-mode working width = {_DC_WG_W_NM:.1f} nm "
             f"(override={WG_WIDTH_OVERRIDE_NM}, selected={selected_width_nm})")
    print("=" * 70)
    print("  DIRECTIONAL COUPLER GAP SWEEP — All 14 rings, input + output")
    print("=" * 70)
    print(f"  Gap range       : {DC_GAP_MIN_NM:.0f} – {DC_GAP_MAX_NM:.0f} nm  ({DC_N_GAPS} pts)")
    print(f"  Lc selector     : '{DC_LC_SELECTOR}'"
          f"  (Lc = R_pm × {FRAC_A if DC_LC_SELECTOR=='a' else FRAC_B:.4f})")
    print(f"  Total couplers  : {2 * _DC_N_RINGS}  ({_DC_N_RINGS} rings × 2)")
    print(f"  FDE domain Y    : {DC_Y_SPAN_UM:.1f} µm  ({DC_MESH_Y} cells)"
          f"  Z: {_DC_Z_SPAN_UM:.1f} µm  ({DC_MESH_Z} cells)")
    print(f"  Max solves      : {2 * _DC_N_RINGS * DC_N_GAPS} (if nothing cached)")
    print(f"  HDF5 files      : {HDF5_PATH.name} (sensor)")
    print(f"                    {HDF5_PATH_SIO2.name} (spectrometer)")
    print("=" * 70)
    print()
    _dc_jobs = []
    for _ri in range(_DC_N_RINGS):
        # Ring 0 = sensor (aqueous), rings 1-13 = spectrometer (SiO₂)
        _is_sensor  = (_ri == 0)
        _n_up       = N_UPPER_CLADDING     if _is_sensor else N_UPPER_CLADDING_SIO2
        _hdf5_path  = HDF5_PATH            if _is_sensor else HDF5_PATH_SIO2
        _lam_nm     = float(cc_lam_res_nm[_ri])
        _label      = cc_labels[_ri]
        _Lc_a       = float(cc_Lc_a_um[_ri])
        _Lc_b       = float(cc_Lc_b_um[_ri])

        for _ctype, _k_tgt, _dn_a, _dn_b in [
            ("input",  float(cc_k1[_ri]),
                       float(cc_dn_in_a[_ri]),  float(cc_dn_in_b[_ri])),
            ("output", float(cc_k2[_ri]),
                       float(cc_dn_out_a[_ri]), float(cc_dn_out_b[_ri])),
        ]:
            # Select target Δn and Lc based on DC_LC_SELECTOR
            _dn_sel  = _dn_a  if DC_LC_SELECTOR == "a" else _dn_b
            _Lc_sel  = _Lc_a  if DC_LC_SELECTOR == "a" else _Lc_b

            _grp = _dc_make_grp_path(_label, _ctype, _lam_nm)

            _dc_jobs.append({
                "ring_idx":     _ri,
                "label":        _label,
                "coupler_type": _ctype,
                "lam_nm":       _lam_nm,
                "n_upper_clad": _n_up,
                "hdf5_path":    _hdf5_path,
                "grp_path":     _grp,
                "Lc_a_um":      _Lc_a,
                "Lc_b_um":      _Lc_b,
                "target_dn_a":  _dn_a,
                "target_dn_b":  _dn_b,
                "target_dn":    _dn_sel,    # primary target for matching
                "Lc_sel_um":    _Lc_sel,    # coupling length matching DC_LC_SELECTOR
                "k_target":     _k_tgt,
            })
    log.info(
        f"Job list: {len(_dc_jobs)} coupler sweeps  "
        f"({DC_N_GAPS} gaps each  →  max {len(_dc_jobs)*DC_N_GAPS} FDE solves if uncached)"
    )
    for _hpath in [HDF5_PATH, HDF5_PATH_SIO2]:
        with h5py.File(_hpath, "a") as _hf_init:
            for _job in _dc_jobs:
                if _job["hdf5_path"] == _hpath:
                    if _job["grp_path"] not in _hf_init:
                        _dc_init_group(
                            _hf_init,
                            _job["grp_path"],
                            _job["label"],
                            _job["coupler_type"],
                            _job["lam_nm"],
                            _job["n_upper_clad"],
                            _job["k_target"],
                            _job["target_dn_a"],
                            _job["target_dn_b"],
                            _job["Lc_a_um"],
                            _job["Lc_b_um"],
                        )
            _hf_init.flush()
    _job_n_done = []
    _any_uncached = False
    for _job in _dc_jobs:
        with h5py.File(_job["hdf5_path"], "r") as _hf_r:
            _n_done = int(_hf_r[_job["grp_path"]]["flags"]["computed"][:].sum())
        _job_n_done.append(_n_done)
        if _n_done < DC_N_GAPS:
            _any_uncached = True
    _n_fully_cached = sum(1 for n in _job_n_done if n == DC_N_GAPS)
    log.info(f"Cache status: {_n_fully_cached}/{len(_dc_jobs)} jobs fully cached.")
    _dc_mode = None
    if _any_uncached:
        log.info("Launching Lumerical MODE session …")
        _dc_mode = lumapi.MODE(hide=False)
    for _ji, _job in enumerate(_dc_jobs):
        _n_done  = _job_n_done[_ji]
        _remain  = DC_N_GAPS - _n_done
        _label   = _job["label"]
        _ctype   = _job["coupler_type"]
        _lam_m   = _job["lam_nm"] * 1e-9
        _n_up    = _job["n_upper_clad"]
        _grp     = _job["grp_path"]

        log.info(
            f"  Job {_ji+1:02d}/{len(_dc_jobs)}  "
            f"{_label:>10}  {_ctype:>6}  "
            f"λ={_job['lam_nm']:.4f} nm  n_upper={_n_up:.4f}  "
            f"cached {_n_done}/{DC_N_GAPS}"
        )

        # Load cache into memory
        _hf = h5py.File(_job["hdf5_path"], "a")
        _ne_arr, _no_arr, _dn_arr, _tee_arr, _teo_arr, _comp_arr = \
            _dc_load_cache(_hf, _grp)

        # Print table header for this coupler
        _hdr = (f"  {'gap(nm)':>9}  {'n_even':>12}  {'n_odd':>12}  "
                f"{'Δn':>14}  {'te_e':>6}  {'te_o':>6}  src")
        print(f"\n  {_label}  {_ctype}  λ={_job['lam_nm']:.4f} nm"
              f"  target Δn={_job['target_dn']:.8f}")
        print(_hdr);  print("  " + "─" * (len(_hdr) - 2))

        # Print already-cached rows first
        for _gi in range(DC_N_GAPS):
            if _comp_arr[_gi]:
                print(
                    f"  {_dc_gaps_nm[_gi]:>9.2f}  {_ne_arr[_gi]:>12.8f}"
                    f"  {_no_arr[_gi]:>12.8f}  {_dn_arr[_gi]:>14.10f}"
                    f"  {_tee_arr[_gi]:>6.4f}  {_teo_arr[_gi]:>6.4f}  cache"
                )

        # FDE loop for remaining gaps
        if _remain > 0 and _dc_mode is not None:
            _t0   = time.time()
            _runs = 0

            for _gi, _gap_m in enumerate(_dc_gaps_m):
                if _comp_arr[_gi]:
                    continue

                try:
                    _ne, _no, _dn, _tee, _teo = _dc_solve_supermodes(
                        _dc_mode, _gap_m, _lam_m, _n_up
                    )
                except Exception as _exc:
                    log.warning(
                        f"  {_label} {_ctype} gap={_dc_gaps_nm[_gi]:.1f} nm  FAILED: {_exc}"
                    )
                    _ne = _no = _dn = _tee = _teo = float("nan")

                # Store in memory
                _ne_arr [_gi] = _ne
                _no_arr [_gi] = _no
                _dn_arr [_gi] = _dn
                _tee_arr[_gi] = _tee
                _teo_arr[_gi] = _teo
                _comp_arr[_gi] = True

                # HDF5 — write and flush immediately (fault-safe)
                _rg = _hf[f"{_grp}/results"]
                _rg["n_even"] [_gi] = _ne
                _rg["n_odd"]  [_gi] = _no
                _rg["delta_n"][_gi] = _dn
                _rg["te_even"][_gi] = _tee
                _rg["te_odd"] [_gi] = _teo
                _hf[f"{_grp}/flags/computed"][_gi] = True
                _hf.flush()

                _runs += 1
                print(
                    f"  {_dc_gaps_nm[_gi]:>9.2f}  {_ne:>12.8f}  {_no:>12.8f}"
                    f"  {_dn:>14.10f}  {_tee:>6.4f}  {_teo:>6.4f}  FDE"
                )

                if _runs % 10 == 0 or _runs == _remain:
                    _el   = time.time() - _t0
                    _rate = _runs / _el if _el > 0 else 1e-9
                    _eta  = (_remain - _runs) / _rate
                    log.info(
                        f"    [{_runs:3d}/{_remain}]  "
                        f"gap={_dc_gaps_nm[_gi]:.1f} nm  "
                        f"Δn={_dn:.8f}  ETA {_eta:.0f} s"
                    )

            _hf[_grp]["metadata"].attrs["timestamp_end"]  = datetime.now().isoformat()
            _hf[_grp]["metadata"].attrs["runs_completed"] = int(_comp_arr.sum())
            _hf.flush()

        _hf.close()

        # Attach final arrays to job dict for analysis and plotting
        _job["ne_arr"]  = _ne_arr.copy()
        _job["no_arr"]  = _no_arr.copy()
        _job["dn_arr"]  = _dn_arr.copy()
    if _dc_mode is not None:
        _dc_mode.close()
        log.info("Lumerical MODE session closed.")
    log.info("All 28 coupler sweeps complete (or loaded from cache).")
    for _job in _dc_jobs:
        _job["match"] = _dc_find_gap(
            _dc_gaps_nm,
            _job["dn_arr"],
            _job["target_dn"],
            _job["lam_nm"],
            _job["Lc_sel_um"],
        )
        _m = _job["match"]
        log.info(
            f"  {_job['label']:>10}  {_job['coupler_type']:>6}  "
            f"Δn_tgt={_job['target_dn']:.8f}  "
            f"gap={_m['gap_nm']:.2f} nm  "
            f"Δn_sim={_m['delta_n_at_gap']:.8f}  "
            f"err={_m['error_pct']:.3f}%  "
            f"k={_m['k_achieved']:.6f}  "
            f"{'IN RANGE' if _m['in_range'] else '⚠ OUT OF RANGE'}"
        )
    _S  = "─" * 158
    _S2 = "═" * 158
    print("\n\n")
    print(_S2)
    print("  DIRECTIONAL COUPLER SYNTHESIS — COMPLETE SUMMARY")
    print(
        f"  Gap sweep {DC_GAP_MIN_NM:.0f}–{DC_GAP_MAX_NM:.0f} nm  │  "
        f"Lc = R_pm × {FRAC_A if DC_LC_SELECTOR=='a' else FRAC_B:.4f}  │  "
        f"Asymmetric critical coupling: r₂·a = r₁  →  k₁ ≠ k₂"
    )
    print(_S2)
    print()
    _HDR = (
        f"  {'Ring':>10}  {'Plt':>5}  {'λ_res(nm)':>12}  {'Coupler':>7}  "
        f"{'k_target':>9}  {'Δn_target':>13}  "
        f"{'Gap_opt(nm)':>13}  {'Δn_sim':>13}  "
        f"{'err%':>7}  {'k_ach':>8}  {'Lc(µm)':>8}  {'OK':>4}"
    )
    print(_HDR)
    print("  " + "─" * 154)
    _in_jobs  = [j for j in _dc_jobs if j["coupler_type"] == "input"]
    _out_jobs = [j for j in _dc_jobs if j["coupler_type"] == "output"]
    for _in_j, _out_j in zip(_in_jobs, _out_jobs):
        _plt_lbl = "Aq" if _in_j["ring_idx"] == 0 else "SiO₂"
        for _j in (_in_j, _out_j):
            _m = _j["match"]
            _ok = "✓" if _m["in_range"] else "⚠"
            print(
                f"  {_j['label']:>10}  {_plt_lbl:>5}  {_j['lam_nm']:>12.6f}  "
                f"{_j['coupler_type']:>7}  {_j['k_target']:>9.6f}  "
                f"{_j['target_dn']:>13.9f}  "
                f"{_m['gap_nm']:>13.3f}  {_m['delta_n_at_gap']:>13.9f}  "
                f"{_m['error_pct']:>7.3f}  {_m['k_achieved']:>8.6f}  "
                f"{_j['Lc_sel_um']:>8.4f}  {_ok:>4}"
            )
        print("  " + "─" * 154)
    print()
    print(_S2)
    print()
    _n_oor = sum(1 for j in _dc_jobs if not j["match"]["in_range"])
    if _n_oor:
        log.warning(
            f"{_n_oor} coupler(s) have target Δn outside the simulated gap range. "
            "Consider extending DC_GAP_MAX_NM or adjusting DC_LC_SELECTOR."
        )
    else:
        log.info("All 28 couplers: optimal gap found within sweep range.")
    plt.rcParams.update({
        "figure.dpi":        150,
        "font.family":       "DejaVu Sans",
        "axes.titlesize":    9,
        "axes.labelsize":    8,
        "xtick.labelsize":   7,
        "ytick.labelsize":   7,
        "legend.fontsize":   7,
        "lines.linewidth":   1.6,
        "lines.markersize":  3,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.25,
        "grid.linestyle":    "--",
    })
    _C_IN  = "#0072B2"    # blue  — input coupler
    _C_OUT = "#D55E00"    # vermilion — output coupler
    fig1, axes1 = plt.subplots(2, 7, figsize=(22, 6), sharey=False)
    fig1.suptitle(
        r"Directional Coupler Supermode $\Delta n$ vs Gap — All 14 Rings",
        fontsize=11, fontweight="bold", y=1.02,
    )
    for _ri in range(_DC_N_RINGS):
        _ax  = axes1[_ri // 7][_ri % 7]
        _inj = _dc_jobs[_ri * 2]
        _ouj = _dc_jobs[_ri * 2 + 1]

        _vi = ~np.isnan(_inj["dn_arr"])
        _vo = ~np.isnan(_ouj["dn_arr"])

        _ax.semilogy(_dc_gaps_nm[_vi], _inj["dn_arr"][_vi],
                     color=_C_IN,  lw=1.6, label="input")
        _ax.semilogy(_dc_gaps_nm[_vo], _ouj["dn_arr"][_vo],
                     color=_C_OUT, lw=1.6, ls="--", label="output")

        # Horizontal target lines
        _ax.axhline(_inj["target_dn"], color=_C_IN,  ls=":", lw=1.0, alpha=0.7)
        _ax.axhline(_ouj["target_dn"], color=_C_OUT, ls=":", lw=1.0, alpha=0.7)

        # Vertical optimal gap lines
        for _j, _c in [(_inj, _C_IN), (_ouj, _C_OUT)]:
            if not np.isnan(_j["match"]["gap_nm"]):
                _ax.axvline(_j["match"]["gap_nm"], color=_c, ls="-.", lw=0.8, alpha=0.75)

        _plt_s = "Aq" if _ri == 0 else "SiO₂"
        _ax.set_title(
            f"{_inj['label']}\n{_plt_s}  {_inj['lam_nm']:.1f} nm",
            fontsize=8, pad=3,
        )
        _ax.set_xlabel("Gap (nm)", fontsize=6.5)
        if _ri % 7 == 0:
            _ax.set_ylabel(r"$\Delta n$", fontsize=7.5)
        if _ri == 0:
            _ax.legend(fontsize=6)
        _ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    plt.tight_layout()
    _f1 = DATA_DIR / f"{VERSION_NAME}_dc_delta_n_vs_gap"
    for _ext in (".png", ".pdf"):
        plt.savefig(str(_f1) + _ext, dpi=150, bbox_inches="tight")
        print(f"  Saved → {str(_f1) + _ext}")
    plt.show()
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle("Optimal Coupler Gaps & Cross-coupling Coefficients — Critical Coupling",
                  fontsize=10, fontweight="bold")
    _ri_arr = np.arange(_DC_N_RINGS)
    _gap_in  = np.array([_dc_jobs[i*2  ]["match"]["gap_nm"]     for i in range(_DC_N_RINGS)])
    _gap_out = np.array([_dc_jobs[i*2+1]["match"]["gap_nm"]     for i in range(_DC_N_RINGS)])
    _k1_ach  = np.array([_dc_jobs[i*2  ]["match"]["k_achieved"] for i in range(_DC_N_RINGS)])
    _k2_ach  = np.array([_dc_jobs[i*2+1]["match"]["k_achieved"] for i in range(_DC_N_RINGS)])
    _k1_tgt  = np.array([float(cc_k1[i]) for i in range(_DC_N_RINGS)])
    _k2_tgt  = np.array([float(cc_k2[i]) for i in range(_DC_N_RINGS)])
    ax2a.plot(_ri_arr, _gap_in,  "o-",  color=_C_IN,  ms=6, lw=2, label="Input  gap  (k₁)")
    ax2a.plot(_ri_arr, _gap_out, "s--", color=_C_OUT, ms=6, lw=2, label="Output gap (k₂)")
    ax2a.set_xticks(_ri_arr)
    ax2a.set_xticklabels([cc_labels[i] for i in range(_DC_N_RINGS)],
                         rotation=45, ha="right", fontsize=7)
    ax2a.set_ylabel("Optimal gap  (nm)")
    ax2a.set_title("Optimal Gap per Ring  (input ≠ output → asymmetric coupling)")
    ax2a.legend()
    ax2a.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax2b.plot(_ri_arr, _k1_ach, "o-",  color=_C_IN,  ms=6, lw=2, label=r"$k_1$ achieved")
    ax2b.plot(_ri_arr, _k2_ach, "s--", color=_C_OUT, ms=6, lw=2, label=r"$k_2$ achieved")
    ax2b.plot(_ri_arr, _k1_tgt, "o:",  color=_C_IN,  ms=4, lw=1.2, alpha=0.45, label=r"$k_1$ target")
    ax2b.plot(_ri_arr, _k2_tgt, "s:",  color=_C_OUT, ms=4, lw=1.2, alpha=0.45, label=r"$k_2$ target")
    ax2b.set_xticks(_ri_arr)
    ax2b.set_xticklabels([cc_labels[i] for i in range(_DC_N_RINGS)],
                         rotation=45, ha="right", fontsize=7)
    ax2b.set_ylabel("Cross-coupling coefficient  k")
    ax2b.set_title(r"Achieved vs Target $k_1$, $k_2$")
    ax2b.legend()
    ax2b.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    plt.tight_layout()
    _f2 = DATA_DIR / f"{VERSION_NAME}_dc_optimal_gaps"
    for _ext in (".png", ".pdf"):
        plt.savefig(str(_f2) + _ext, dpi=150, bbox_inches="tight")
        print(f"  Saved → {str(_f2) + _ext}")
    plt.show()
    dc_gap_input_nm   = np.array([_dc_jobs[i*2  ]["match"]["gap_nm"]
                                   for i in range(_DC_N_RINGS)])
    dc_gap_output_nm  = np.array([_dc_jobs[i*2+1]["match"]["gap_nm"]
                                   for i in range(_DC_N_RINGS)])
    dc_dn_input_sim   = np.array([_dc_jobs[i*2  ]["match"]["delta_n_at_gap"]
                                   for i in range(_DC_N_RINGS)])
    dc_dn_output_sim  = np.array([_dc_jobs[i*2+1]["match"]["delta_n_at_gap"]
                                   for i in range(_DC_N_RINGS)])
    dc_k1_achieved    = np.array([_dc_jobs[i*2  ]["match"]["k_achieved"]
                                   for i in range(_DC_N_RINGS)])
    dc_k2_achieved    = np.array([_dc_jobs[i*2+1]["match"]["k_achieved"]
                                   for i in range(_DC_N_RINGS)])
    dc_err_input_pct  = np.array([_dc_jobs[i*2  ]["match"]["error_pct"]
                                   for i in range(_DC_N_RINGS)])
    dc_err_output_pct = np.array([_dc_jobs[i*2+1]["match"]["error_pct"]
                                   for i in range(_DC_N_RINGS)])
    print("\n  Exported arrays (shape = (14,), index 0 = sensor ring):")
    for _vn in [
        "dc_gap_input_nm",  "dc_gap_output_nm",
        "dc_dn_input_sim",  "dc_dn_output_sim",
        "dc_k1_achieved",   "dc_k2_achieved",
        "dc_err_input_pct", "dc_err_output_pct",
    ]:
        print(f"    {_vn}")
    print()
    print("  Next step → CELL 13: FDTD/varFDTD validation using these gap values.")

    state.update({k: globals().get(k) for k in [
        'DC_GAP_MAX_NM', 'DC_GAP_MIN_NM', 'DC_LC_SELECTOR', 'DC_MESH_Y', 'DC_MESH_Z', 'DC_N_GAPS',
        'DC_N_MODES', 'DC_Y_SPAN_UM', '_C_IN', '_C_OUT', '_DC_CORE_T_UM', '_DC_HALF_T_UM',
        '_DC_MARGIN_UM', '_DC_MESH_STEP_UM', '_DC_N_RINGS', '_DC_N_SPEC', '_DC_SIM_Y_SPAN_ORIG_UM', '_DC_SIO2_Z_CTR',
        '_DC_SIO2_Z_SPAN', '_DC_WG_W_M', '_DC_WG_W_NM', '_DC_WG_W_UM', '_DC_Z_ABOVE_UM', '_DC_Z_BELOW_UM',
        '_DC_Z_CTR', '_DC_Z_SPAN_UM', '_HDR', '_Lc_a', '_Lc_b', '_Lc_sel',
        '_S', '_S2', '_any_uncached', '_ax', '_c', '_comp_arr',
        '_ctype', '_dc_gaps_m', '_dc_gaps_nm', '_dc_jobs', '_dc_mode', '_dn',
        '_dn_a', '_dn_arr', '_dn_b', '_dn_sel', '_el', '_eta',
        '_ext', '_f1', '_f2', '_gap_in', '_gap_m', '_gap_out',
        '_gi', '_grp', '_hdf5_path', '_hdr', '_hf', '_hf_init',
        '_hf_r', '_hpath', '_in_j', '_in_jobs', '_inj', '_is_sensor',
        '_j', '_ji', '_job', '_job_n_done', '_k1_ach', '_k1_tgt',
        '_k2_ach', '_k2_tgt', '_k_tgt', '_label', '_lam_m', '_lam_nm',
        '_m', '_n_done', '_n_fully_cached', '_n_oor', '_n_up', '_ne',
        '_ne_arr', '_no', '_no_arr', '_ok', '_ouj', '_out_j',
        '_out_jobs', '_plt_lbl', '_plt_s', '_rate', '_remain', '_rg',
        '_ri', '_ri_arr', '_runs', '_t0', '_tee', '_tee_arr',
        '_teo', '_teo_arr', '_vi', '_vn', '_vo', 'ax2a',
        'ax2b', 'axes1', 'dc_dn_input_sim', 'dc_dn_output_sim', 'dc_err_input_pct', 'dc_err_output_pct',
        'dc_gap_input_nm', 'dc_gap_output_nm', 'dc_k1_achieved', 'dc_k2_achieved', 'fig1', 'fig2',
    ] if k in globals()})
    return state


if __name__ == "__main__":
    run()
