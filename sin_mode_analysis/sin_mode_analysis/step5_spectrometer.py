"""
step5_spectrometer.py — 13 spectrometer rings (radius sweep + phase match).

Extends the sensor-ring analysis to the 13 SiO2-clad spectrometer rings,
each tuned to one of 13 staggered resonances across the 10 nm FSR.
Now also captures the radiative bend loss (Im(neff) -> alpha_bend) per
ring at its own wavelength, mirroring the sensor-ring step. The target
FSR and waveguide width are inherited from step3 (sibling import).
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
from .step3_ring_radius import RR_WG_WIDTH_NM, RR_FSR_NM

from .config import *  # shared platform constants & paths
from .lumerical_session import import_lumapi
from .plotting import apply_style, save_fig, make_colorbar
from . import plotting as _plot

# ── Data-contract inputs (injected by main.py via `state`) ──────────
_exc = None
neff_real_sio2 = None
te_frac_sio2 = None

apply_style()

# ─────────────────────────────────────────────────────────
#  Module constants / design knobs for this step
# ─────────────────────────────────────────────────────────
N_SPEC_RINGS        = 13
SPEC_LAM0_NM        = 1550.0
SPEC_DELTA_LAM_NM   = 10.0 / 13.0          # ≈ 0.769231 nm  (exact rational step)
SPEC_LAM_NM         = SPEC_LAM0_NM + np.arange(N_SPEC_RINGS) * SPEC_DELTA_LAM_NM

def _sp_per_m_to_dbcm(a_per_m):  return a_per_m * _SP_DBCM_PER_INVM


def _sp_dbcm_to_per_m(a_dbcm):   return a_dbcm / _SP_DBCM_PER_INVM


def _sp_set_near_n(mode, n_target) -> None:
    """
    Seed the FDE eigensolver to search near a target effective index.
    Version-tolerant: set whichever of these properties this build exposes.
    This is the key robustness fix for leaky bent modes (mode finding).
    Identical to Cell 4 _rr_set_near_n.
    """
    for _p, _v in (("use max index", 0),
                   ("search", "near n"),
                   ("n", float(n_target))):
        try:
            mode.set(_p, _v)
        except Exception:
            pass


def _sp_build_fde(mode, radius_m: float, wavelength_m: float,
                  for_loss: bool = False, neff_guess=None) -> None:
    """
    Build bent-waveguide FDE cross-section for one (R, λ) point.
    Geometry is IDENTICAL to Cell 4 _rr_build_fde except:
      - upper cladding / background index = SPEC_N_UPPER  (SiO₂, 1.4469)
      - bend orientation = 0  (horizontal ring in XY plane — corrected)
    Stack (bottom to top along Z):
      SiO₂ substrate  →  SiN core 400 nm  →  SiO₂ upper cladding
    Cross-section:  Y = width axis (1000 nm),  Z = height axis (400 nm)

    for_loss=False : original LIGHT geometry, default search (n_g stencil).
    for_loss=True  : WIDE lateral domain + PML, more trial modes, and (when a
                     neff_guess is given) a "near n" search so the leaky
                     fundamental is reliably found.
    """
    m = mode
    m.switchtolayout()
    m.selectall()
    m.delete()

    _yspan_um = _sp_y_span_loss_um if for_loss else _sp_y_span_um
    _mesh_y   = _sp_mesh_y_loss    if for_loss else MESH_CELLS_Y
    _n_trial  = (SPEC_LOSS_TRIAL_MODES if for_loss else N_MODES_REQUEST)

    # ── FDE solver ────────────────────────────────────────────────────────────
    m.addfde()
    m.set("solver type",           "2D X normal")   # propagation along +X
    m.set("x",                     0.0)
    m.set("y",                     0.0)
    m.set("z",                     _sp_z_ctr      * 1e-6)
    m.set("y span",                _yspan_um      * 1e-6)
    m.set("z span",                _sp_z_span_um  * 1e-6)
    m.set("wavelength",            wavelength_m)
    m.set("number of trial modes", _n_trial)
    m.set("mesh cells y",          _mesh_y)
    m.set("mesh cells z",          MESH_CELLS_Z)
    m.set("bent waveguide",        1)
    m.set("bend radius",           radius_m)
    m.set("bend orientation",      0)      # 0 = curvature centre along +Y → ring in XY plane

    # PML on all outer boundaries for the loss solve so the radiated tail is
    # absorbed (a metal boundary gives a trapped, real-neff mode => Im=0).
    if for_loss and SPEC_USE_PML_FOR_LOSS:
        for _bc in ("y min bc", "y max bc", "z min bc", "z max bc"):
            m.set(_bc, "PML")
        m.set("pml layers", SPEC_PML_LAYERS)

    # seed the eigensolver near the known real n_eff (robust mode finding)
    if for_loss and (neff_guess is not None):
        _sp_set_near_n(m, neff_guess)

    # ── Background (SiO₂ upper cladding — fills entire domain) ───────────────
    m.addrect()
    m.set("name",     "SP_bg")
    m.set("x",        0.0);  m.set("x span", 1.0e-6)
    m.set("y",        0.0);  m.set("y span", _yspan_um      * 1e-6)
    m.set("z",        _sp_z_ctr      * 1e-6)
    m.set("z span",   _sp_z_span_um  * 1e-6)
    m.set("material", "<Object defined dielectric>")
    m.set("index",    SPEC_N_UPPER)        # SiO₂ = 1.4469

    # ── Lower SiO₂ substrate slab (overrides background in -Z region) ────────
    m.addrect()
    m.set("name",     "SP_lower_clad")
    m.set("x",        0.0);  m.set("x span", 1.0e-6)
    m.set("y",        0.0);  m.set("y span", _yspan_um      * 1e-6)
    m.set("z",        _sp_sio2_z_ctr * 1e-6)
    m.set("z span",   _sp_sio2_z_span * 1e-6)
    m.set("material", "<Object defined dielectric>")
    m.set("index",    N_SIO2_FIXED)        # SiO₂ = 1.4469 (symmetric)

    # ── SiN core (highest priority, centred at z = 0) ─────────────────────────
    m.addrect()
    m.set("name",     "SP_core")
    m.set("x",        0.0);  m.set("x span", 1.0e-6)
    m.set("y",        0.0);  m.set("y span", _sp_wg_w_m)          # 1000 nm
    m.set("z",        0.0);  m.set("z span", _sp_core_t_um * 1e-6) # 400 nm
    m.set("material", "<Object defined dielectric>")
    m.set("index",    N_SIN_FIXED)         # SiN = 1.99


def _sp_collect_modes(mode, n_try: int):
    """
    Robustly collect found modes as a list of (k, neff_complex, te_fraction)
    by reading mode1..mode<n_try> and stopping at the first that is absent.
    Returns [] if findmodes produced nothing (so the caller can handle it).
    Identical to Cell 4 _rr_collect_modes.
    """
    out = []
    for k in range(1, n_try + 1):
        try:
            nc = complex(np.asarray(
                mode.getdata(f"FDE::data::mode{k}", "neff")).flat[0])
            te = float(np.asarray(
                mode.getdata(f"FDE::data::mode{k}", "TE polarization fraction")).flat[0])
        except Exception:
            break
        out.append((k, nc, te))
    return out


def _sp_solve_neff(mode, radius_m: float, wavelength_m: float,
                   for_loss: bool = False, neff_guess=None):
    """
    One complete FDE solve at (radius_m, wavelength_m).
    Returns (Re(neff), Im(neff), TE fraction, loss_dB_per_m).
    loss_dB_per_m is only read for the loss solve (else NaN).

    Light solve (for_loss=False): selects mode1 (fundamental) — unchanged, so
                                  neff / ng / FSR stay bit-for-bit identical.
    Loss solve  (for_loss=True) : selects the TE-like mode whose Re(neff) is
                                  closest to neff_guess (robust against leaky /
                                  spurious PML modes); raises if NO mode found.
    """
    _n_try = SPEC_LOSS_TRIAL_MODES if for_loss else N_MODES_REQUEST
    _sp_build_fde(mode, radius_m, wavelength_m,
                  for_loss=for_loss, neff_guess=neff_guess)
    mode.run()
    mode.findmodes()

    found = _sp_collect_modes(mode, _n_try)
    if not found:
        raise RuntimeError(
            f"findmodes found no modes (R={radius_m*1e6:.3f} um, "
            f"lam={wavelength_m*1e9:.1f} nm, for_loss={for_loss})")

    if for_loss and (neff_guess is not None):
        _te_like = [mm for mm in found if mm[2] >= 0.5] or found
        sel = min(_te_like, key=lambda mm: abs(mm[1].real - neff_guess))
    else:
        sel = found[0]                       # fundamental = mode1 (original)
    ksel, nc, te_v = sel

    loss_dbm = np.nan
    if for_loss:
        try:
            loss_dbm = float(np.asarray(
                mode.getdata(f"FDE::data::mode{ksel}", "loss")).flat[0])
        except Exception:
            loss_dbm = 4.0 * np.pi * abs(nc.imag) / wavelength_m * (10.0 / np.log(10))
    return nc.real, nc.imag, te_v, loss_dbm


def _sp_neff_ng(mode, radius_m: float, lam0_m: float, dlam_m: float):
    """
    Three-point central-difference group index at lam0 for a given radius.
    Identical to Cell 4 _rr_fsr_stencil, but lam0 and dlam are explicit
    arguments so the function works for any of the 13 ring wavelengths.
    Uses LIGHT solves only (for_loss=False), so n_g is unchanged.

    Returns (neff_at_lam0, ng, te_frac, neff_lo, neff_hi).

    Formula:  ng = neff(λ₀) − λ₀ · [neff(λ₀+Δλ) − neff(λ₀−Δλ)] / (2·Δλ)
    """
    lam_lo_m = lam0_m - dlam_m
    lam_hi_m = lam0_m + dlam_m

    neff_lo, _, _,    _ = _sp_solve_neff(mode, radius_m, lam_lo_m, for_loss=False)
    neff_0,  _, te_v, _ = _sp_solve_neff(mode, radius_m, lam0_m,   for_loss=False)
    neff_hi, _, _,    _ = _sp_solve_neff(mode, radius_m, lam_hi_m, for_loss=False)

    dneff_dlam    = (neff_hi - neff_lo) / (2.0 * dlam_m)
    ng            = neff_0 - lam0_m * dneff_dlam

    return neff_0, ng, te_v, neff_lo, neff_hi


def _sp_loss_solve(mode, radius_m: float, lam0_m: float, neff_guess: float):
    """One PML solve at the ring's λₙ (seeded near neff_guess)
    -> (Im(neff), loss_dB/m).  Mirrors Cell 4 _rr_loss_solve but the
    wavelength is explicit so each ring uses its own λₙ."""
    _, nimag, _, loss_dbm = _sp_solve_neff(mode, radius_m, lam0_m,
                                           for_loss=True, neff_guess=neff_guess)
    return nimag, loss_dbm


def _sp_make_group_path(ring_idx, lam_nm, r_min_um, r_max_um, n_radii):
    """Return the full HDF5 group path for one ring sweep."""
    key = (
        f"rr_{SPEC_FSR_NM:.0f}nm_{lam_nm:.6f}nm_"
        f"{SPEC_WG_WIDTH_NM:.0f}nm_"
        f"{r_min_um:.4f}-{r_max_um:.4f}um_{n_radii}pts"
    )
    return f"spectrometer_rings/ring_{ring_idx:02d}/{key}"


def _sp_init_group(hf, grp_path, ring_idx, lam_nm,
                   r_min_um, r_max_um, radii_um, dlam_nm):
    """
    Pre-allocate all datasets for one ring sweep inside an already-open
    h5py file.  Safe to call on an existing group (idempotent): only MISSING
    datasets/attrs are created, so an older FSR-only group gains the new
    bend-loss datasets without touching existing data. loss_computed is seeded
    from existing Im(neff) finiteness so prior bend-loss runs are not redone.
    Mirrors Cell 4 _rr_init_hdf5 exactly.
    """
    N  = len(radii_um)
    g  = hf.require_group(grp_path)
    mg = g.require_group("metadata")

    mg.attrs.setdefault("ring_index",       ring_idx)
    mg.attrs.setdefault("fsr_nm",           SPEC_FSR_NM)
    mg.attrs.setdefault("lam0_nm",          lam_nm)
    mg.attrs.setdefault("wg_width_nm",      SPEC_WG_WIDTH_NM)
    mg.attrs.setdefault("wg_height_nm",     _sp_core_t_um * 1e3)
    mg.attrs.setdefault("r_min_um",         r_min_um)
    mg.attrs.setdefault("r_max_um",         r_max_um)
    mg.attrs.setdefault("n_radii",          N)
    mg.attrs.setdefault("delta_lam_nm",     dlam_nm)
    mg.attrs.setdefault("n_SiN",            N_SIN_FIXED)
    mg.attrs.setdefault("n_SiO2",           N_SIO2_FIXED)
    mg.attrs.setdefault("n_upper_clad",     SPEC_N_UPPER)
    mg.attrs.setdefault("bend_orientation", 0)
    mg.attrs.setdefault("version_name",     VERSION_NAME_SIO2)
    mg.attrs.setdefault("timestamp_start",  datetime.now().isoformat())
    # NEW bend-loss metadata (always refreshed)
    mg.attrs["schema"]          = "v2_bendloss"
    mg.attrs["y_span_loss_um"]  = _sp_y_span_loss_um
    mg.attrs["pml_layers"]      = SPEC_PML_LAYERS
    mg.attrs["alpha_prop_dbcm"] = SPEC_ALPHA_PROP_DBCM

    if "radii_um" not in mg:
        mg.create_dataset("radii_um",      data=radii_um)
    if "lam_stencil_nm" not in mg:
        mg.create_dataset("lam_stencil_nm",
                          data=np.array([lam_nm - dlam_nm,
                                         lam_nm,
                                         lam_nm + dlam_nm]))

    _nan = np.full(N, np.nan, dtype=np.float64)
    rg   = g.require_group("results")
    for ds in ("neff", "ng", "te_frac", "ngL_um", "neff_lo", "neff_hi",
               "neff_imag", "alpha_bend_dbcm", "loss_dbm"):
        if ds not in rg:
            rg.create_dataset(ds, data=_nan.copy(), chunks=(N,))

    fg = g.require_group("flags")
    if "computed" not in fg:
        fg.create_dataset("computed",
                          data=np.zeros(N, dtype=bool), chunks=(N,))
    if "loss_computed" not in fg:
        # seed from any existing Im(neff) so prior bend-loss runs are not redone
        _seed = (np.isfinite(rg["neff_imag"][:]) if "neff_imag" in rg
                 else np.zeros(N, dtype=bool))
        fg.create_dataset("loss_computed", data=_seed, chunks=(N,))


def _sp_load_cache(hf, grp_path, N):
    """
    Read all result arrays for one ring from an open HDF5 file.
    Returns the FSR arrays (with ngL in metres), the NEW bend-loss arrays,
    and both boolean flags.

    NOTE: ngL is stored in µm in HDF5 → converted to metres on load,
    consistent with Cell 4 _rr_load_cache.
    """
    g  = hf[grp_path]
    rg = g["results"]
    fg = g["flags"]

    neff_a       = rg["neff"][:]
    ng_a         = rg["ng"][:]
    te_a         = rg["te_frac"][:]
    ngL_a        = rg["ngL_um"][:] * 1e-6     # µm → m
    neff_lo_a    = rg["neff_lo"][:]
    neff_hi_a    = rg["neff_hi"][:]
    neff_imag_a  = rg["neff_imag"][:]
    alpha_bend_a = rg["alpha_bend_dbcm"][:]
    loss_dbm_a   = rg["loss_dbm"][:]
    computed     = fg["computed"][:]
    loss_done    = fg["loss_computed"][:]

    return (neff_a, ng_a, te_a, ngL_a, neff_lo_a, neff_hi_a,
            neff_imag_a, alpha_bend_a, loss_dbm_a, computed, loss_done)


def _sp_phase_match(radii_um_v, neff_v, ng_v, lam0_m, ngL_v_m, fsr_m):
    """
    Given valid (non-NaN) arrays from one ring sweep, compute:
      1.  FSR-matched radius (closest ng·L to λ₀²/FSR)
      2.  Integer resonance order m
      3.  Phase-matched radius R_pm  (m·λ₀ = neff·2πR, solved iteratively)

    All input arrays must be the same length (already filtered to valid rows).
    ngL_v_m is in metres.  Returns a results dict.
    """
    R_v_m  = radii_um_v * 1e-6
    L_v_m  = 2.0 * np.pi * R_v_m

    # ── FSR matching ──────────────────────────────────────────────────────────
    target_ngL_m = lam0_m**2 / fsr_m
    dist         = np.abs(ngL_v_m - target_ngL_m)
    bi           = int(np.argmin(dist))

    # ── Integer resonance order at the FSR-matched radius ────────────────────
    m_cont = neff_v[bi] * L_v_m[bi] / lam0_m
    m_best = int(round(float(m_cont)))

    neff_b   = float(neff_v[bi])
    ng_b     = float(ng_v[bi])
    R_fsr_um = float(radii_um_v[bi])
    ngL_b_um = float(ngL_v_m[bi]) * 1e6   # µm for reporting

    FSR_fsr_nm    = (lam0_m**2 / (ng_b * ngL_v_m[bi])) * 1e9
    lam_res_b_nm  = (neff_b * L_v_m[bi] / m_best) * 1e9

    # ── Phase-matched radius (iterative refinement) ───────────────────────────
    # Build linear interpolant  neff_bent(R)  from sweep data
    _interp = interp1d(
        R_v_m, neff_v,
        kind="linear",
        bounds_error=False,
        fill_value=(neff_v[0], neff_v[-1]),
    )
    R_pm_m = m_best * lam0_m / (2.0 * np.pi * neff_b)   # first-order seed

    for _ in range(_PM_MAX_ITER):
        neff_at  = float(_interp(R_pm_m))
        R_pm_new = m_best * lam0_m / (2.0 * np.pi * neff_at)
        if abs(R_pm_new - R_pm_m) < _PM_TOL_M:
            R_pm_m = R_pm_new
            break
        R_pm_m = R_pm_new

    neff_pm    = float(_interp(R_pm_m))
    R_pm_um    = R_pm_m * 1e6
    L_pm_um    = 2.0 * np.pi * R_pm_um
    FSR_pm_nm  = (lam0_m**2 / (ng_b * L_pm_um * 1e-6)) * 1e9
    lam_res_pm = (neff_pm * L_pm_um * 1e-6 / m_best) * 1e9
    resid_pm   = abs(lam_res_pm - lam0_m * 1e9) * 1e3    # [pm]

    return {
        "m":              m_best,
        # FSR-matched
        "R_fsr_um":       R_fsr_um,
        "L_fsr_um":       2.0 * np.pi * R_fsr_um,
        "neff_fsr":       neff_b,
        "ng_fsr":         ng_b,
        "ngL_fsr_um":     ngL_b_um,
        "FSR_fsr_nm":     FSR_fsr_nm,
        "lam_res_fsr_nm": lam_res_b_nm,
        # Phase-matched
        "R_pm_um":        R_pm_um,
        "L_pm_um":        L_pm_um,
        "neff_pm":        neff_pm,
        "ng_pm":          ng_b,      # ng evaluated at FSR-matched R; R_pm ≈ R_fsr
        "FSR_pm_nm":      FSR_pm_nm,
        "lam_res_pm_nm":  lam_res_pm,
        "resid_pm_pm":    resid_pm,
        "delta_R_nm":     (R_pm_um - R_fsr_um) * 1e3,
    }

def run(state=None):
    """Execute this step. `state` carries the shared namespace
    between steps (the notebook's old kernel globals + bridge).
    Returns the updated `state`."""
    state = {} if state is None else state
    global N_SPEC_RINGS, SPEC_ALPHA_PROP_DBCM, SPEC_DELTA_LAM_NG_NM, SPEC_DELTA_LAM_NM, SPEC_FSR_NM, SPEC_FWHM_NM, SPEC_LAM0_NM, SPEC_LAM_NM, SPEC_LOSS_TRIAL_MODES, SPEC_MESH_DY_NM, SPEC_N_RADII, SPEC_N_UPPER, SPEC_PML_LAYERS, SPEC_R_HALF_SPAN_UM, SPEC_R_MIN_FLOOR_UM, SPEC_USE_PML_FOR_LOSS, SPEC_WG_WIDTH_NM, SPEC_Y_SPAN_LOSS_UM, _H1, _H2, _H3, _H4, _L_m, _N_n, _PM_MAX_ITER, _PM_TOL_M, _R_est_um, _R_m, _R_um, _SEP, _SEP2, _SP_DBCM_PER_INVM, _a_bend_match, _a_bend_per_m, _a_dbcm, _a_tot_dbcm, _ab_all, _ab_b, _ab_fin, _abend_n, _any_pos, _any_uncached, _av, _c, _cbar, _cmap, _cnorm, _colors, _comp_n, _delta, _dlam_ng_m, _dneff_dlam_str, _dng, _done_n, _el, _elapsed, _eta, _ext, _fig5_stem, _fig_stem, _fsr_all, _fsr_m, _fsr_was_cached, _glv, _gp, _grp_paths, _gv, _hdr, _i, _l, _lam0_n_m, _lam0_n_nm, _lam_b, _lam_m_arr, _lam_nm_arr, _lam_v, _loss_dbm, _loss_n, _lossdbm_n, _match_full_idx, _n, _n_done, _n_fsr_only, _neff_n, _neff_str, _neff_str_fit, _neff_v, _ngL_m, _ngL_m_i, _ngL_n, _ng_all, _ng_n, _ng_str, _ng_v, _nhi_n, _nhi_v, _nimag, _nimag_n, _nlo_n, _nlo_v, _nv, _ok, _pm, _poly_str, _r_all, _radii_um_n, _rate, _ratio, _remaining, _rfsr_v, _rg, _rmax, _rmin, _rpm_v, _runs, _rv, _sm, _sp_core_t_um, _sp_half_t_um, _sp_hf, _sp_mesh_y_loss, _sp_mode, _sp_sio2_z_ctr, _sp_sio2_z_span, _sp_wg_w_m, _sp_y_span_loss_um, _sp_y_span_um, _sp_z_above_um, _sp_z_below_um, _sp_z_ctr, _sp_z_span_um, _spec_r_max, _spec_r_min, _spec_radii_um, _spec_results, _spec_w_idx, _spec_w_nm, _spec_w_um, _src, _t0, _target_ngL_n_m, _target_ngL_n_um, _te_n, _te_str, _te_v, _tgt, _valid_n, _valid_str, _vn, _wl_str_m, ax00, ax01, ax10, ax11, axb0, axb1, axes, fig, fig5, spec_FSR_pm_nm, spec_L_pm_um, spec_Q_bend, spec_Q_i_total, spec_R_pm_um, spec_alpha_bend_dbcm, spec_alpha_total_dbcm, spec_lam_res_pm_nm, spec_m, spec_neff_imag, spec_neff_pm, spec_ng_pm
    globals()['lumapi'] = import_lumapi()
    globals().update(state)

    SPEC_WG_WIDTH_NM    = RR_WG_WIDTH_NM        # 1000 nm  — inherited
    SPEC_FSR_NM         = RR_FSR_NM             # 10.0 nm  — inherited
    SPEC_N_UPPER        = N_UPPER_CLADDING_SIO2 # 1.4469   — SiO₂ symmetric
    SPEC_FWHM_NM        = 0.5                   # [nm]  design FWHM (for the Q_loaded summary)
    SPEC_N_RADII        = 100
    SPEC_R_HALF_SPAN_UM = 19    # [µm]  search half-width around analytical estimate
    SPEC_R_MIN_FLOOR_UM = 20    # [µm]  absolute lower bound (bend loss guard)
    SPEC_DELTA_LAM_NG_NM = 5.0   # [nm]
    SPEC_USE_PML_FOR_LOSS = True    # PML boundaries for the loss solve (False = loss-blind)
    SPEC_Y_SPAN_LOSS_UM   = 16.0    # [µm]  widened lateral span so the PML is past the caustic
    SPEC_MESH_DY_NM       = 50.0    # [nm]  target y cell size over the widened span
    SPEC_PML_LAYERS       = 16      # PML layers (raise if alpha_bend is not converged)
    SPEC_LOSS_TRIAL_MODES = 12      # trial modes for the (leaky) loss solve — more = safer
    SPEC_ALPHA_PROP_DBCM  = 1.0     # [dB/cm] assumed propagation loss, for the total-Q summary
    _PM_TOL_M    = 1e-13    # |ΔR| < 0.1 pm
    _PM_MAX_ITER = 20
    _sp_core_t_um    = CORE_THICKNESS_UM
    _sp_half_t_um    = _sp_core_t_um / 2.0
    _sp_z_below_um   = SIM_Z_BELOW_UM
    _sp_z_above_um   = SIM_Z_ABOVE_UM
    _sp_z_span_um    = _sp_z_below_um + _sp_core_t_um + _sp_z_above_um
    _sp_y_span_um    = SIM_Y_SPAN_UM
    _sp_sio2_z_ctr   = -(_sp_half_t_um + _sp_z_below_um / 2.0)  # below core centre
    _sp_sio2_z_span  = _sp_z_below_um
    _sp_z_ctr        = (_sp_z_above_um - _sp_z_below_um) / 2.0
    _sp_wg_w_m       = SPEC_WG_WIDTH_NM * 1e-9
    _sp_y_span_loss_um = SPEC_Y_SPAN_LOSS_UM if SPEC_USE_PML_FOR_LOSS else _sp_y_span_um
    _sp_mesh_y_loss    = (int(round(_sp_y_span_loss_um * 1e3 / SPEC_MESH_DY_NM))
                          if SPEC_USE_PML_FOR_LOSS else MESH_CELLS_Y)
    _SP_DBCM_PER_INVM = (10.0 / np.log(10)) / 100.0          # 1/m -> dB/cm
    print("=" * 70)
    print("  SPECTROMETER RING ARRAY — SiO₂/SiN/SiO₂  (symmetric cladding)")
    print("=" * 70)
    print(f"  Rings          : {N_SPEC_RINGS}   (n = 0 … {N_SPEC_RINGS-1})")
    print(f"  λ₀  (ring 0)   : {SPEC_LAM_NM[0]:.6f} nm")
    print(f"  λ₁₂ (ring 12)  : {SPEC_LAM_NM[-1]:.6f} nm")
    print(f"  Step           : {SPEC_DELTA_LAM_NM:.6f} nm  (= 10/13 nm)")
    print(f"  Waveguide      : {SPEC_WG_WIDTH_NM:.0f} nm × {_sp_core_t_um*1e3:.0f} nm  (w × h)")
    print(f"  n_core         : {N_SIN_FIXED}    n_clad = {SPEC_N_UPPER} (SiO₂)")
    print(f"  Target FSR     : {SPEC_FSR_NM:.1f} nm")
    print(f"  Radii/ring     : {SPEC_N_RADII}")
    print(f"  ng stencil     : ±{SPEC_DELTA_LAM_NG_NM:.1f} nm  (3 light solves for n_g + "
          f"1 PML solve for loss, per radius)")
    print(f"  Bend-loss solve: PML={SPEC_USE_PML_FOR_LOSS}, "
          f"y span {_sp_y_span_loss_um:.1f} µm, mesh_y {_sp_mesh_y_loss}, "
          f"{SPEC_PML_LAYERS} PML layers, near-n seed, {SPEC_LOSS_TRIAL_MODES} trial modes")
    print(f"  HDF5 file      : {HDF5_PATH_SIO2}")
    print("=" * 70)
    print()
    _spec_w_um  = SPEC_WG_WIDTH_NM * 1e-3
    _spec_w_idx = int(np.argmin(np.abs(SWEEP_WIDTHS_UM - _spec_w_um)))
    _spec_w_nm  = SWEEP_WIDTHS_UM[_spec_w_idx] * 1e3
    log.info(f"Straight ng extraction  w={SPEC_WG_WIDTH_NM:.0f} nm → index {_spec_w_idx} "
             f"({_spec_w_nm:.2f} nm)")
    _neff_str = neff_real_sio2[_spec_w_idx, :, 0].copy()   # shape (13,)
    _te_str   = te_frac_sio2  [_spec_w_idx, :, 0].copy()
    if float(np.nanmean(_te_str)) < 0.5:                    # safety: mode 0 must be TE
        log.warning("Mode 0 SiO₂ sweep is TM — using mode 1 for ng_straight.")
        _neff_str = neff_real_sio2[_spec_w_idx, :, 1].copy()
    _wl_str_m  = SWEEP_WL_NM * 1e-9                        # shape (13,)  [m]
    _valid_str = ~np.isnan(_neff_str)
    _poly_str  = np.polyfit(_wl_str_m[_valid_str],
                            _neff_str[_valid_str], deg=1)    # linear fit
    _dneff_dlam_str = _poly_str[0]                           # [1/m]
    _neff_str_fit   = np.polyval(_poly_str, _wl_str_m)      # shape (13,)
    _ng_str         = _neff_str_fit - _wl_str_m * _dneff_dlam_str  # shape (13,)
    log.info(f"ng_straight_sio2 range: {_ng_str.min():.6f} – {_ng_str.max():.6f}")
    _fsr_m      = SPEC_FSR_NM * 1e-9
    _lam_nm_arr = SPEC_LAM_NM                               # shape (13,)  [nm]
    _lam_m_arr  = _lam_nm_arr * 1e-9                        # shape (13,)  [m]
    _dlam_ng_m  = SPEC_DELTA_LAM_NG_NM * 1e-9               # stencil half-span [m]
    _R_est_um   = _lam_m_arr**2 / (_fsr_m * 2.0 * np.pi * _ng_str)
    _spec_radii_um = []
    _spec_r_min    = []
    _spec_r_max    = []
    for _n in range(N_SPEC_RINGS):
        _rmin = max(SPEC_R_MIN_FLOOR_UM, _R_est_um[_n] - SPEC_R_HALF_SPAN_UM)
        _rmax = _R_est_um[_n] + SPEC_R_HALF_SPAN_UM
        _spec_radii_um.append(np.linspace(_rmin, _rmax, SPEC_N_RADII))
        _spec_r_min.append(_rmin)
        _spec_r_max.append(_rmax)
    print(f"\n  Estimated radius range:  "
          f"{_R_est_um.min():.4f} – {_R_est_um.max():.4f} µm  "
          f"(straight ng estimate)\n")
    _spec_results = {}   # ring_idx → dict from _sp_phase_match + raw arrays + bend loss
    _grp_paths = []
    for _n in range(N_SPEC_RINGS):
        _gp = _sp_make_group_path(
            _n, float(_lam_nm_arr[_n]),
            _spec_r_min[_n], _spec_r_max[_n], SPEC_N_RADII,
        )
        _grp_paths.append(_gp)
    _sp_hf = h5py.File(HDF5_PATH_SIO2, "a")
    _any_uncached = False
    for _n in range(N_SPEC_RINGS):
        _gp = _grp_paths[_n]
        _sp_init_group(
            _sp_hf, _gp, _n, float(_lam_nm_arr[_n]),
            _spec_r_min[_n], _spec_r_max[_n],
            _spec_radii_um[_n], SPEC_DELTA_LAM_NG_NM,
        )
        _sp_hf.flush()
        _c = _sp_hf[_gp]["flags"]["computed"][:]
        _l = _sp_hf[_gp]["flags"]["loss_computed"][:]
        if not (_c & _l).all():
            _any_uncached = True
    _sp_mode = None
    if _any_uncached:
        log.info("Launching MODE session for spectrometer ring sweeps …")
        _sp_mode = lumapi.MODE(hide=False)
    for _n in range(N_SPEC_RINGS):

        _lam0_n_nm  = float(_lam_nm_arr[_n])
        _lam0_n_m   = _lam0_n_nm * 1e-9
        _gp         = _grp_paths[_n]
        _radii_um_n = _spec_radii_um[_n]
        _N_n        = SPEC_N_RADII

        # Target ng·L for this ring (changes with λₙ — this is why radii differ)
        _target_ngL_n_m  = _lam0_n_m**2 / _fsr_m
        _target_ngL_n_um = _target_ngL_n_m * 1e6

        # ── Load cache into fresh in-memory arrays ────────────────────────────────
        # These arrays mirror Cell 4 _neff_arr, _ng_arr, _alpha_bend_arr etc. exactly.
        (_neff_n, _ng_n, _te_n, _ngL_n, _nlo_n, _nhi_n,
         _nimag_n, _abend_n, _lossdbm_n, _comp_n, _loss_n) = \
            _sp_load_cache(_sp_hf, _gp, _N_n)
        # _ngL_n is in METRES after _sp_load_cache

        _done_n     = _comp_n & _loss_n
        _n_done     = int(_done_n.sum())
        _n_fsr_only = int((_comp_n & ~_loss_n).sum())
        _remaining  = _N_n - _n_done

        log.info(
            f"Ring {_n:02d} │ λ={_lam0_n_nm:.6f} nm │ "
            f"R=[{_radii_um_n[0]:.3f}, {_radii_um_n[-1]:.3f}] µm │ "
            f"target ng·L={_target_ngL_n_um:.4f} µm │ "
            f"done {_n_done}/{_N_n}"
            + (f"  (+{_n_fsr_only} FSR-only → loss solve only)" if _n_fsr_only else "")
        )

        # ── Print table header for this ring ─────────────────────────────────────
        _hdr = (f"  {'R (µm)':>10}  {'neff':>10}  {'ng':>10}  "
                f"{'TE':>6}  {'ng·L (µm)':>12}  {'a_bend(dB/cm)':>14}  {'Δ/tgt':>9}  src")
        print(f"\n  Ring {_n:02d} │ λ₀ = {_lam0_n_nm:.4f} nm  "
              f"│ target ng·L = {_target_ngL_n_um:.4f} µm")
        print(_hdr)
        print("  " + "─" * (len(_hdr) - 2))

        # Print fully-cached rows immediately so the table is contiguous
        for _i in range(_N_n):
            if _done_n[_i]:
                _ngL_m_i = _ngL_n[_i]          # already in metres
                _delta   = (_ngL_m_i - _target_ngL_n_m) / _target_ngL_n_m * 100.0
                print(
                    f"  {_radii_um_n[_i]:>10.4f}  {_neff_n[_i]:>10.6f}  "
                    f"{_ng_n[_i]:>10.6f}  {_te_n[_i]:>6.4f}  "
                    f"{_ngL_m_i*1e6:>12.4f}  {_abend_n[_i]:>14.4f}  "
                    f"{_delta:>+8.3f}%  cache\n"
                )

        # ── FDE sweep for missing rows ────────────────────────────────────────────
        if _remaining > 0 and _sp_mode is not None:
            _t0       = time.time()
            _runs     = 0

            for _i, _R_um in enumerate(_radii_um_n):
                if _comp_n[_i] and _loss_n[_i]:
                    continue   # fully done — skip

                _fsr_was_cached = bool(_comp_n[_i])
                _R_m = _R_um * 1e-6

                # ── (a) FSR stencil (n_g) — only if not already cached ───────────
                if not _comp_n[_i]:
                    try:
                        _neff_v, _ng_v, _te_v, _nlo_v, _nhi_v = \
                            _sp_neff_ng(_sp_mode, _R_m, _lam0_n_m, _dlam_ng_m)
                    except Exception as _exc:
                        log.warning(f"  Ring {_n:02d} │ R={_R_um:.4f} µm FSR solve FAILED: {_exc}")
                        # give up this radius entirely (avoid retrying a broken setup)
                        _comp_n[_i] = True
                        _loss_n[_i] = True
                        _sp_hf[f"{_gp}/flags/computed"][_i]      = True
                        _sp_hf[f"{_gp}/flags/loss_computed"][_i] = True
                        _sp_hf.flush()
                        continue

                    _L_m   = 2.0 * np.pi * _R_m
                    _ngL_m = _ng_v * _L_m          # [m]

                    # Memory arrays (ngL in metres — same as Cell 4)
                    _neff_n[_i] = _neff_v
                    _ng_n  [_i] = _ng_v
                    _te_n  [_i] = _te_v
                    _ngL_n [_i] = _ngL_m           # metres
                    _nlo_n [_i] = _nlo_v
                    _nhi_n [_i] = _nhi_v
                    _comp_n[_i] = True

                    # HDF5 — write immediately (fault-safe), ngL stored in µm
                    _rg = _sp_hf[f"{_gp}/results"]
                    _rg["neff"]   [_i] = _neff_v
                    _rg["ng"]     [_i] = _ng_v
                    _rg["te_frac"][_i] = _te_v
                    _rg["ngL_um"] [_i] = _ngL_m * 1e6   # µm in HDF5
                    _rg["neff_lo"][_i] = _nlo_v
                    _rg["neff_hi"][_i] = _nhi_v
                    _sp_hf[f"{_gp}/flags/computed"][_i] = True
                    _sp_hf.flush()
                else:
                    _neff_v = _neff_n[_i]      # reuse cached FSR result as the seed

                # ── (b) radiative bend loss — PML solve seeded by Re(neff) ───────
                try:
                    _nimag, _loss_dbm = _sp_loss_solve(
                        _sp_mode, _R_m, _lam0_n_m, neff_guess=_neff_v)
                except Exception as _exc:
                    log.warning(f"  Ring {_n:02d} │ R={_R_um:.4f} µm bend-loss solve "
                                f"FAILED: {_exc}; alpha_bend set to NaN (FSR data kept)")
                    _nimag, _loss_dbm = np.nan, np.nan

                if np.isfinite(_nimag):
                    _a_dbcm = _sp_per_m_to_dbcm(4.0 * np.pi * abs(_nimag) / _lam0_n_m)
                else:
                    _a_dbcm = np.nan

                _nimag_n  [_i] = _nimag
                _abend_n  [_i] = _a_dbcm
                _lossdbm_n[_i] = _loss_dbm
                _loss_n   [_i] = True

                _rg = _sp_hf[f"{_gp}/results"]
                _rg["neff_imag"]      [_i] = _nimag
                _rg["alpha_bend_dbcm"][_i] = _a_dbcm
                _rg["loss_dbm"]       [_i] = _loss_dbm
                _sp_hf[f"{_gp}/flags/loss_computed"][_i] = True
                _sp_hf.flush()

                _runs  += 1
                _ngL_m  = _ngL_n[_i]
                _delta  = (_ngL_m - _target_ngL_n_m) / _target_ngL_n_m * 100.0
                _src    = "loss" if _fsr_was_cached else "FDE"
                print(
                    f"  {_R_um:>10.4f}  {_neff_n[_i]:>10.6f}  {_ng_n[_i]:>10.6f}  "
                    f"{_te_n[_i]:>6.4f}  {_ngL_m*1e6:>12.4f}  {_a_dbcm:>14.4f}  "
                    f"{_delta:>+8.3f}%  {_src:>4}\n"
                )

                if _runs % 10 == 0 or _runs == _remaining:
                    _el   = time.time() - _t0
                    _rate = _runs / _el if _el > 0 else 1e-9
                    _eta  = (_remaining - _runs) / _rate
                    log.info(
                        f"  Ring {_n:02d} [{_runs:3d}/{_remaining}]  "
                        f"R={_R_um:.3f} µm  ng={_ng_n[_i]:.6f}  "
                        f"a_bend={_a_dbcm:.4f} dB/cm  Δ={_delta:+.3f}%  "
                        f"ETA {_eta:.0f} s"
                    )

            # Close-out metadata for this ring
            _sp_hf[_gp]["metadata"].attrs["timestamp_end"]  = \
                datetime.now().isoformat()
            _sp_hf[_gp]["metadata"].attrs["runs_completed"] = \
                int((_comp_n & _loss_n).sum())
            _sp_hf.flush()
            _elapsed = time.time() - _t0
            log.info(
                f"  Ring {_n:02d} done  "
                f"({_runs} radii processed in {_elapsed:.1f} s)"
            )

        # ── Phase matching — always from in-memory FSR arrays (unchanged) ────────
        _valid_n = ~np.isnan(_ng_n)
        if not np.any(_valid_n):
            log.error(f"Ring {_n:02d}: ALL rows failed — check warnings above.")
            continue

        _pm = _sp_phase_match(
            radii_um_v = _radii_um_n[_valid_n],
            neff_v     = _neff_n    [_valid_n],
            ng_v       = _ng_n      [_valid_n],
            lam0_m     = _lam0_n_m,
            ngL_v_m    = _ngL_n     [_valid_n],   # already metres
            fsr_m      = _fsr_m,
        )

        # Attach auxiliary information for summary tables and plots
        _pm["lam_nm"]        = _lam0_n_nm
        _pm["ng_straight"]   = float(_ng_str[_n])
        _pm["neff_straight"] = float(_neff_str_fit[_n])
        _pm["radii_um"]      = _radii_um_n
        _pm["neff_arr"]      = _neff_n.copy()
        _pm["ng_arr"]        = _ng_n.copy()
        _pm["ngL_arr_m"]     = _ngL_n.copy()     # metres
        _pm["valid_mask"]    = _valid_n
        # NEW — bend-loss arrays
        _pm["neff_imag_arr"]  = _nimag_n.copy()
        _pm["alpha_bend_arr"] = _abend_n.copy()  # dB/cm
        _pm["loss_dbm_arr"]   = _lossdbm_n.copy()

        # ── bend loss at the matched (FSR) radius — mirrors Cell 4 best-match ─────
        _match_full_idx = int(np.argmin(np.abs(_radii_um_n - _pm["R_fsr_um"])))
        _a_bend_match   = float(_abend_n[_match_full_idx])
        _pm["alpha_bend_match_dbcm"] = _a_bend_match
        _pm["neff_imag_match"]       = float(_nimag_n[_match_full_idx])
        _pm["loss_dbm_match"]        = float(_lossdbm_n[_match_full_idx])

        if np.isfinite(_a_bend_match):
            _a_bend_per_m = _sp_dbcm_to_per_m(_a_bend_match)
            _pm["Q_bend"] = (2.0 * np.pi * _pm["ng_pm"] / (_a_bend_per_m * _lam0_n_m)
                             if _a_bend_per_m > 0 else float("inf"))
            _a_tot_dbcm   = SPEC_ALPHA_PROP_DBCM + _a_bend_match
            _pm["alpha_total_dbcm"] = _a_tot_dbcm
            _pm["Q_i_total"] = (2.0 * np.pi * _pm["ng_pm"]
                                / (_sp_dbcm_to_per_m(_a_tot_dbcm) * _lam0_n_m))
        else:
            _pm["Q_bend"]           = np.nan
            _pm["alpha_total_dbcm"] = np.nan
            _pm["Q_i_total"]        = np.nan
        _pm["Q_loaded"] = _lam0_n_nm / SPEC_FWHM_NM

        _spec_results[_n] = _pm

        # Quick per-ring result line
        log.info(
            f"  Ring {_n:02d} │ RESULT │ "
            f"R_pm={_pm['R_pm_um']:.6f} µm  "
            f"neff={_pm['neff_pm']:.6f}  "
            f"ng={_pm['ng_pm']:.6f}  "
            f"L={_pm['L_pm_um']:.6f} µm  "
            f"FSR={_pm['FSR_pm_nm']:.6f} nm  "
            f"|Δλ|={_pm['resid_pm_pm']:.5f} pm  "
            f"a_bend={_pm['alpha_bend_match_dbcm']:.4f} dB/cm"
        )
    if _sp_mode is not None:
        _sp_mode.close()
        log.info("MODE session closed.")
    _sp_hf.close()
    log.info(f"HDF5 closed → {HDF5_PATH_SIO2}")
    _SEP  = "─" * 134
    _SEP2 = "═" * 134
    print("\n\n")
    print(_SEP2)
    print("  SPECTROMETER RING ARRAY — MASTER DESIGN SUMMARY")
    print(f"  SiO₂/SiN/SiO₂  │  w={SPEC_WG_WIDTH_NM:.0f} nm  "
          f"h={_sp_core_t_um*1e3:.0f} nm  │  FSR={SPEC_FSR_NM:.1f} nm  │  "
          f"n_SiN={N_SIN_FIXED}  n_SiO₂={N_SIO2_FIXED}  │  bend_orientation=0")
    print(_SEP2)
    print()
    print("  TABLE 1 — Straight waveguide vs Bent FDE dispersion  "
          "(@ FSR-matched radius)")
    print(_SEP)
    _H1 = (f"  {'n':>3}  {'λ_n (nm)':>13}  "
           f"{'neff_str':>10}  {'ng_str':>10}  "
           f"{'neff_bent':>11}  {'ng_bent':>11}  "
           f"{'Δng':>10}  {'TE_frac':>8}")
    print(_H1)
    print("  " + "─" * 130)
    for _n in range(N_SPEC_RINGS):
        if _n not in _spec_results:
            print(f"  {_n:>3}  FAILED"); continue
        _pm  = _spec_results[_n]
        _dng = _pm["ng_fsr"] - _pm["ng_straight"]
        # TE fraction is not carried in _spec_results; printed as "—" (kept from
        # the original Cell 10). The per-radius TE fraction lives in the HDF5
        # results/te_frac dataset if you need it.
        print(
            f"  {_n:>3}  {_pm['lam_nm']:>13.6f}  "
            f"{_pm['neff_straight']:>10.6f}  {_pm['ng_straight']:>10.6f}  "
            f"{_pm['neff_fsr']:>11.6f}  {_pm['ng_fsr']:>11.6f}  "
            f"{_dng:>+10.6f}  {'—':>8}"
        )
    print()
    print("  TABLE 2 — FSR-Matched Radius  (closest ng·L to λₙ²/FSR)")
    print(_SEP)
    _H2 = (f"  {'n':>3}  {'λ_n (nm)':>13}  {'m':>6}  "
           f"{'R_fsr (µm)':>12}  {'L_fsr (µm)':>13}  "
           f"{'neff_fsr':>10}  {'ng_fsr':>10}  "
           f"{'ng·L (µm)':>12}  {'FSR (nm)':>12}  {'λ_res (nm)':>14}")
    print(_H2)
    print("  " + "─" * 130)
    for _n in range(N_SPEC_RINGS):
        if _n not in _spec_results:
            print(f"  {_n:>3}  FAILED"); continue
        _pm = _spec_results[_n]
        print(
            f"  {_n:>3}  {_pm['lam_nm']:>13.6f}  {_pm['m']:>6}  "
            f"{_pm['R_fsr_um']:>12.6f}  {_pm['L_fsr_um']:>13.6f}  "
            f"{_pm['neff_fsr']:>10.6f}  {_pm['ng_fsr']:>10.6f}  "
            f"{_pm['ngL_fsr_um']:>12.4f}  {_pm['FSR_fsr_nm']:>12.6f}  "
            f"{_pm['lam_res_fsr_nm']:>14.9f}"
        )
    print()
    print("  TABLE 3 — Phase-Matched Radii  (m·λₙ = neff·2πR condition)")
    print(_SEP)
    _H3 = (f"  {'n':>3}  {'λ_n (nm)':>13}  {'m':>6}  "
           f"{'R_pm (µm)':>12}  {'L_pm (µm)':>13}  "
           f"{'neff_pm':>10}  {'ng_pm':>10}  "
           f"{'FSR_pm (nm)':>13}  {'λ_res_pm (nm)':>16}  "
           f"{'|Δλ| (pm)':>11}  {'ΔR (nm)':>9}")
    print(_H3)
    print("  " + "─" * 130)
    for _n in range(N_SPEC_RINGS):
        if _n not in _spec_results:
            print(f"  {_n:>3}  FAILED"); continue
        _pm = _spec_results[_n]
        print(
            f"  {_n:>3}  {_pm['lam_nm']:>13.6f}  {_pm['m']:>6}  "
            f"{_pm['R_pm_um']:>12.6f}  {_pm['L_pm_um']:>13.6f}  "
            f"{_pm['neff_pm']:>10.6f}  {_pm['ng_pm']:>10.6f}  "
            f"{_pm['FSR_pm_nm']:>13.6f}  {_pm['lam_res_pm_nm']:>16.9f}  "
            f"{_pm['resid_pm_pm']:>11.5f}  {_pm['delta_R_nm']:>+9.3f}"
        )
    print()
    print("  TABLE 4 — Radiative Bend Loss  (at the phase-matched radius)")
    print(_SEP)
    _H4 = (f"  {'n':>3}  {'λ_n (nm)':>13}  {'R_pm (µm)':>12}  "
           f"{'Im(neff)':>12}  {'a_bend (dB/cm)':>15}  {'Q_bend':>11}  "
           f"{'a_tot (dB/cm)':>14}  {'Q_i_total':>11}  {'Q_loaded':>11}  {'Qi/QL':>8}")
    print(_H4)
    print("  " + "─" * 130)
    for _n in range(N_SPEC_RINGS):
        if _n not in _spec_results:
            print(f"  {_n:>3}  FAILED"); continue
        _pm    = _spec_results[_n]
        _ratio = (_pm["Q_i_total"] / _pm["Q_loaded"]
                  if np.isfinite(_pm["Q_i_total"]) else np.nan)
        print(
            f"  {_n:>3}  {_pm['lam_nm']:>13.6f}  {_pm['R_pm_um']:>12.6f}  "
            f"{_pm['neff_imag_match']:>12.3e}  {_pm['alpha_bend_match_dbcm']:>15.4f}  "
            f"{_pm['Q_bend']:>11.3e}  {_pm['alpha_total_dbcm']:>14.4f}  "
            f"{_pm['Q_i_total']:>11.3e}  {_pm['Q_loaded']:>11.3e}  {_ratio:>8.1f}"
        )
    print()
    print("  (Qi/QL >> 1  ⇒  coupling, not radiation, sets the linewidth — same"
          " argument as the sensor ring.)")
    if not SPEC_USE_PML_FOR_LOSS:
        print("  [!] SPEC_USE_PML_FOR_LOSS = False → bend loss is NOT physical "
              "(set it True).")
    print()
    print(_SEP2)
    print()
    _fsr_all = np.array([_spec_results[_n]["FSR_pm_nm"]
                         for _n in range(N_SPEC_RINGS) if _n in _spec_results])
    _r_all   = np.array([_spec_results[_n]["R_pm_um"]
                         for _n in range(N_SPEC_RINGS) if _n in _spec_results])
    _ng_all  = np.array([_spec_results[_n]["ng_pm"]
                         for _n in range(N_SPEC_RINGS) if _n in _spec_results])
    _ab_all  = np.array([_spec_results[_n]["alpha_bend_match_dbcm"]
                         for _n in range(N_SPEC_RINGS) if _n in _spec_results])
    _ab_fin  = _ab_all[np.isfinite(_ab_all)]
    print(f"  FSR spread   :  {_fsr_all.min():.5f} – {_fsr_all.max():.5f} nm  "
          f"│  Δ = {(_fsr_all.max()-_fsr_all.min())*1e3:.3f} pm")
    print(f"  Radius spread:  {_r_all.min():.4f}  – {_r_all.max():.4f}  µm  "
          f"│  Δ = {(_r_all.max()-_r_all.min())*1e3:.2f} nm")
    print(f"  ng range     :  {_ng_all.min():.6f} – {_ng_all.max():.6f}")
    if _ab_fin.size:
        print(f"  a_bend range :  {_ab_fin.min():.4f} – {_ab_fin.max():.4f} dB/cm  "
              f"(median {np.median(_ab_fin):.4f})")
    else:
        print("  a_bend range :  no finite values (check PML / warnings)")
    print()
    spec_R_pm_um         = np.array([_spec_results[_n]["R_pm_um"]
                                      for _n in range(N_SPEC_RINGS)])
    spec_L_pm_um         = np.array([_spec_results[_n]["L_pm_um"]
                                      for _n in range(N_SPEC_RINGS)])
    spec_neff_pm         = np.array([_spec_results[_n]["neff_pm"]
                                      for _n in range(N_SPEC_RINGS)])
    spec_ng_pm           = np.array([_spec_results[_n]["ng_pm"]
                                      for _n in range(N_SPEC_RINGS)])
    spec_FSR_pm_nm       = np.array([_spec_results[_n]["FSR_pm_nm"]
                                      for _n in range(N_SPEC_RINGS)])
    spec_lam_res_pm_nm   = np.array([_spec_results[_n]["lam_res_pm_nm"]
                                      for _n in range(N_SPEC_RINGS)])
    spec_m               = np.array([_spec_results[_n]["m"]
                                      for _n in range(N_SPEC_RINGS)], dtype=int)
    spec_neff_imag       = np.array([_spec_results[_n]["neff_imag_match"]
                                      for _n in range(N_SPEC_RINGS)])
    spec_alpha_bend_dbcm = np.array([_spec_results[_n]["alpha_bend_match_dbcm"]
                                      for _n in range(N_SPEC_RINGS)])
    spec_alpha_total_dbcm= np.array([_spec_results[_n]["alpha_total_dbcm"]
                                      for _n in range(N_SPEC_RINGS)])
    spec_Q_bend          = np.array([_spec_results[_n]["Q_bend"]
                                      for _n in range(N_SPEC_RINGS)])
    spec_Q_i_total       = np.array([_spec_results[_n]["Q_i_total"]
                                      for _n in range(N_SPEC_RINGS)])
    print("  Exported arrays (shape = (13,)):")
    for _vn in ("spec_R_pm_um", "spec_L_pm_um", "spec_neff_pm",
                "spec_ng_pm", "spec_FSR_pm_nm", "spec_lam_res_pm_nm", "spec_m",
                "spec_neff_imag", "spec_alpha_bend_dbcm", "spec_alpha_total_dbcm",
                "spec_Q_bend", "spec_Q_i_total"):
        print(f"    {_vn}")
    print()
    plt.rcParams.update({
        "figure.dpi":        150,
        "font.family":       "DejaVu Sans",
        "axes.titlesize":    11,
        "axes.labelsize":    10,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "legend.fontsize":   7.5,
        "lines.linewidth":   1.8,
        "lines.markersize":  5,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.28,
        "grid.linestyle":    "--",
    })
    _cmap   = cm.viridis
    _cnorm  = Normalize(vmin=0, vmax=N_SPEC_RINGS - 1)
    _colors = [_cmap(_cnorm(_n)) for _n in range(N_SPEC_RINGS)]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f"Spectrometer Ring Array — SiO₂/SiN/SiO₂  │  "
        f"w={SPEC_WG_WIDTH_NM:.0f} nm  h={_sp_core_t_um*1e3:.0f} nm  │  "
        f"FSR={SPEC_FSR_NM:.1f} nm  │  {N_SPEC_RINGS} rings",
        fontsize=12, fontweight="bold", y=1.01,
    )
    ax00 = axes[0, 0]
    for _n in range(N_SPEC_RINGS):
        if _n not in _spec_results: continue
        _pm = _spec_results[_n]
        _rv = _pm["radii_um"][_pm["valid_mask"]]
        _nv = _pm["neff_arr"][_pm["valid_mask"]]
        ax00.plot(_rv, _nv, color=_colors[_n], lw=1.4, alpha=0.85)
        ax00.scatter([_pm["R_pm_um"]], [_pm["neff_pm"]],
                     s=45, color=_colors[_n], edgecolors="k", lw=0.5, zorder=5)
    ax00.set_xlabel("Bend radius  R  (µm)")
    ax00.set_ylabel(r"$n_\mathrm{eff}$  (bent FDE)")
    ax00.set_title(r"Effective index vs Radius  (filled dots = $R_\mathrm{pm}$)")
    ax00.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax00.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax01 = axes[0, 1]
    for _n in range(N_SPEC_RINGS):
        if _n not in _spec_results: continue
        _pm = _spec_results[_n]
        _rv = _pm["radii_um"][_pm["valid_mask"]]
        _gv = _pm["ng_arr"]  [_pm["valid_mask"]]
        ax01.plot(_rv, _gv, color=_colors[_n], lw=1.4, alpha=0.85,
                  label=f"n={_n}  {_pm['lam_nm']:.2f} nm")
        ax01.scatter([_pm["R_pm_um"]], [_pm["ng_pm"]],
                     s=45, color=_colors[_n], edgecolors="k", lw=0.5, zorder=5)
        ax01.axhline(float(_ng_str[_n]),
                     color=_colors[_n], ls=":", lw=0.7, alpha=0.45)
    ax01.set_xlabel("Bend radius  R  (µm)")
    ax01.set_ylabel(r"$n_g$  (bent FDE)")
    ax01.set_title(r"Group index vs Radius  (dotted = straight $n_g$,  dots = $R_\mathrm{pm}$)")
    ax01.legend(ncol=2, fontsize=6.5, framealpha=0.8, columnspacing=0.5)
    ax01.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax01.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax10 = axes[1, 0]
    _lam_v  = [_spec_results[_n]["lam_nm"]   for _n in range(N_SPEC_RINGS) if _n in _spec_results]
    _rpm_v  = [_spec_results[_n]["R_pm_um"]  for _n in range(N_SPEC_RINGS) if _n in _spec_results]
    _rfsr_v = [_spec_results[_n]["R_fsr_um"] for _n in range(N_SPEC_RINGS) if _n in _spec_results]
    ax10.plot(_lam_v, _rpm_v,  "o-", color="#009E73", lw=2.0, ms=6,
              label=r"$R_\mathrm{pm}$  (phase-matched)")
    ax10.plot(_lam_v, _rfsr_v, "s--", color="#D55E00", lw=1.6, ms=5,
              label=r"$R_\mathrm{FSR}$  (ng·L matched)")
    ax10.set_xlabel(r"Ring resonance wavelength  $\lambda_n$  (nm)")
    ax10.set_ylabel("Radius  R  (µm)")
    ax10.set_title("Phase-Matched Radius vs Ring Wavelength")
    ax10.legend()
    ax10.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax10.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax11 = axes[1, 1]
    for _n in range(N_SPEC_RINGS):
        if _n not in _spec_results: continue
        _pm  = _spec_results[_n]
        _rv  = _pm["radii_um"][_pm["valid_mask"]]
        _glv = _pm["ngL_arr_m"][_pm["valid_mask"]] * 1e6   # µm
        _tgt = float(_lam_m_arr[_n])**2 / _fsr_m * 1e6
        ax11.plot(_rv, _glv, color=_colors[_n], lw=1.2, alpha=0.75)
        ax11.axhline(_tgt, color=_colors[_n], ls=":", lw=0.7, alpha=0.45)
        ax11.scatter([_pm["R_pm_um"]], [_pm["L_pm_um"] * _pm["ng_pm"]],
                     s=40, color=_colors[_n], edgecolors="k", lw=0.5, zorder=5)
    ax11.set_xlabel("Bend radius  R  (µm)")
    ax11.set_ylabel(r"$n_g \cdot L$  (µm)")
    ax11.set_title(r"$n_g \cdot L$ vs Radius  "
                   r"(dotted = target $\lambda_n^2/\mathrm{FSR}$,  dots = $R_\mathrm{pm}$)")
    ax11.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax11.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    _sm   = cm.ScalarMappable(cmap=_cmap, norm=_cnorm)
    _sm.set_array([])
    _cbar = fig.colorbar(_sm, ax=[ax00, ax01],
                         orientation="vertical", fraction=0.015, pad=0.02)
    _cbar.set_label("Ring index  n", fontsize=9)
    _cbar.set_ticks(range(N_SPEC_RINGS))
    _cbar.ax.tick_params(labelsize=7)
    plt.tight_layout()
    _fig_stem = DATA_DIR / f"{VERSION_NAME_SIO2}_spectrometer_rings"
    for _ext in (".png", ".pdf"):
        plt.savefig(str(_fig_stem) + _ext, dpi=150, bbox_inches="tight")
        print(f"  Saved → {str(_fig_stem) + _ext}")
    plt.close('all')
    fig5, (axb0, axb1) = plt.subplots(1, 2, figsize=(15, 5))
    fig5.suptitle(
        f"Spectrometer Ring Array — Radiative Bend Loss  │  "
        f"PML, y span {_sp_y_span_loss_um:.0f} µm, {SPEC_PML_LAYERS} layers",
        fontsize=12, fontweight="bold", y=1.02,
    )
    _any_pos = False
    for _n in range(N_SPEC_RINGS):
        if _n not in _spec_results: continue
        _pm = _spec_results[_n]
        _rv = _pm["radii_um"][_pm["valid_mask"]]
        _av = _pm["alpha_bend_arr"][_pm["valid_mask"]]
        _ok = np.isfinite(_av) & (_av > 0)
        if np.any(_ok):
            _any_pos = True
            axb0.semilogy(_rv[_ok], _av[_ok], color=_colors[_n], lw=1.3, alpha=0.85)
        if np.isfinite(_pm["alpha_bend_match_dbcm"]) and _pm["alpha_bend_match_dbcm"] > 0:
            axb0.scatter([_pm["R_pm_um"]], [_pm["alpha_bend_match_dbcm"]],
                         s=45, color=_colors[_n], edgecolors="k", lw=0.5, zorder=5)
    axb0.axhline(SPEC_ALPHA_PROP_DBCM, color="#E69F00", ls="--", lw=1.6,
                 label=f"assumed $\\alpha_\\mathrm{{prop}}$ = {SPEC_ALPHA_PROP_DBCM:.1f} dB/cm")
    axb0.set_xlabel("Bend radius  R  (µm)")
    axb0.set_ylabel(r"$\alpha_\mathrm{bend}$  (dB/cm, log scale)")
    axb0.set_title(r"Bend loss vs Radius  (dots = $R_\mathrm{pm}$)")
    axb0.legend()
    axb0.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    if not _any_pos:
        axb0.text(0.5, 0.5, "no positive alpha_bend yet\n(enable PML / check warnings)",
                  transform=axb0.transAxes, ha="center", va="center", fontsize=9)
    _lam_b = [_spec_results[_n]["lam_nm"]
              for _n in range(N_SPEC_RINGS) if _n in _spec_results]
    _ab_b  = [_spec_results[_n]["alpha_bend_match_dbcm"]
              for _n in range(N_SPEC_RINGS) if _n in _spec_results]
    axb1.plot(_lam_b, _ab_b, "^-", color="#9467BD", lw=2.0, ms=7,
              label=r"$\alpha_\mathrm{bend}$ at $R_\mathrm{pm}$")
    axb1.axhline(SPEC_ALPHA_PROP_DBCM, color="#E69F00", ls="--", lw=1.6,
                 label=f"assumed $\\alpha_\\mathrm{{prop}}$ = {SPEC_ALPHA_PROP_DBCM:.1f} dB/cm")
    axb1.set_xlabel(r"Ring resonance wavelength  $\lambda_n$  (nm)")
    axb1.set_ylabel(r"$\alpha_\mathrm{bend}$  (dB/cm)")
    axb1.set_title(r"Bend loss at the phase-matched radius vs $\lambda_n$")
    axb1.legend()
    axb1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    axb1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    plt.tight_layout()
    _fig5_stem = DATA_DIR / f"{VERSION_NAME_SIO2}_spectrometer_bendloss"
    for _ext in (".png", ".pdf"):
        plt.savefig(str(_fig5_stem) + _ext, dpi=150, bbox_inches="tight")
        print(f"  Saved → {str(_fig5_stem) + _ext}")
    plt.close('all')

    state.update({k: globals().get(k) for k in [
        'N_SPEC_RINGS', 'SPEC_ALPHA_PROP_DBCM', 'SPEC_DELTA_LAM_NG_NM', 'SPEC_DELTA_LAM_NM', 'SPEC_FSR_NM', 'SPEC_FWHM_NM',
        'SPEC_LAM0_NM', 'SPEC_LAM_NM', 'SPEC_LOSS_TRIAL_MODES', 'SPEC_MESH_DY_NM', 'SPEC_N_RADII', 'SPEC_N_UPPER',
        'SPEC_PML_LAYERS', 'SPEC_R_HALF_SPAN_UM', 'SPEC_R_MIN_FLOOR_UM', 'SPEC_USE_PML_FOR_LOSS', 'SPEC_WG_WIDTH_NM', 'SPEC_Y_SPAN_LOSS_UM',
        '_H1', '_H2', '_H3', '_H4', '_L_m', '_N_n',
        '_PM_MAX_ITER', '_PM_TOL_M', '_R_est_um', '_R_m', '_R_um', '_SEP',
        '_SEP2', '_SP_DBCM_PER_INVM', '_a_bend_match', '_a_bend_per_m', '_a_dbcm', '_a_tot_dbcm',
        '_ab_all', '_ab_b', '_ab_fin', '_abend_n', '_any_pos', '_any_uncached',
        '_av', '_c', '_cbar', '_cmap', '_cnorm', '_colors',
        '_comp_n', '_delta', '_dlam_ng_m', '_dneff_dlam_str', '_dng', '_done_n',
        '_el', '_elapsed', '_eta', '_ext', '_fig5_stem', '_fig_stem',
        '_fsr_all', '_fsr_m', '_fsr_was_cached', '_glv', '_gp', '_grp_paths',
        '_gv', '_hdr', '_i', '_l', '_lam0_n_m', '_lam0_n_nm',
        '_lam_b', '_lam_m_arr', '_lam_nm_arr', '_lam_v', '_loss_dbm', '_loss_n',
        '_lossdbm_n', '_match_full_idx', '_n', '_n_done', '_n_fsr_only', '_neff_n',
        '_neff_str', '_neff_str_fit', '_neff_v', '_ngL_m', '_ngL_m_i', '_ngL_n',
        '_ng_all', '_ng_n', '_ng_str', '_ng_v', '_nhi_n', '_nhi_v',
        '_nimag', '_nimag_n', '_nlo_n', '_nlo_v', '_nv', '_ok',
        '_pm', '_poly_str', '_r_all', '_radii_um_n', '_rate', '_ratio',
        '_remaining', '_rfsr_v', '_rg', '_rmax', '_rmin', '_rpm_v',
        '_runs', '_rv', '_sm', '_sp_core_t_um', '_sp_half_t_um', '_sp_hf',
        '_sp_mesh_y_loss', '_sp_mode', '_sp_sio2_z_ctr', '_sp_sio2_z_span', '_sp_wg_w_m', '_sp_y_span_loss_um',
        '_sp_y_span_um', '_sp_z_above_um', '_sp_z_below_um', '_sp_z_ctr', '_sp_z_span_um', '_spec_r_max',
        '_spec_r_min', '_spec_radii_um', '_spec_results', '_spec_w_idx', '_spec_w_nm', '_spec_w_um',
        '_src', '_t0', '_target_ngL_n_m', '_target_ngL_n_um', '_te_n', '_te_str',
        '_te_v', '_tgt', '_valid_n', '_valid_str', '_vn', '_wl_str_m',
        'ax00', 'ax01', 'ax10', 'ax11', 'axb0', 'axb1',
        'axes', 'fig', 'fig5', 'spec_FSR_pm_nm', 'spec_L_pm_um', 'spec_Q_bend',
        'spec_Q_i_total', 'spec_R_pm_um', 'spec_alpha_bend_dbcm', 'spec_alpha_total_dbcm', 'spec_lam_res_pm_nm', 'spec_m',
        'spec_neff_imag', 'spec_neff_pm', 'spec_ng_pm',
    ] if k in globals()})
    return state


if __name__ == "__main__":
    run()
