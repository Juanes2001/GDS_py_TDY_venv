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
save_fig = None
selected_width_nm = None

apply_style()

# ─────────────────────────────────────────────────────────
#  Module constants / design knobs for this step
# ─────────────────────────────────────────────────────────
FWHM_SENSOR_NM = SENSOR_FWHM_NM   # [nm]  sensor ring target FWHM      (config)
FWHM_SPEC_NM   = SPEC_FWHM_NM     # [nm]  spectrometer rings FWHM       (config)
RR_FSR_NM      = TARGET_FSR_NM    # [nm]  target free spectral range    (config)
RR_LAM0_NM     = LAMBDA0_NM       # [nm]  resonance wavelength          (config)
RR_WG_WIDTH_NM = (float(WG_WIDTH_OVERRIDE_NM) if WG_WIDTH_OVERRIDE_NM is not None
                  else float(WG_WIDTH_FALLBACK_NM))   # provisional; resolved in run()
RR_R_MIN_UM  = 18.0    # [µm]
RR_R_MAX_UM  = 20.0   # [µm]
RR_N_RADII   = 100
_RR_DELTA_LAM_NM = 5.0   # [nm]  half-span for central difference
RR_USE_PML_FOR_LOSS = True    # PML boundaries for the loss solve (False = loss-blind)
RR_Y_SPAN_LOSS_UM   = 16.0    # [µm]  widened lateral span so the PML is past the caustic
RR_MESH_DY_NM       = 50.0    # [nm]  target y cell size over the widened span
RR_PML_LAYERS       = 16      # PML layers (raise if alpha_bend is not converged)
RR_LOSS_TRIAL_MODES = 12      # trial modes for the (leaky) loss solve — more = safer
RR_ALPHA_PROP_DBCM  = 1.0     # [dB/cm] assumed propagation loss, for the total-Q summary
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
_y_span_loss_um = RR_Y_SPAN_LOSS_UM if RR_USE_PML_FOR_LOSS else _sim_y_span_um
_mesh_y_loss    = (int(round(_y_span_loss_um * 1e3 / RR_MESH_DY_NM))
                   if RR_USE_PML_FOR_LOSS else MESH_CELLS_Y)
_radii_um = np.linspace(RR_R_MIN_UM, RR_R_MAX_UM, RR_N_RADII)
_radii_m  = _radii_um * 1e-6
_N        = RR_N_RADII
_DBCM_PER_INVM = (10.0 / np.log(10)) / 100.0          # 1/m -> dB/cm

def _per_m_to_dbcm(a_per_m):  return a_per_m * _DBCM_PER_INVM


def _dbcm_to_per_m(a_dbcm):   return a_dbcm / _DBCM_PER_INVM


def _rr_init_hdf5(hf) -> None:
    g = hf.require_group(_RR_HDF5_GROUP)

    mg = g.require_group("metadata")
    mg.attrs["fsr_nm"]          = RR_FSR_NM
    mg.attrs["lam0_nm"]         = RR_LAM0_NM
    mg.attrs["wg_width_nm"]     = RR_WG_WIDTH_NM
    mg.attrs["wg_height_nm"]    = _core_t_um * 1e3
    mg.attrs["r_min_um"]        = RR_R_MIN_UM
    mg.attrs["r_max_um"]        = RR_R_MAX_UM
    mg.attrs["n_radii"]         = _N
    mg.attrs["delta_lam_nm"]    = _RR_DELTA_LAM_NM
    mg.attrs["n_SiN"]           = N_SIN_FIXED
    mg.attrs["n_SiO2"]          = N_SIO2_FIXED
    mg.attrs["n_upper_clad"]    = N_UPPER_CLADDING
    mg.attrs["version_name"]    = VERSION_NAME
    mg.attrs["schema"]          = "v2_bendloss"
    mg.attrs["y_span_loss_um"]  = _y_span_loss_um
    mg.attrs["pml_layers"]      = RR_PML_LAYERS
    mg.attrs["alpha_prop_dbcm"] = RR_ALPHA_PROP_DBCM
    if "timestamp_start" not in mg.attrs:
        mg.attrs["timestamp_start"] = datetime.now().isoformat()

    if "radii_um" not in mg:
        mg.create_dataset("radii_um", data=_radii_um)
    if "lam_stencil_nm" not in mg:
        mg.create_dataset("lam_stencil_nm",
                          data=np.array([RR_LAM0_NM - _RR_DELTA_LAM_NM,
                                         RR_LAM0_NM,
                                         RR_LAM0_NM + _RR_DELTA_LAM_NM]))

    rg = g.require_group("results")
    _nan = np.full(_N, np.nan, dtype=np.float64)
    for ds_name in ("neff", "ng", "te_frac", "ngL_um", "neff_lo", "neff_hi",
                    "neff_imag", "alpha_bend_dbcm", "loss_dbm"):
        if ds_name not in rg:
            rg.create_dataset(ds_name, data=_nan.copy(), chunks=(_N,))

    fg = g.require_group("flags")
    if "computed" not in fg:
        fg.create_dataset("computed", data=np.zeros(_N, dtype=bool), chunks=(_N,))
    if "loss_computed" not in fg:
        # seed from any existing Im(neff) so prior bend-loss runs are not redone
        _seed = (np.isfinite(rg["neff_imag"][:]) if "neff_imag" in rg
                 else np.zeros(_N, dtype=bool))
        fg.create_dataset("loss_computed", data=_seed, chunks=(_N,))


def _rr_load_cache(hf,
                   neff_arr, ng_arr, te_arr,
                   ngL_arr, neff_lo_arr, neff_hi_arr,
                   neff_imag_arr, alpha_bend_arr, loss_dbm_arr,
                   computed, loss_done) -> None:
    g  = hf[_RR_HDF5_GROUP]
    rg = g["results"]
    fg = g["flags"]

    neff_arr[:]       = rg["neff"][:]
    ng_arr[:]         = rg["ng"][:]
    te_arr[:]         = rg["te_frac"][:]
    ngL_arr[:]        = rg["ngL_um"][:] * 1e-6   # stored in µm, keep in m here
    neff_lo_arr[:]    = rg["neff_lo"][:]
    neff_hi_arr[:]    = rg["neff_hi"][:]
    neff_imag_arr[:]  = rg["neff_imag"][:]
    alpha_bend_arr[:] = rg["alpha_bend_dbcm"][:]
    loss_dbm_arr[:]   = rg["loss_dbm"][:]
    computed[:]       = fg["computed"][:]
    loss_done[:]      = fg["loss_computed"][:]


def _rr_set_near_n(mode, n_target) -> None:
    """
    Seed the FDE eigensolver to search near a target effective index.
    Version-tolerant: set whichever of these properties this build exposes.
    This is the key robustness fix for leaky bent modes (mode finding).
    """
    for _p, _v in (("use max index", 0),
                   ("search", "near n"),
                   ("n", float(n_target))):
        try:
            mode.set(_p, _v)
        except Exception:
            pass


def _rr_build_fde(mode, radius_m: float, wavelength_m: float,
                  for_loss: bool = False, neff_guess=None) -> None:
    """
    Build bent waveguide cross-section + FDE solver for one (R, λ) point.

    for_loss=False : original LIGHT geometry, default search (n_g stencil).
    for_loss=True  : WIDE lateral domain + PML, more trial modes, and (when a
                     neff_guess is given) a "near n" search so the leaky
                     fundamental is reliably found.
    """
    m = mode
    m.switchtolayout()
    m.selectall()
    m.delete()

    _yspan_um = _y_span_loss_um if for_loss else _sim_y_span_um
    _mesh_y   = _mesh_y_loss    if for_loss else MESH_CELLS_Y
    _n_trial  = (RR_LOSS_TRIAL_MODES if for_loss else N_MODES_REQUEST)

    m.addfde()
    m.set("solver type",           "2D X normal")
    m.set("x",                     0.0)
    m.set("y",                     0.0)
    m.set("z",                     _sim_z_ctr_um  * 1e-6)
    m.set("y span",                _yspan_um      * 1e-6)
    m.set("z span",                _sim_z_span_um * 1e-6)
    m.set("wavelength",            wavelength_m)
    m.set("number of trial modes", _n_trial)
    m.set("mesh cells y",          _mesh_y)
    m.set("mesh cells z",          MESH_CELLS_Z)
    m.set("bent waveguide",        1)
    m.set("bend radius",           radius_m)
    m.set("bend orientation",      0)

    # PML on all outer boundaries for the loss solve so the radiated tail is
    # absorbed (a metal boundary gives a trapped, real-neff mode => Im=0).
    if for_loss and RR_USE_PML_FOR_LOSS:
        for _bc in ("y min bc", "y max bc", "z min bc", "z max bc"):
            m.set(_bc, "PML")
        m.set("pml layers", RR_PML_LAYERS)

    # seed the eigensolver near the known real n_eff (robust mode finding)
    if for_loss and (neff_guess is not None):
        _rr_set_near_n(m, neff_guess)

    m.addrect()
    m.set("name",    "RR_bg")
    m.set("x",       0.0);  m.set("x span", 1.0e-6)
    m.set("y",       0.0);  m.set("y span", _yspan_um * 1e-6)
    m.set("z",       _sim_z_ctr_um  * 1e-6)
    m.set("z span",  _sim_z_span_um * 1e-6)
    m.set("material", "<Object defined dielectric>")
    m.set("index",    N_UPPER_CLADDING)

    m.addrect()
    m.set("name",    "RR_lower_clad")
    m.set("x",       0.0);  m.set("x span", 1.0e-6)
    m.set("y",       0.0);  m.set("y span", _yspan_um * 1e-6)
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


def _rr_collect_modes(mode, n_try: int):
    """
    Robustly collect found modes as a list of (k, neff_complex, te_fraction)
    by reading mode1..mode<n_try> and stopping at the first that is absent.
    Returns [] if findmodes produced nothing (so the caller can handle it).
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


def _rr_solve_neff(mode, radius_m: float, wavelength_m: float,
                   for_loss: bool = False, neff_guess=None):
    """
    Single FDE solve. Returns (Re(neff), Im(neff), TE fraction, loss_dB_per_m).
    loss_dB_per_m is only read for the loss solve (else NaN).

    Light solve (for_loss=False): selects mode1 (fundamental) — unchanged.
    Loss solve  (for_loss=True) : selects the TE-like mode whose Re(neff) is
                                  closest to neff_guess (robust against leaky /
                                  spurious PML modes); raises if NO mode found.
    """
    _n_try = RR_LOSS_TRIAL_MODES if for_loss else N_MODES_REQUEST
    _rr_build_fde(mode, radius_m, wavelength_m,
                  for_loss=for_loss, neff_guess=neff_guess)
    mode.run()
    mode.findmodes()

    found = _rr_collect_modes(mode, _n_try)
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


def _rr_fsr_stencil(mode, radius_m: float):
    """Three LIGHT solves -> (neff_at_lam0, ng, te_frac, neff_lo, neff_hi).
    Identical to the original FSR-matching computation (n_g unchanged)."""
    neff_lo, _, _,    _ = _rr_solve_neff(mode, radius_m, _lam_lo_m, for_loss=False)
    neff_0,  _, te_v, _ = _rr_solve_neff(mode, radius_m, _lam0_m,   for_loss=False)
    neff_hi, _, _,    _ = _rr_solve_neff(mode, radius_m, _lam_hi_m, for_loss=False)
    ng = neff_0 - _lam0_m * (neff_hi - neff_lo) / _dlam_m
    return neff_0, ng, te_v, neff_lo, neff_hi


def _rr_loss_solve(mode, radius_m: float, neff_guess: float):
    """One PML solve at lam0 (seeded near neff_guess) -> (Im(neff), loss_dB/m)."""
    _, nimag, _, loss_dbm = _rr_solve_neff(mode, radius_m, _lam0_m,
                                           for_loss=True, neff_guess=neff_guess)
    return nimag, loss_dbm

def run(state=None):
    """Execute this step. `state` carries the shared namespace
    between steps (the notebook's old kernel globals + bridge).
    Returns the updated `state`."""
    state = {} if state is None else state
    global FWHM_SENSOR_NM, FWHM_SPEC_NM, RR_ALPHA_PROP_DBCM, RR_FSR_NM, RR_LAM0_NM, RR_LOSS_TRIAL_MODES, RR_MESH_DY_NM, RR_N_RADII, RR_PML_LAYERS, RR_R_MAX_UM, RR_R_MIN_UM, RR_USE_PML_FOR_LOSS, RR_WG_WIDTH_NM, RR_Y_SPAN_LOSS_UM, _C_BEND, _C_BEST, _C_NEFF, _C_NG, _C_NGL, _C_TGT, _DBCM_PER_INVM, _FSR_m, _L_m, _N, _RR_DELTA_LAM_NM, _RR_GROUP_KEY, _RR_HDF5_GROUP, _R_m, _R_um, _R_v, _a_bend_per_m, _a_dbcm, _a_tot_dbcm, _abend_v, _alpha_bend_arr, _bi, _computed, _core_t_um, _delta, _dist, _dlam_m, _done, _elapsed, _elapsed_total, _eta, _fsr_was_cached, _group_existed, _half_t_um, _have_loss, _hdr, _hf, _i, _lam0_m, _lam_hi_m, _lam_lo_m, _loss_dbm, _loss_dbm_arr, _loss_done, _mesh_y_loss, _n_done, _n_fsr_only, _neff_arr, _neff_hi_arr, _neff_imag_arr, _neff_lo_arr, _neff_v, _ngL_arr, _ngL_m, _ngL_v, _ng_arr, _ng_v, _nhi, _nimag, _nimag_v, _nlo, _pos, _radii_m, _radii_um, _rate, _remaining, _rg, _rr_mode, _runs_done, _sim_y_span_um, _sim_z_above_um, _sim_z_below_um, _sim_z_ctr_um, _sim_z_span_um, _sio2_z_ctr_um, _sio2_z_span_um, _src, _suffix, _t0, _target_ngL_m, _target_ngL_um, _te_arr, _te_v, _valid, _wg_w_m, _y_span_loss_um, ax1, ax2, ax3, ax4, fig1, fig2, fig3, fig4, rr_FSR_pred_nm, rr_Q_i_total, rr_Q_loaded, rr_best_L_um, rr_best_Q_bend, rr_best_R_um, rr_best_alpha_bend_dbcm, rr_best_neff, rr_best_neff_imag, rr_best_ng, rr_best_ngL_um
    globals()['lumapi'] = import_lumapi()
    globals().update(state)

    if WG_WIDTH_OVERRIDE_NM is not None:
        RR_WG_WIDTH_NM = float(WG_WIDTH_OVERRIDE_NM)
    elif selected_width_nm is not None:
        RR_WG_WIDTH_NM = float(selected_width_nm)
    else:
        RR_WG_WIDTH_NM = float(WG_WIDTH_FALLBACK_NM)
        log.warning("step3: no single-mode width from the modal step and no "
                    f"WG_WIDTH_OVERRIDE_NM; using fallback {RR_WG_WIDTH_NM:.0f} nm.")
    _wg_w_m       = RR_WG_WIDTH_NM * 1e-9
    _RR_GROUP_KEY = (
        f"rr_{RR_FSR_NM:.0f}nm_{RR_LAM0_NM:.0f}nm_"
        f"{RR_WG_WIDTH_NM:.0f}nm_"
        f"{RR_R_MIN_UM:.1f}-{RR_R_MAX_UM:.1f}um_{RR_N_RADII}pts"
    )
    _RR_HDF5_GROUP = f"ring_radius_sweep/{_RR_GROUP_KEY}"
    log.info(f"step3 single-mode working width = {RR_WG_WIDTH_NM:.1f} nm "
             f"(override={WG_WIDTH_OVERRIDE_NM}, selected={selected_width_nm})")
    print("=" * 65)
    print("  Ring Resonator — Radius Sweep for FSR Matching + Bend Loss")
    print("=" * 65)
    print(f"  Target FSR           : {RR_FSR_NM:.2f} nm")
    print(f"  Resonance wavelength : {RR_LAM0_NM:.1f} nm")
    print(f"  Waveguide width      : {RR_WG_WIDTH_NM:.0f} nm")
    print(f"  Waveguide height     : {_core_t_um*1e3:.0f} nm")
    print(f"  n_core               : {N_SIN_FIXED}   (N_SIN_FIXED)")
    print(f"  n_lower_clad         : {N_SIO2_FIXED}  (N_SIO2_FIXED)")
    print(f"  n_upper_clad         : {N_UPPER_CLADDING}   (N_UPPER_CLADDING)")
    print(f"  Required ng·L        : {_target_ngL_um:.4f} µm")
    print(f"  ng stencil           : ±{_RR_DELTA_LAM_NM:.1f} nm  "
          f"(3 light solves for n_g + 1 PML solve for loss, per radius)")
    print(f"  Radius sweep         : {RR_R_MIN_UM:.1f} – {RR_R_MAX_UM:.1f} µm  ({_N} pts)")
    print(f"  Bend-loss solve      : PML={RR_USE_PML_FOR_LOSS}, "
          f"y span {_y_span_loss_um:.1f} µm, mesh_y {_mesh_y_loss}, "
          f"{RR_PML_LAYERS} PML layers, near-n seed, {RR_LOSS_TRIAL_MODES} trial modes")
    print(f"  HDF5 group           : {_RR_HDF5_GROUP}")
    print(f"  HDF5 file            : {HDF5_PATH}")
    print("=" * 65)
    _neff_arr       = np.full(_N, np.nan)
    _ng_arr         = np.full(_N, np.nan)
    _te_arr         = np.full(_N, np.nan)
    _ngL_arr        = np.full(_N, np.nan)   # [m]
    _neff_lo_arr    = np.full(_N, np.nan)
    _neff_hi_arr    = np.full(_N, np.nan)
    _neff_imag_arr  = np.full(_N, np.nan)   # NEW
    _alpha_bend_arr = np.full(_N, np.nan)   # NEW [dB/cm]
    _loss_dbm_arr   = np.full(_N, np.nan)   # NEW [dB/m]
    _computed       = np.zeros(_N, dtype=bool)   # FSR done
    _loss_done      = np.zeros(_N, dtype=bool)   # bend-loss attempted
    _hf = h5py.File(HDF5_PATH, "a")   # 'a' = read/write, create if missing
    _group_existed = _RR_HDF5_GROUP in _hf
    _rr_init_hdf5(_hf)                # create group and/or add any missing datasets
    _hf.flush()
    if _group_existed:
        log.info(f"Ring sweep group found → {_RR_HDF5_GROUP}")
        _rr_load_cache(
            _hf,
            _neff_arr, _ng_arr, _te_arr,
            _ngL_arr, _neff_lo_arr, _neff_hi_arr,
            _neff_imag_arr, _alpha_bend_arr, _loss_dbm_arr,
            _computed, _loss_done,
        )
    _done       = _computed & _loss_done
    _n_done     = int(_done.sum())
    _n_fsr_only = int((_computed & ~_loss_done).sum())
    _remaining  = _N - _n_done
    if _group_existed:
        log.info(f"Fully cached (FSR + loss): {_n_done}/{_N}  |  to do: {_remaining}")
        if _n_fsr_only > 0:
            log.info(f"{_n_fsr_only} radii have FSR cached but no bend loss → only the "
                     f"(fast) PML loss solve will run for them.")
        if _remaining == 0:
            log.info("All radii fully cached — skipping FDE entirely.")
    else:
        log.info(f"No ring sweep cache found — initialised group: {_RR_HDF5_GROUP}")
    _hdr = (f"{'Radius (µm)':>12}  {'neff':>10}  {'ng':>10}  {'TE frac':>8}  "
            f"{'ng·L (µm)':>12}  {'a_bend(dB/cm)':>14}  {'Δ/target':>10}  {'source':>8}")
    print(f"\n{_hdr}")
    print("-" * len(_hdr))
    for _i in range(_N):
        if _done[_i]:
            _ngL_m = _ngL_arr[_i]
            _delta = (_ngL_m - _target_ngL_m) / _target_ngL_m * 100.0
            print(f"  {_radii_um[_i]:>10.2f}  {_neff_arr[_i]:>10.4f}  "
                  f"{_ng_arr[_i]:>10.4f}  {_te_arr[_i]:>8.3f}  "
                  f"{_ngL_m*1e6:>12.4f}  {_alpha_bend_arr[_i]:>14.4f}  "
                  f"{_delta:>+9.2f}%  {'cache':>8}\n")
    if _remaining > 0:
        _runs_done = 0
        _t0 = time.time()
        log.info(f"Launching MODE session  ({_remaining} radii to (re)process) …")
        _rr_mode = lumapi.MODE(hide=False)

        try:
            for _i, (_R_um, _R_m) in enumerate(zip(_radii_um, _radii_m)):
                if _computed[_i] and _loss_done[_i]:
                    continue   # fully done — skip

                _fsr_was_cached = bool(_computed[_i])

                # ── (a) FSR stencil (n_g) — only if not already cached ───────────
                if not _computed[_i]:
                    try:
                        _neff_v, _ng_v, _te_v, _nlo, _nhi = _rr_fsr_stencil(_rr_mode, _R_m)
                    except Exception as _exc:
                        log.warning(f"  R = {_R_um:6.2f} µm  FSR solve FAILED: {_exc}")
                        # give up this radius entirely (avoid retrying a broken setup)
                        _computed[_i]  = True
                        _loss_done[_i] = True
                        _hf[f"{_RR_HDF5_GROUP}/flags/computed"][_i]      = True
                        _hf[f"{_RR_HDF5_GROUP}/flags/loss_computed"][_i] = True
                        _hf.flush()
                        continue

                    _L_m   = 2.0 * np.pi * _R_m
                    _ngL_m = _ng_v * _L_m
                    _neff_arr[_i]    = _neff_v
                    _ng_arr[_i]      = _ng_v
                    _te_arr[_i]      = _te_v
                    _ngL_arr[_i]     = _ngL_m
                    _neff_lo_arr[_i] = _nlo
                    _neff_hi_arr[_i] = _nhi
                    _computed[_i]    = True

                    _rg = _hf[f"{_RR_HDF5_GROUP}/results"]
                    _rg["neff"]   [_i] = _neff_v
                    _rg["ng"]     [_i] = _ng_v
                    _rg["te_frac"][_i] = _te_v
                    _rg["ngL_um"] [_i] = _ngL_m * 1e6
                    _rg["neff_lo"][_i] = _nlo
                    _rg["neff_hi"][_i] = _nhi
                    _hf[f"{_RR_HDF5_GROUP}/flags/computed"][_i] = True
                    _hf.flush()
                else:
                    _neff_v = _neff_arr[_i]      # reuse cached FSR result as the seed

                # ── (b) radiative bend loss — PML solve seeded by Re(neff) ───────
                try:
                    _nimag, _loss_dbm = _rr_loss_solve(_rr_mode, _R_m, neff_guess=_neff_v)
                except Exception as _exc:
                    log.warning(f"  R = {_R_um:6.2f} µm  bend-loss solve FAILED: {_exc}; "
                                f"alpha_bend set to NaN (FSR data kept)")
                    _nimag, _loss_dbm = np.nan, np.nan

                if np.isfinite(_nimag):
                    _a_dbcm = _per_m_to_dbcm(4.0 * np.pi * abs(_nimag) / _lam0_m)
                else:
                    _a_dbcm = np.nan

                _neff_imag_arr[_i]  = _nimag
                _alpha_bend_arr[_i] = _a_dbcm
                _loss_dbm_arr[_i]   = _loss_dbm
                _loss_done[_i]      = True

                _rg = _hf[f"{_RR_HDF5_GROUP}/results"]
                _rg["neff_imag"]      [_i] = _nimag
                _rg["alpha_bend_dbcm"][_i] = _a_dbcm
                _rg["loss_dbm"]       [_i] = _loss_dbm
                _hf[f"{_RR_HDF5_GROUP}/flags/loss_computed"][_i] = True
                _hf.flush()

                _runs_done += 1
                _ngL_m = _ngL_arr[_i]
                _delta = (_ngL_m - _target_ngL_m) / _target_ngL_m * 100.0
                _src   = "loss" if _fsr_was_cached else "FDE"
                print(f"  {_R_um:>10.2f}  {_neff_arr[_i]:>10.4f}  {_ng_arr[_i]:>10.4f}  "
                      f"{_te_arr[_i]:>8.3f}  {_ngL_m*1e6:>12.4f}  {_a_dbcm:>14.4f}  "
                      f"{_delta:>+9.2f}%  {_src:>8}")

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
                     f"({_runs_done} radii processed in {_elapsed_total:.1f} s)")
    _hf[_RR_HDF5_GROUP]["metadata"].attrs["timestamp_end"]  = datetime.now().isoformat()
    _hf[_RR_HDF5_GROUP]["metadata"].attrs["runs_completed"] = int((_computed & _loss_done).sum())
    _hf.flush()
    _hf.close()
    log.info(f"HDF5 closed  →  {HDF5_PATH}")
    _valid = ~np.isnan(_ng_arr)
    if not np.any(_valid):
        raise RuntimeError(
            "All radius sweep FSR runs failed.  "
            "Check the WARNING lines above for the specific error."
        )
    _R_v     = _radii_um[_valid]
    _neff_v  = _neff_arr[_valid]
    _ng_v    = _ng_arr[_valid]
    _te_v    = _te_arr[_valid]
    _ngL_v   = _ngL_arr[_valid]          # [m]
    _abend_v = _alpha_bend_arr[_valid]   # [dB/cm]  (may contain NaN where loss failed)
    _nimag_v = _neff_imag_arr[_valid]
    _dist = np.abs(_ngL_v - _target_ngL_m)
    _bi   = int(np.argmin(_dist))
    rr_best_R_um            = _R_v[_bi]
    rr_best_L_um            = 2.0 * np.pi * rr_best_R_um
    rr_best_neff            = _neff_v[_bi]
    rr_best_ng              = _ng_v[_bi]
    rr_best_ngL_um          = _ngL_v[_bi] * 1e6
    rr_FSR_pred_nm          = (_lam0_m**2 / _ngL_v[_bi]) * 1e9
    rr_best_neff_imag       = _nimag_v[_bi]
    rr_best_alpha_bend_dbcm = _abend_v[_bi]
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
    _have_loss = bool(np.isfinite(rr_best_alpha_bend_dbcm))
    if _have_loss:
        _a_bend_per_m  = _dbcm_to_per_m(rr_best_alpha_bend_dbcm)
        rr_best_Q_bend = (2.0 * np.pi * rr_best_ng / (_a_bend_per_m * _lam0_m)
                          if _a_bend_per_m > 0 else float("inf"))
        _a_tot_dbcm    = RR_ALPHA_PROP_DBCM + rr_best_alpha_bend_dbcm
        rr_Q_i_total   = 2.0 * np.pi * rr_best_ng / (_dbcm_to_per_m(_a_tot_dbcm) * _lam0_m)
    else:
        _a_bend_per_m  = np.nan
        rr_best_Q_bend = np.nan
        _a_tot_dbcm    = np.nan
        rr_Q_i_total   = np.nan
    rr_Q_loaded = RR_LAM0_NM / FWHM_SENSOR_NM
    print("\n" + "=" * 65)
    print("  RADIATIVE BEND LOSS  (at the matched radius)")
    print("=" * 65)
    if _have_loss:
        print(f"  Im(neff)                 = {rr_best_neff_imag:.3e}")
        print(f"  alpha_bend               = {rr_best_alpha_bend_dbcm:.4f} dB/cm "
              f"(= {_a_bend_per_m:.3e} 1/m)")
        print(f"  Q_bend (radiation only)  = {rr_best_Q_bend:.3e}")
        print(f"  alpha_prop (assumed)     = {RR_ALPHA_PROP_DBCM:.2f} dB/cm")
        print(f"  alpha_total = prop+bend  = {_a_tot_dbcm:.4f} dB/cm")
        print(f"  Q_i (total, loss-limited)= {rr_Q_i_total:.3e}")
        print(f"  Q_loaded (FWHM={FWHM_SENSOR_NM:.2f} nm)  = {rr_Q_loaded:.3e}")
        print(f"  Q_i / Q_loaded           = {rr_Q_i_total/rr_Q_loaded:.1f}   "
              f"(>> 1  ⇒  coupling sets the linewidth)")
    else:
        print("  Bend-loss solve did not return a value at the matched radius.")
        print("  See the WARNING lines above; try widening RR_Y_SPAN_LOSS_UM or")
        print("  refining RR_MESH_DY_NM, then re-run (only the loss solve repeats).")
    if not RR_USE_PML_FOR_LOSS:
        print("  [!] RR_USE_PML_FOR_LOSS = False → loss is NOT physical (set it True).")
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
    _C_BEND = "#9467BD"
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
    fig1.tight_layout()
    save_fig(fig1, f"{VERSION_NAME}_ring_radius_ng")
    plt.close(fig1)
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
    fig2.tight_layout()
    save_fig(fig2, f"{VERSION_NAME}_ring_radius_neff")
    plt.close(fig2)
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
    fig3.tight_layout()
    save_fig(fig3, f"{VERSION_NAME}_ring_radius_ngL")
    plt.close(fig3)
    fig4, ax4 = plt.subplots(figsize=(7, 4.5))
    _pos = np.isfinite(_abend_v) & (_abend_v > 0)    # semilog needs positive values
    if np.any(_pos):
        ax4.semilogy(_R_v[_pos], _abend_v[_pos], "^-", color=_C_BEND,
                     label=r"$\alpha_\mathrm{bend}$  (FDE, from Im$(n_{eff})$)")
        ax4.axhline(RR_ALPHA_PROP_DBCM, color=_C_TGT, ls="--", lw=1.6,
                    label=f"assumed $\\alpha_\\mathrm{{prop}}$ = {RR_ALPHA_PROP_DBCM:.1f} dB/cm")
        if _have_loss:
            ax4.axvline(rr_best_R_um, color=_C_BEST, ls=":", lw=1.6,
                        label=f"matched R = {rr_best_R_um:.2f} µm "
                              f"→ $\\alpha_\\mathrm{{bend}}$ = {rr_best_alpha_bend_dbcm:.3g} dB/cm")
        ax4.set_ylabel(r"$\alpha_\mathrm{bend}$  (dB/cm, log scale)")
    else:
        ax4.plot(_R_v, _abend_v, "^-", color=_C_BEND, label=r"$\alpha_\mathrm{bend}$")
        ax4.set_ylabel(r"$\alpha_\mathrm{bend}$  (dB/cm)")
        ax4.text(0.5, 0.5, "no positive alpha_bend yet\n(enable PML / check warnings)",
                 transform=ax4.transAxes, ha="center", va="center", fontsize=9)
    ax4.set_xlabel("Bend radius  $R$  (µm)")
    ax4.set_title(f"Radiative Bend Loss vs. Bend Radius\n({_suffix})")
    ax4.legend(fontsize=8)
    ax4.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig4.tight_layout()
    save_fig(fig4, f"{VERSION_NAME}_ring_radius_bendloss")
    plt.close(fig4)
    print(f"\n  Cached in : {HDF5_PATH}")
    print(f"  HDF5 group: {_RR_HDF5_GROUP}")
    print(f"\n  Variables exported to the next cell:")
    print(f"    rr_best_R_um            = {rr_best_R_um:.4f}   # µm")
    print(f"    rr_best_L_um            = {rr_best_L_um:.4f}   # µm")
    print(f"    rr_best_neff            = {rr_best_neff:.6f}")
    print(f"    rr_best_ng              = {rr_best_ng:.6f}")
    print(f"    rr_FSR_pred_nm          = {rr_FSR_pred_nm:.4f}   # nm")
    print(f"    rr_best_neff_imag       = {rr_best_neff_imag:.3e}")
    print(f"    rr_best_alpha_bend_dbcm = {rr_best_alpha_bend_dbcm:.4f}   # dB/cm")
    print(f"    rr_best_Q_bend          = {rr_best_Q_bend:.3e}")
    print(f"    rr_Q_i_total            = {rr_Q_i_total:.3e}   # prop+bend, loss-limited")
    print(f"    rr_Q_loaded             = {rr_Q_loaded:.3e}   # FWHM = {FWHM_SENSOR_NM} nm")

    state.update({k: globals().get(k) for k in [
        'FWHM_SENSOR_NM', 'FWHM_SPEC_NM', 'RR_ALPHA_PROP_DBCM', 'RR_FSR_NM', 'RR_LAM0_NM', 'RR_LOSS_TRIAL_MODES',
        'RR_MESH_DY_NM', 'RR_N_RADII', 'RR_PML_LAYERS', 'RR_R_MAX_UM', 'RR_R_MIN_UM', 'RR_USE_PML_FOR_LOSS',
        'RR_WG_WIDTH_NM', 'RR_Y_SPAN_LOSS_UM', '_C_BEND', '_C_BEST', '_C_NEFF', '_C_NG',
        '_C_NGL', '_C_TGT', '_DBCM_PER_INVM', '_FSR_m', '_L_m', '_N',
        '_RR_DELTA_LAM_NM', '_RR_GROUP_KEY', '_RR_HDF5_GROUP', '_R_m', '_R_um', '_R_v',
        '_a_bend_per_m', '_a_dbcm', '_a_tot_dbcm', '_abend_v', '_alpha_bend_arr', '_bi',
        '_computed', '_core_t_um', '_delta', '_dist', '_dlam_m', '_done',
        '_elapsed', '_elapsed_total', '_eta', '_fsr_was_cached', '_group_existed', '_half_t_um',
        '_have_loss', '_hdr', '_hf', '_i', '_lam0_m', '_lam_hi_m',
        '_lam_lo_m', '_loss_dbm', '_loss_dbm_arr', '_loss_done', '_mesh_y_loss', '_n_done',
        '_n_fsr_only', '_neff_arr', '_neff_hi_arr', '_neff_imag_arr', '_neff_lo_arr', '_neff_v',
        '_ngL_arr', '_ngL_m', '_ngL_v', '_ng_arr', '_ng_v', '_nhi',
        '_nimag', '_nimag_v', '_nlo', '_pos', '_radii_m', '_radii_um',
        '_rate', '_remaining', '_rg', '_rr_mode', '_runs_done', '_sim_y_span_um',
        '_sim_z_above_um', '_sim_z_below_um', '_sim_z_ctr_um', '_sim_z_span_um', '_sio2_z_ctr_um', '_sio2_z_span_um',
        '_src', '_suffix', '_t0', '_target_ngL_m', '_target_ngL_um', '_te_arr',
        '_te_v', '_valid', '_wg_w_m', '_y_span_loss_um', 'ax1', 'ax2',
        'ax3', 'ax4', 'fig1', 'fig2', 'fig3', 'fig4',
        'rr_FSR_pred_nm', 'rr_Q_i_total', 'rr_Q_loaded', 'rr_best_L_um', 'rr_best_Q_bend', 'rr_best_R_um',
        'rr_best_alpha_bend_dbcm', 'rr_best_neff', 'rr_best_neff_imag', 'rr_best_ng', 'rr_best_ngL_um',
    ] if k in globals()})
    return state


if __name__ == "__main__":
    run()
