"""
step1_cascade_sweep.py — 14-ring cascade builder, ONA multiport sweep engine, HDF5 cache and the core result-plotting helpers.

This is the heart of the INTERCONNECT package and the engine that
every other step imports from. It builds the unidirectional 14-ring
cascade (RING_1 = aqueous sensor, RING_2..14 = SiO2 spectrometer
drops) wired to a 15-input ONA, sweeps the sensor-ring n_eff/n_g
(SWEEP_NEFF/SWEEP_NG) with per-point HDF5 caching/resume, and
provides get_results()/_valid_mask() plus the drop-power, through-
port and resonance-tracking figures. run() executes the full sweep
and returns the populated `state` (sweep_results / computed /
wavelengths_m) consumed by the downstream steps.
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
exc = None
k = None

apply_style()

# ─────────────────────────────────────────────────────────
#  Module constants / design knobs for this step
# ─────────────────────────────────────────────────────────
THROUGH_SENSOR_INPUT  = 1    # ONA input for RING_1 through
THROUGH_FINAL_INPUT   = 15   # ONA input for RING_14 through

def drop_ona_input(ring_k: int) -> int:
    """Return the ONA input port number (1-based) for RING_k drop.
    ring_k = 2..14  →  ONA input = ring_k  (i.e. 2..14)
    """
    assert 2 <= ring_k <= N_RINGS, f"ring_k must be 2..{N_RINGS}"
    return ring_k   # ONA input 2 = RING_2 drop, ..., input 14 = RING_14 drop


def ring_name(ring_id: int) -> str:
    return f"RING_{ring_id}"


class ICScriptError(RuntimeError):
    pass


def _eval(ic, cmd: str) -> None:
    cmd = cmd.strip().rstrip(";") + ";"
    try:
        ic.eval(cmd)
    except Exception as exc:
        raise ICScriptError(
            f"\n  INTERCONNECT rejected:\n    {cmd}\n  Error: {exc}"
        ) from exc


def _try_eval(ic, cmd: str) -> bool:
    cmd = cmd.strip().rstrip(";") + ";"
    try:
        ic.eval(cmd)
        return True
    except Exception:
        return False


def _apply_ring_params(ic, ring_idx: int,
                       neff_override: Optional[float] = None,
                       ng_override:   Optional[float] = None) -> None:
    name = ring_name(ring_idx + 1)
    neff = neff_override if neff_override is not None else float(RING_NEFF_TE[ring_idx])
    ng   = ng_override   if ng_override   is not None else float(RING_NG_TE[ring_idx])
    pol  = RING_POLARIZATION[ring_idx].upper()
    d_si = (float(RING_D_TE_PS2_PER_KM[ring_idx] if pol == "TE"
                  else RING_D_TM_PS2_PER_KM[ring_idx]) * 1e-15)
    res_hz        = SPEED_OF_LIGHT / float(RING_LAMBDA_RES_M[ring_idx])
    circumference = RING_RADIUS_M[ring_idx] * 2.0 * math.pi

    _eval(ic, f'setnamed("{name}", "length",                   {circumference:.12e})')
    _eval(ic, f'setnamed("{name}", "frequency",                {res_hz:.12e})')
    _eval(ic, f'setnamed("{name}", "effective index 1",        {neff:.12f})')
    _eval(ic, f'setnamed("{name}", "group index 1",            {ng:.12f})')
    _eval(ic, f'setnamed("{name}", "loss 1",                   {RING_LOSS_DB_PER_M[ring_idx]:.6f})')
    _eval(ic, f'setnamed("{name}", "dispersion 1",             {d_si:.12e})')
    _eval(ic, f'setnamed("{name}", "coupling coefficient 1 1", {RING_KAPPA_INPUT_SQ[ring_idx]:.12f})')
    _eval(ic, f'setnamed("{name}", "coupling coefficient 1 2", {RING_KAPPA_DROP_SQ[ring_idx]:.12f})')
    _eval(ic, f'setnamed("{name}", "configuration", "unidirectional")')


def _update_ring1_neff_ng(ic, neff: float, ng: float) -> None:
    name = ring_name(1)
    _eval(ic, f'setnamed("{name}", "effective index 1", {neff:.12f})')
    _eval(ic, f'setnamed("{name}", "group index 1",     {ng:.12f})')


def _build_circuit(ic) -> None:
    """
    Build the 14-ring cascade with ONA multiport measurement.

    V3 TOPOLOGY
    ────────────────────────────────────────────────────────────────
    ONA  output    → RING_1  input

    RING_1  out1   → ONA  input 1          (sensor through)
    RING_1  out2   → RING_2 input           (sensor drop → cascade)

    For n = 2..13:
      RING_n  out1 → RING_{n+1} input       (through → next ring)
      RING_n  out2 → ONA  input n            (drop → ONA input n)

    RING_14 out1   → ONA  input 15          (final through)
    RING_14 out2   → ONA  input 14          (RING_14 drop → ONA input 14)

    ONA input mapping:
      input  1 = RING_1  through  (sensor)
      input  k = RING_k  drop     k = 2..14
      input 15 = RING_14 through  (cascade end)
    """
    _eval(ic, "switchtodesign")
    _try_eval(ic, "selectall")
    _try_eval(ic, "delete")

    pwr_W   = 10.0 ** (ONA_POWER_DBM / 10.0) * 1e-3
    f_start = SPEED_OF_LIGHT / ONA_LAMBDA_STOP_M
    f_stop  = SPEED_OF_LIGHT / ONA_LAMBDA_START_M

    # ── ONA — 15 input ports ──────────────────────────────────────────────────
    _eval(ic, 'addelement("Optical Network Analyzer")')
    _eval(ic, f'set("name", "{ONA_NAME}")')
    _try_eval(ic, 'set("x position", 0)')
    _try_eval(ic, 'set("y position", 0)')
    _eval(ic, f'setnamed("{ONA_NAME}", "input parameter",       "start and stop")')
    _eval(ic, f'setnamed("{ONA_NAME}", "number of input ports", {int(ONA_N_INPUT_PORTS)})')
    _eval(ic, f'setnamed("{ONA_NAME}", "start frequency",       {f_start:.12e})')
    _eval(ic, f'setnamed("{ONA_NAME}", "stop frequency",        {f_stop:.12e})')
    _eval(ic, f'setnamed("{ONA_NAME}", "number of points",      {int(ONA_N_POINTS)})')
    _eval(ic, f'setnamed("{ONA_NAME}", "power",                 {pwr_W:.12e})')
    log.info(f"  {ONA_NAME} added — {ONA_N_INPUT_PORTS} input ports configured.")

    # ── Rings ─────────────────────────────────────────────────────────────────
    for i in range(N_RINGS):
        rn = ring_name(i + 1)
        _eval(ic, 'addelement("Double Bus Ring Resonator")')
        _eval(ic, f'set("name", "{rn}")')
        _try_eval(ic, f'set("x position", {float((i + 1) * 220)})')
        _try_eval(ic, f'set("y position", 0)')
        _apply_ring_params(ic, ring_idx=i)
        log.info(
            f"  RING_{i+1:2d} added  "
            f"[unidir, neff={RING_NEFF_TE[i]:.6f}, "
            f"L={RING_RADIUS_M[i]*2*math.pi*1e6:.4f} µm]"
        )

    # ── NO OPM elements — drops go directly to ONA inputs ────────────────────

    # ── Connections ───────────────────────────────────────────────────────────
    def wire(elem_a: str, port_a: str, elem_b: str, port_b: str) -> None:
        _eval(ic, f'connect("{elem_a}", "{port_a}", "{elem_b}", "{port_b}")')

    # ONA output → RING_1 input
    wire(ONA_NAME, "output", ring_name(1), "input")

    # RING_1 through → ONA input 1  (sensor through)
    wire(ring_name(1), "output 1", ONA_NAME, f"input {THROUGH_SENSOR_INPUT}")

    # RING_1 drop → RING_2 input  (cascade entry)
    wire(ring_name(1), "output 2", ring_name(2), "input")

    # RING_2 .. RING_13: through → next ring,  drop → ONA input n
    for i in range(2, N_RINGS):      # i = 2, 3, ..., 13
        wire(ring_name(i), "output 1", ring_name(i + 1), "input")
        wire(ring_name(i), "output 2", ONA_NAME, f"input {drop_ona_input(i)}")
        log.info(f"  RING_{i} drop → ONA input {drop_ona_input(i)}")

    # RING_14 drop → ONA input 14
    wire(ring_name(N_RINGS), "output 2", ONA_NAME, f"input {drop_ona_input(N_RINGS)}")
    log.info(f"  RING_14 drop → ONA input {drop_ona_input(N_RINGS)}")

    # RING_14 through → ONA input 15  (final through)
    wire(ring_name(N_RINGS), "output 1", ONA_NAME, f"input {THROUGH_FINAL_INPUT}")
    log.info(f"  RING_14 through → ONA input {THROUGH_FINAL_INPUT}")

    log.info(
        f"V3 circuit built: {N_RINGS} rings, {N_DROPS} drop ports → ONA inputs 2-14, "
        f"no OPM elements."
    )


def _extract_results(ic) -> tuple:
    """
    Extract all ONA transmission spectra after run().

    ONA PORT MAP (V3)
    ─────────────────
    input  1 → RING_1  through   → T_sensor_through_dB   (n_wl,)
    input  k → RING_k  drop      → T_drop_dB[k-2, :]     k=2..14  (13, n_wl)
    input 15 → RING_14 through   → T_final_through_dB    (n_wl,)

    INTEGRATED DETECTOR POWER
    ─────────────────────────
    For each drop port k (ring 2..14):

        P_det[k] [W] = P_source [W] × mean(|T_drop_k(λ)|²)

    where mean is taken over the full ONA wavelength band.
    This replicates a broadband photodetector integrating all arriving power.
    The result is reported in dBm.

    Returns
    -------
    wl_m                  : (n_wl,)       wavelengths [m], ascending
    T_sensor_through_dB   : (n_wl,)       RING_1 through spectrum [dB]
    T_final_through_dB    : (n_wl,)       RING_14 through spectrum [dB]
    T_drop_dB             : (N_DROPS, n_wl)  drop spectra for rings 2..14 [dB]
    drop_power_dBm        : (N_DROPS,)    integrated detector power per drop [dBm]
    """
    # ── Frequency / wavelength axis from ONA input 1 ─────────────────────────
    raw_ref = ic.getresult(ONA_NAME, f"input {THROUGH_SENSOR_INPUT}/mode 1/transmission")
    f_arr   = np.asarray(raw_ref["frequency"]).flatten()
    sort_i  = np.argsort(f_arr)[::-1]          # descending f → ascending λ
    wl_m    = SPEED_OF_LIGHT / f_arr[sort_i]
    n_wl    = len(wl_m)

    # Source power in Watts (needed for absolute power calculation)
    p_source_W = 10.0 ** (ONA_POWER_DBM / 10.0) * 1e-3

    # ── Helper: read one ONA input, return T [linear] sorted ascending in λ ──
    def _read_T_linear(port_label: str) -> np.ndarray:
        raw = ic.getresult(ONA_NAME, f"{port_label}/mode 1/transmission")
        T   = np.asarray(raw["TE transmission"]).flatten()[sort_i]
        return np.abs(T)

    def _T_to_dB(T_lin: np.ndarray) -> np.ndarray:
        return 10.0 * np.log10(np.where(T_lin > 0, T_lin, 1e-30))

    # ── ONA input 1: sensor through ───────────────────────────────────────────
    T_sensor_lin        = _read_T_linear(f"input {THROUGH_SENSOR_INPUT}")
    T_sensor_through_dB = _T_to_dB(T_sensor_lin)

    # ── ONA input 15: cascade final through ───────────────────────────────────
    T_final_lin          = _read_T_linear(f"input {THROUGH_FINAL_INPUT}")
    T_final_through_dB   = _T_to_dB(T_final_lin)

    # ── ONA inputs 2..14: spectrometer drops ─────────────────────────────────
    T_drop_dB      = np.full((N_DROPS, n_wl), np.nan)
    drop_power_dBm = np.full(N_DROPS, np.nan)

    for k in range(2, N_RINGS + 1):          # k = 2..14
        drop_idx = k - 2                      # 0-based index into N_DROPS arrays
        ona_port = drop_ona_input(k)          # ONA input number (= k)
        try:
            T_lin = _read_T_linear(f"input {ona_port}")
            T_drop_dB[drop_idx, :] = _T_to_dB(T_lin)

            # ── Integrated detector power ─────────────────────────────────────
            # mean(|T|²) over the ONA band = fraction of source power reaching
            # the detector.  For a broadband (incoherent) flat-spectrum source
            # this is the physical power seen by a wideband photodetector.
            # |T|² is already the power transmittance (ONA returns amplitude
            # transmission, so |T_lin|² is the power ratio).
            mean_T_sq  = float(np.mean(T_lin ** 2))
            p_det_W    = p_source_W * mean_T_sq
            drop_power_dBm[drop_idx] = 10.0 * np.log10(max(p_det_W, 1e-40) * 1e3)

        except Exception as exc:
            log.warning(f"  Drop extraction failed RING_{k} (ONA input {ona_port}): {exc}")

    return wl_m, T_sensor_through_dB, T_final_through_dB, T_drop_dB, drop_power_dBm


def _init_hdf5(wl_ref_m: np.ndarray) -> None:
    n_pts = SWEEP_N_POINTS
    n_wl  = len(wl_ref_m)
    with h5py.File(HDF5_PATH, "w") as f:
        md = f.create_group("metadata")
        md.create_dataset("neff_sweep",    data=SWEEP_NEFF)
        md.create_dataset("ng_sweep",      data=SWEEP_NG)
        md.create_dataset("wavelengths_m", data=wl_ref_m)
        md.attrs["version_name"]          = VERSION_NAME
        md.attrs["n_rings"]               = N_RINGS
        md.attrs["n_drops"]               = N_DROPS
        md.attrs["drop_layout"]           = "drop_k monitors RING_(k+2), k=0..12 (0-based)"
        md.attrs["sweep_n_points"]        = SWEEP_N_POINTS
        md.attrs["ring_model"]            = "Double Bus Ring Resonator"
        md.attrs["ring_configuration"]    = "unidirectional"
        md.attrs["topology"]              = (
            "V3: ONA multiport, no OPMs. "
            "ONA input 1=RING_1 through, input k=RING_k drop (k=2-14), "
            "input 15=RING_14 through"
        )
        md.attrs["ona_lambda_start_m"]    = ONA_LAMBDA_START_M
        md.attrs["ona_lambda_stop_m"]     = ONA_LAMBDA_STOP_M
        md.attrs["ona_n_points"]          = ONA_N_POINTS
        md.attrs["ona_power_dBm"]         = ONA_POWER_DBM
        md.attrs["ona_n_input_ports"]     = ONA_N_INPUT_PORTS
        md.attrs["power_calc_method"]     = (
            "P_det[W] = P_source[W] * mean(|T_drop(lambda)|^2) over ONA band"
        )
        md.attrs["timestamp_start"]       = datetime.now().isoformat()

        for i in range(N_RINGS):
            p = f"ring{i+1}_"
            md.attrs[p + "radius_m"]        = RING_RADIUS_M[i]
            md.attrs[p + "circumference_m"] = RING_RADIUS_M[i] * 2.0 * math.pi
            md.attrs[p + "lambda_res_m"]    = RING_LAMBDA_RES_M[i]
            md.attrs[p + "neff_TE"]         = RING_NEFF_TE[i]
            md.attrs[p + "ng_TE"]           = RING_NG_TE[i]
            md.attrs[p + "kappa_input_sq"]  = RING_KAPPA_INPUT_SQ[i]
            md.attrs[p + "kappa_drop_sq"]   = RING_KAPPA_DROP_SQ[i]
            md.attrs[p + "loss_dB_per_m"]   = RING_LOSS_DB_PER_M[i]

        for k in range(2, N_RINGS + 1):
            md.attrs[f"drop{k-1}_ring"]    = f"RING_{k} output 2 (drop) → ONA input {k}"

        rg = f.create_group("results")
        # Sensor through and final through spectra
        rg.create_dataset("T_sensor_through_dB",
                          data=np.full((n_pts, n_wl), np.nan), chunks=(1, n_wl))
        rg.create_dataset("T_final_through_dB",
                          data=np.full((n_pts, n_wl), np.nan), chunks=(1, n_wl))
        # Drop spectra: (sweep_pts, N_DROPS, n_wl)
        rg.create_dataset("T_drop_dB",
                          data=np.full((n_pts, N_DROPS, n_wl), np.nan),
                          chunks=(1, 1, n_wl))
        # Integrated detector power per drop: (sweep_pts, N_DROPS)
        rg.create_dataset("drop_power_dBm",
                          data=np.full((n_pts, N_DROPS), np.nan),
                          chunks=(1, N_DROPS))

        f.create_group("flags").create_dataset(
            "computed", data=np.zeros(n_pts, dtype=bool), chunks=(1,))

    log.info(f"HDF5 V3 initialised ({N_DROPS} drops, {n_wl} wavelengths) → {HDF5_PATH}")


def run_interconnect_sweep(hide_gui: bool = False) -> Dict[str, Any]:
    n_pts = SWEEP_N_POINTS
    wavelengths_m         = None
    T_sensor_through_dB   = None
    T_final_through_dB    = None
    T_drop_dB             = None
    drop_power_dBm        = None
    computed   = np.zeros(n_pts, dtype=bool)
    hdf5_ready = False

    # ── Resume from cache ─────────────────────────────────────────────────────
    if HDF5_PATH.exists():
        log.info(f"Cache found → {HDF5_PATH}")
        try:
            with h5py.File(HDF5_PATH, "r") as f:
                wavelengths_m       = f["metadata/wavelengths_m"][:]
                T_sensor_through_dB = f["results/T_sensor_through_dB"][:]
                T_final_through_dB  = f["results/T_final_through_dB"][:]
                T_drop_dB           = f["results/T_drop_dB"][:]
                drop_power_dBm      = f["results/drop_power_dBm"][:]
                computed[:]         = f["flags/computed"][:]
            hdf5_ready = True
            n_cached   = int(computed.sum())
            log.info(f"Cached: {n_cached}/{n_pts}  |  Remaining: {n_pts - n_cached}")
            if n_pts - n_cached == 0:
                log.info("All sweep points cached — skipping INTERCONNECT launch.")
                return _pack_results(wavelengths_m, T_sensor_through_dB,
                                     T_final_through_dB, T_drop_dB,
                                     drop_power_dBm, computed)
        except Exception as exc:
            log.warning(f"Cache unreadable ({exc}). Starting fresh.")
            wavelengths_m = T_sensor_through_dB = T_final_through_dB = None
            T_drop_dB = drop_power_dBm = None
            computed[:] = False
            hdf5_ready  = False
    else:
        log.info("No cache — starting fresh sweep.")

    log.info("Launching INTERCONNECT …")
    ic         = lumapi.INTERCONNECT(hide=hide_gui)
    runs_done  = 0
    runs_total = int((~computed).sum())
    t_start    = time.time()

    try:
        _build_circuit(ic)
        log.info(f"Circuit ready — {runs_total} sweep points to compute …")

        for s_idx in range(n_pts):
            if computed[s_idx]:
                continue

            neff_val = float(SWEEP_NEFF[s_idx])
            ng_val   = float(SWEEP_NG[s_idx])

            _eval(ic, "switchtodesign")
            _update_ring1_neff_ng(ic, neff_val, ng_val)

            # ── Run ───────────────────────────────────────────────────────────
            try:
                _eval(ic, "run")
            except ICScriptError as exc:
                log.warning(
                    f"  RUN FAILED  pt={s_idx:3d}  "
                    f"neff={neff_val:.6f}  ng={ng_val:.6f}  →  {exc}"
                )
                computed[s_idx] = True
                if hdf5_ready:
                    with h5py.File(HDF5_PATH, "r+") as hf:
                        hf["flags/computed"][s_idx] = True
                        hf.flush()
                continue

            # ── Extract ───────────────────────────────────────────────────────
            try:
                wl_m, t_sen, t_fin, t_drop, p_drop = _extract_results(ic)
            except Exception as exc:
                log.warning(f"  EXTRACT FAILED  pt={s_idx:3d}: {exc}")
                computed[s_idx] = True
                continue

            # ── Initialise arrays on first valid point ────────────────────────
            if wavelengths_m is None:
                n_wl                = len(wl_m)
                wavelengths_m       = wl_m
                T_sensor_through_dB = np.full((n_pts, n_wl),          np.nan)
                T_final_through_dB  = np.full((n_pts, n_wl),          np.nan)
                T_drop_dB           = np.full((n_pts, N_DROPS, n_wl), np.nan)
                drop_power_dBm      = np.full((n_pts, N_DROPS),        np.nan)
                if not hdf5_ready:
                    _init_hdf5(wl_m)
                    hdf5_ready = True

            # ── Store in memory ───────────────────────────────────────────────
            T_sensor_through_dB[s_idx, :]    = t_sen
            T_final_through_dB [s_idx, :]    = t_fin
            T_drop_dB          [s_idx, :, :] = t_drop
            drop_power_dBm     [s_idx, :]    = p_drop
            computed           [s_idx]        = True

            # ── Flush to HDF5 ─────────────────────────────────────────────────
            with h5py.File(HDF5_PATH, "r+") as hf:
                hf["results/T_sensor_through_dB"][s_idx, :]    = t_sen
                hf["results/T_final_through_dB"] [s_idx, :]    = t_fin
                hf["results/T_drop_dB"]          [s_idx, :, :] = t_drop
                hf["results/drop_power_dBm"]     [s_idx, :]    = p_drop
                hf["flags/computed"]             [s_idx]        = True
                hf.flush()

            runs_done += 1
            if runs_done % 5 == 0 or runs_done == runs_total:
                elapsed = time.time() - t_start
                rate    = runs_done / elapsed if elapsed > 0 else 1e-9
                eta     = (runs_total - runs_done) / rate
                log.info(
                    f"  [{runs_done:3d}/{runs_total}]  "
                    f"neff={neff_val:.6f}  ng={ng_val:.6f}  │  "
                    f"{rate:.2f} sim/s  │  ETA {eta:5.0f} s"
                )

        if hdf5_ready:
            with h5py.File(HDF5_PATH, "r+") as hf:
                hf["metadata"].attrs["timestamp_end"]  = datetime.now().isoformat()
                hf["metadata"].attrs["runs_completed"] = int(computed.sum())

    finally:
        try:
            ic.close()
        except Exception:
            pass
        log.info("INTERCONNECT session closed.")

    elapsed = time.time() - t_start
    log.info(
        f"Sweep done │ {runs_done} new runs │ "
        f"total={elapsed:.1f} s │ avg={elapsed/max(runs_done, 1):.2f} s/sim"
    )
    return _pack_results(wavelengths_m, T_sensor_through_dB,
                         T_final_through_dB, T_drop_dB, drop_power_dBm, computed)


def _pack_results(wl, t_sen, t_fin, t_drop, p_drop, comp) -> Dict[str, Any]:
    return dict(
        neff_sweep            = SWEEP_NEFF,
        ng_sweep              = SWEEP_NG,
        wavelengths_m         = wl,
        T_sensor_through_dB   = t_sen,
        T_final_through_dB    = t_fin,
        T_drop_dB             = t_drop,
        drop_power_dBm        = p_drop,
        computed              = comp,
    )


def load_results(path: Path = HDF5_PATH) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with h5py.File(path, "r") as f:
            if "metadata/wavelengths_m" not in f:
                return None
            wl = f["metadata/wavelengths_m"][:]
            if wl is None or len(wl) == 0:
                return None
            return dict(
                neff_sweep            = f["metadata/neff_sweep"][:],
                ng_sweep              = f["metadata/ng_sweep"][:],
                wavelengths_m         = wl,
                T_sensor_through_dB   = f["results/T_sensor_through_dB"][:],
                T_final_through_dB    = f["results/T_final_through_dB"][:],
                T_drop_dB             = f["results/T_drop_dB"][:],
                drop_power_dBm        = f["results/drop_power_dBm"][:],
                computed              = f["flags/computed"][:],
            )
    except Exception as exc:
        log.warning(f"Could not read HDF5 ({exc}): {path}")
        return None


def get_results(path: Path = HDF5_PATH) -> Dict[str, Any]:
    mem_reason = ""
    try:
        r  = sweep_results
        wl = r.get("wavelengths_m")
        if wl is not None and len(wl) > 0:
            return r
        mem_reason = "sweep_results in memory but wavelengths_m is None"
    except NameError:
        mem_reason = "sweep_results not defined (Cell 3 not run)"
    r_hdf5 = load_results(path)
    if r_hdf5 is not None:
        log.info(f"Loaded from HDF5: {int(r_hdf5['computed'].sum())}/{SWEEP_N_POINTS} pts.")
        return r_hdf5
    hdf5_reason = (
        f"HDF5 exists but empty — delete and re-run Cell 3:\n    {path}"
        if path.exists() else f"HDF5 not found:\n    {path}"
    )
    raise RuntimeError(
        f"\n{'='*65}\n  No results available.\n"
        f"  Memory : {mem_reason}\n  HDF5   : {hdf5_reason}\n"
        f"  ► Run Cell 3.\n{'='*65}"
    )


def _valid_mask(r: Dict) -> np.ndarray:
    return r["computed"].astype(bool)


def plot_drop_power_vs_neff(
    results=None, figsize=(11, 6), save: bool = True,
) -> plt.Figure:
    """
    Power vs neff — primary sensor readout plot.

    Each curve is the integrated detector power [dBm] at the drop port
    of one spectrometer ring as the sensor ring (RING_1) sweeps its neff.

    Physical meaning:
      As RING_1 neff increases, its resonance shifts to longer wavelengths.
      Each spectrometer ring has a fixed resonance at its design wavelength.
      When the sensor resonance approaches/crosses a spectrometer resonance,
      the drop power of that spectrometer ring changes — its unique response
      to the sensor shift.  Reading all 13 drops gives a power vector that
      encodes the sensor state with high redundancy.

    P_det [dBm] = 10 log10(P_source [mW] × mean_λ(|T_drop(λ)|²))
    """
    if results is None:
        results = get_results()
    neff_arr  = results["neff_sweep"]
    p_drop    = results["drop_power_dBm"]   # (n_pts, N_DROPS)
    mask      = _valid_mask(results)
    neff_v    = neff_arr[mask]
    p_v       = p_drop[mask, :]             # (n_valid, N_DROPS)

    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=figsize)

    for k in range(N_DROPS):
        ring_label = DROP_LABELS[k]
        ax.plot(
            neff_v, p_v[:, k],
            color=cmap(k / N_DROPS), lw=1.5,
            marker="o", ms=2.5, alpha=0.85,
            label=ring_label,
        )

    ax.set_xlabel("Ring 1 — $n_{eff}$ (TE)  [sensor]", fontsize=12)
    ax.set_ylabel("Detector power  (dBm)", fontsize=12)
    ax.set_title(
        "Integrated Drop Power vs Sensor $n_{eff}$  [V3 — ONA multiport]\n"
        r"$P_{det} = P_{src} \cdot \langle|T_{drop}(\lambda)|^2\rangle_\lambda$"
        "   —   13 spectrometer rings",
        fontsize=12,
    )
    ax.legend(ncol=3, framealpha=0.88, fontsize=8, loc="best")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig.tight_layout()

    if save:
        fig.savefig(FIGURES_DIR / "drop_power_vs_neff_all.png")
        fig.savefig(FIGURES_DIR / "drop_power_vs_neff_all.pdf")
        log.info("Saved → drop_power_vs_neff_all.png/pdf")
    return fig


def plot_sensor_through_sweep(
    results=None, n_curves: int = 200,
    figsize=(10, 5), cmap_name: str = "plasma", save: bool = True,
) -> plt.Figure:
    """
    ONA input 1 (RING_1 through) transmission spectra.
    Colour encodes the sensor neff value for each sweep point.
    """
    if results is None:
        results = get_results()
    neff_arr = results["neff_sweep"]
    wl_nm    = results["wavelengths_m"] * 1e9
    T_data   = results["T_sensor_through_dB"]
    mask     = _valid_mask(results)
    valid_idx = np.where(mask)[0]
    n_sel    = min(n_curves, len(valid_idx))
    sel_idx  = valid_idx[
        np.round(np.linspace(0, len(valid_idx) - 1, n_sel)).astype(int)
    ]
    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=neff_arr[sel_idx].min(), vmax=neff_arr[sel_idx].max())
    sm   = ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])

    fig, ax = plt.subplots(figsize=figsize)
    for idx in sel_idx:
        ax.plot(wl_nm, T_data[idx], color=cmap(norm(neff_arr[idx])),
                lw=0.8, alpha=0.70)
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Ring 1 — $n_{eff}$ (TE)", fontsize=10)
    ax.set_xlabel("Wavelength  (nm)")
    ax.set_ylabel("Transmission  (dB)")
    ax.set_title(
        f"Sensor Through Spectrum — ONA input 1  (RING_1 through)\n"
        f"({n_sel} curves,  colour = $n_{{eff,1}}$)"
    )
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig.tight_layout()
    if save:
        stem = f"sensor_through_sweep_{n_sel}curves"
        fig.savefig(FIGURES_DIR / f"{stem}.png")
        fig.savefig(FIGURES_DIR / f"{stem}.pdf")
        log.info(f"Saved → {stem}.png/pdf")
    return fig


def plot_final_through_sweep(
    results=None, n_curves: int = 200,
    figsize=(10, 5), cmap_name: str = "viridis", save: bool = True,
) -> plt.Figure:
    """
    ONA input 15 (RING_14 through) transmission spectra — cascade output.
    """
    if results is None:
        results = get_results()
    neff_arr  = results["neff_sweep"]
    wl_nm     = results["wavelengths_m"] * 1e9
    T_data    = results["T_final_through_dB"]
    mask      = _valid_mask(results)
    valid_idx = np.where(mask)[0]
    n_sel     = min(n_curves, len(valid_idx))
    sel_idx   = valid_idx[
        np.round(np.linspace(0, len(valid_idx) - 1, n_sel)).astype(int)
    ]
    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=neff_arr[sel_idx].min(), vmax=neff_arr[sel_idx].max())
    sm   = ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])

    fig, ax = plt.subplots(figsize=figsize)
    for idx in sel_idx:
        ax.plot(wl_nm, T_data[idx], color=cmap(norm(neff_arr[idx])),
                lw=0.8, alpha=0.70)
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Ring 1 — $n_{eff}$ (TE)", fontsize=10)
    ax.set_xlabel("Wavelength  (nm)")
    ax.set_ylabel("Transmission  (dB)")
    ax.set_title(
        f"Cascade Final Through — ONA input 15  (RING_14 through)\n"
        f"({n_sel} curves,  colour = $n_{{eff,1}}$)"
    )
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig.tight_layout()
    if save:
        stem = f"final_through_sweep_{n_sel}curves"
        fig.savefig(FIGURES_DIR / f"{stem}.png")
        fig.savefig(FIGURES_DIR / f"{stem}.pdf")
        log.info(f"Saved → {stem}.png/pdf")
    return fig


def plot_drop_spectrum_heatmap(
    results=None, drop_k: int = 1,
    figsize=(10, 3.5), cmap_name: str = "inferno",
    vmin_dB=None, vmax_dB=None, save: bool = True,
) -> plt.Figure:
    """
    Heatmap: neff (y) × wavelength (x) of drop transmission [dB].
    drop_k is 1-based (1..13):
      drop_k=1  → RING_2 drop  (ONA input 2)
      drop_k=13 → RING_14 drop (ONA input 14)
    """
    if results is None:
        results = get_results()
    assert 1 <= drop_k <= N_DROPS, f"drop_k must be 1..{N_DROPS}"
    ring_label = DROP_LABELS[drop_k - 1]

    neff_arr = results["neff_sweep"]
    wl_nm    = results["wavelengths_m"] * 1e9
    t_drop   = results["T_drop_dB"]            # (n_pts, N_DROPS, n_wl)
    mask     = _valid_mask(results)
    neff_v   = neff_arr[mask]
    spec_v   = t_drop[mask, drop_k - 1, :]     # (n_valid, n_wl)

    _vmin = vmin_dB if vmin_dB is not None else np.nanpercentile(spec_v, 2)
    _vmax = vmax_dB if vmax_dB is not None else np.nanpercentile(spec_v, 98)

    fig, ax = plt.subplots(figsize=figsize)
    img = ax.pcolormesh(wl_nm, neff_v, spec_v,
                        cmap=cmap_name, vmin=_vmin, vmax=_vmax, shading="auto")
    cbar = fig.colorbar(img, ax=ax, pad=0.01)
    cbar.set_label("Transmission  (dB)", fontsize=10)
    ax.set_xlabel("Wavelength  (nm)")
    ax.set_ylabel("Ring 1 — $n_{eff}$ (TE)")
    ax.set_title(
        f"Drop Spectrum Heatmap — {ring_label}  (ONA input {drop_k + 1})\n"
        f"neff sweep on sensor ring (RING_1)"
    )
    fig.tight_layout()
    if save:
        fname = f"drop{drop_k}_spectrum_heatmap"
        fig.savefig(FIGURES_DIR / f"{fname}.png")
        fig.savefig(FIGURES_DIR / f"{fname}.pdf")
        log.info(f"Saved → {fname}.png/pdf")
    return fig


def plot_all_drop_heatmaps(
    results=None, ncols: int = 4,
    cmap_name: str = "inferno", save: bool = True,
) -> plt.Figure:
    if results is None:
        results = get_results()
    neff_arr = results["neff_sweep"]
    wl_nm    = results["wavelengths_m"] * 1e9
    t_drop   = results["T_drop_dB"]
    mask     = _valid_mask(results)
    neff_v   = neff_arr[mask]

    nrows = math.ceil(N_DROPS / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.8, nrows * 2.8),
                             sharex=True, sharey=True)
    axes_flat = axes.flatten()

    all_vals = t_drop[mask].ravel()
    vmin = np.nanpercentile(all_vals, 2)
    vmax = np.nanpercentile(all_vals, 98)

    im = None
    for k in range(1, N_DROPS + 1):
        ax    = axes_flat[k - 1]
        spec  = t_drop[mask, k - 1, :]
        im    = ax.pcolormesh(wl_nm, neff_v, spec,
                              cmap=cmap_name, vmin=vmin, vmax=vmax, shading="auto")
        ax.set_title(f"{DROP_LABELS[k-1]} drop", fontsize=8)
        if k % ncols == 1:
            ax.set_ylabel("$n_{eff,1}$", fontsize=7)
        if k > (nrows - 1) * ncols:
            ax.set_xlabel("λ (nm)", fontsize=7)
        ax.tick_params(labelsize=6)

    for ax in axes_flat[N_DROPS:]:
        ax.set_visible(False)

    if im is not None:
        fig.colorbar(im, ax=axes_flat[:N_DROPS], shrink=0.6, pad=0.02,
                     label="Transmission (dB)", fraction=0.015)
    fig.suptitle(
        "All Spectrometer Drop Spectra  [V3 — ONA multiport]\n"
        "neff sweep on sensor ring (RING_1)",
        fontsize=11,
    )
    fig.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "all_drop_heatmaps_grid.png", dpi=200)
        fig.savefig(FIGURES_DIR / "all_drop_heatmaps_grid.pdf")
        log.info("Saved → all_drop_heatmaps_grid.png/pdf")
    return fig


def plot_power_heatmap_drops_vs_neff(
    results=None, figsize=(10, 5), cmap_name: str = "plasma", save: bool = True,
) -> plt.Figure:
    """
    2-D heatmap: X = neff sweep, Y = spectrometer ring index (1-13),
    colour = integrated detector power [dBm].

    Shows at a glance which ring is "lit up" at each sensor state.
    """
    if results is None:
        results = get_results()
    neff_arr = results["neff_sweep"]
    p_drop   = results["drop_power_dBm"]     # (n_pts, N_DROPS)
    mask     = _valid_mask(results)
    neff_v   = neff_arr[mask]
    p_v      = p_drop[mask, :].T             # (N_DROPS, n_valid)

    fig, ax = plt.subplots(figsize=figsize)
    img = ax.pcolormesh(
        neff_v,
        np.arange(1, N_DROPS + 1),
        p_v,
        cmap=cmap_name, shading="auto",
    )
    cbar = fig.colorbar(img, ax=ax, pad=0.02)
    cbar.set_label("Detector power  (dBm)", fontsize=10)
    ax.set_xlabel("Ring 1 — $n_{eff}$ (TE)  [sensor]", fontsize=11)
    ax.set_ylabel("Spectrometer ring index  (1 = RING_2 … 13 = RING_14)", fontsize=10)
    ax.set_yticks(np.arange(1, N_DROPS + 1))
    ax.set_yticklabels(DROP_LABELS, fontsize=7)
    ax.set_title(
        "Detector Power Heatmap — All Drops vs Sensor $n_{eff}$  [V3]\n"
        r"Colour: $P_{det}$ [dBm]  per spectrometer ring",
        fontsize=12,
    )
    fig.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "power_heatmap_drops_vs_neff.png", dpi=200)
        fig.savefig(FIGURES_DIR / "power_heatmap_drops_vs_neff.pdf")
        log.info("Saved → power_heatmap_drops_vs_neff.png/pdf")
    return fig


def plot_resonance_tracking(
    results=None, figsize=(7, 4.5), save: bool = True,
) -> plt.Figure:
    if results is None:
        results = get_results()
    neff_arr = results["neff_sweep"]
    wl_nm    = results["wavelengths_m"] * 1e9
    T_data   = results["T_sensor_through_dB"]
    mask     = _valid_mask(results)
    neff_v   = neff_arr[mask]
    T_v      = T_data[mask, :]
    dip_idx  = np.argmin(T_v, axis=1)
    lam_dip  = wl_nm[dip_idx]
    coeffs   = np.polyfit(neff_v, lam_dip, 1)
    sens     = coeffs[0]

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(neff_v, lam_dip, s=20, zorder=5, color="#2166ac", label="Resonance dip")
    ax.plot(neff_v, np.poly1d(coeffs)(neff_v), "r--", lw=1.5,
            label=f"Linear fit   ∂λ/∂n = {sens:.3f} nm / RIU")
    ax.set_xlabel("Ring 1 — $n_{eff}$ (TE)")
    ax.set_ylabel("Resonance wavelength  (nm)")
    ax.set_title(
        f"Resonance Tracking — Sensor Through (ONA input 1)\n"
        f"Sensitivity: {sens:.3f} nm / RIU"
    )
    ax.legend(framealpha=0.9)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig.tight_layout()
    if save:
        fname = "resonance_tracking_sensor"
        fig.savefig(FIGURES_DIR / f"{fname}.png")
        fig.savefig(FIGURES_DIR / f"{fname}.pdf")
        log.info(f"Saved → {fname}.png/pdf")
    log.info(f"Resonance sensitivity: {sens:.4f} nm/RIU")
    return fig

def run(state=None):
    """Execute this step. `state` carries the shared namespace
    between steps (the notebook's old kernel globals + bridge).
    Returns the updated `state`."""
    state = {} if state is None else state
    global DROP_LABELS, THROUGH_FINAL_INPUT, THROUGH_SENSOR_INPUT, T_drop_dB, T_final_through_dB, T_sensor_through_dB, _res, computed, drop_power_dBm, fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, neff_sweep, ng_sweep, sweep_results, wavelengths_m, wavelengths_nm
    globals()['lumapi'] = import_lumapi()
    globals().update(state)

    sweep_results = run_interconnect_sweep(hide_gui=False)
    neff_sweep          = sweep_results["neff_sweep"]
    ng_sweep            = sweep_results["ng_sweep"]
    wavelengths_m       = sweep_results["wavelengths_m"]
    T_sensor_through_dB = sweep_results["T_sensor_through_dB"]
    T_final_through_dB  = sweep_results["T_final_through_dB"]
    T_drop_dB           = sweep_results["T_drop_dB"]
    drop_power_dBm      = sweep_results["drop_power_dBm"]
    computed            = sweep_results["computed"]
    wavelengths_nm      = wavelengths_m * 1e9 if wavelengths_m is not None else None
    print(f"\n  Sweep complete — {computed.sum()} / {len(computed)} pts computed")
    if wavelengths_m is not None:
        print(f"  T_sensor_through_dB  shape : {T_sensor_through_dB.shape}")
        print(f"  T_final_through_dB   shape : {T_final_through_dB.shape}")
        print(f"  T_drop_dB            shape : {T_drop_dB.shape}   (RING_2..14 drops)")
        print(f"  drop_power_dBm       shape : {drop_power_dBm.shape}")
    print(f"  HDF5                        : {HDF5_PATH}")
    plt.rcParams.update({
        "font.family"    : "serif",
        "font.serif"     : ["DejaVu Serif", "Georgia", "Times New Roman"],
        "font.size"      : 11, "axes.labelsize": 12, "axes.titlesize": 13,
        "legend.fontsize": 9,  "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.linewidth" : 0.8, "axes.grid": True, "grid.alpha": 0.3,
        "grid.linewidth" : 0.5, "lines.linewidth": 1.4,
        "figure.dpi"     : 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    })
    DROP_LABELS = [f"RING_{k+2}" for k in range(N_DROPS)]   # RING_2 .. RING_14
    try:
        _res = get_results()
    except RuntimeError as exc:
        print(str(exc))
        raise
    else:
        # ── Plot 1: PRIMARY — integrated power vs neff for all 13 drops ──────────
        fig1 = plot_drop_power_vs_neff(_res)

        # ── Plot 2: Sensor through spectra coloured by neff ───────────────────────
        fig2 = plot_sensor_through_sweep(_res, n_curves=200)

        # ── Plot 3: Cascade final through spectra ─────────────────────────────────
        fig3 = plot_final_through_sweep(_res, n_curves=200)

        # ── Plot 4: Drop heatmaps — first and last spectrometer rings ────────────
        fig4 = plot_drop_spectrum_heatmap(_res, drop_k=1)    # RING_2 drop
        fig5 = plot_drop_spectrum_heatmap(_res, drop_k=13)   # RING_14 drop

        # ── Plot 5: All 13 drop heatmaps in a grid ───────────────────────────────
        fig6 = plot_all_drop_heatmaps(_res)

        # ── Plot 6: 2-D power heatmap (rings × neff) ─────────────────────────────
        fig7 = plot_power_heatmap_drops_vs_neff(_res)

        # ── Plot 7: Resonance wavelength tracking ────────────────────────────────
        fig8 = plot_resonance_tracking(_res)

        plt.show()
        print(f"\n  Figures → {FIGURES_DIR}")
        print(f"  HDF5    → {HDF5_PATH}")

    state.update({k: globals().get(k) for k in [
        'DROP_LABELS', 'THROUGH_FINAL_INPUT', 'THROUGH_SENSOR_INPUT', 'T_drop_dB', 'T_final_through_dB', 'T_sensor_through_dB',
        '_res', 'computed', 'drop_power_dBm', 'fig1', 'fig2', 'fig3',
        'fig4', 'fig5', 'fig6', 'fig7', 'fig8', 'neff_sweep',
        'ng_sweep', 'sweep_results', 'wavelengths_m', 'wavelengths_nm',
    ] if k in globals()})
    return state


if __name__ == "__main__":
    run()
