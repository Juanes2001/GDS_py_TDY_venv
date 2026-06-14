"""
step5_two_ring_radius_sweep.py — 2-ring radius sweep (sensor + one spectrometer) with its own HDF5 cache.

A focused 2-ring INTERCONNECT experiment that sweeps the sensor-ring
radius to study FSR/alignment sensitivity, with an independent
circuit builder, sweep engine and cache so it can run standalone.
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
from .step1_cascade_sweep import (
    ICScriptError, _eval, _try_eval,
)

from .config import *  # shared platform constants & paths
from .lumerical_session import import_lumapi
from .plotting import apply_style, save_fig, make_colorbar
from . import plotting as _plot

apply_style()

# ─────────────────────────────────────────────────────────
#  Module constants / design knobs for this step
# ─────────────────────────────────────────────────────────
RS_IDX = 0   # posición en los arrays RING_* para el sensor
RE_IDX = 1   # posición en los arrays RING_* para el espectrómetro

def _build_2ring_circuit(ic) -> None:
    """
    Construye el circuito de 2 anillos en INTERCONNECT.

    TOPOLOGÍA
    ──────────────────────────────────────────────────────
    ONA_2ring  output  → RS_SENSOR  input
    RS_SENSOR  out1    → ONA_2ring  input 1   (sensor through)
    RS_SENSOR  out2    → RE_SPEC    input     (sensor drop → espectrómetro)
    RE_SPEC    out2    → ONA_2ring  input 2   (espectrómetro DROP ← medición)
    RE_SPEC    out1    → ONA_2ring  input 3   (espectrómetro through)
    """
    _eval(ic, "switchtodesign")
    _try_eval(ic, "selectall")
    _try_eval(ic, "delete")

    pwr_W   = 10.0 ** (ONA_POWER_DBM / 10.0) * 1e-3
    f_start = SPEED_OF_LIGHT / ONA_LAMBDA_STOP_M
    f_stop  = SPEED_OF_LIGHT / ONA_LAMBDA_START_M

    # ── ONA ──────────────────────────────────────────────────────────────────
    _eval(ic, 'addelement("Optical Network Analyzer")')
    _eval(ic, f'set("name", "{ONA2_NAME}")')
    _try_eval(ic, 'set("x position", 0)')
    _try_eval(ic, 'set("y position", 0)')
    _eval(ic, f'setnamed("{ONA2_NAME}", "input parameter",       "start and stop")')
    _eval(ic, f'setnamed("{ONA2_NAME}", "number of input ports", {ONA2_N_INPUTS})')
    _eval(ic, f'setnamed("{ONA2_NAME}", "start frequency",       {f_start:.12e})')
    _eval(ic, f'setnamed("{ONA2_NAME}", "stop frequency",        {f_stop:.12e})')
    _eval(ic, f'setnamed("{ONA2_NAME}", "number of points",      {ONA_N_POINTS})')
    _eval(ic, f'setnamed("{ONA2_NAME}", "power",                 {pwr_W:.12e})')

    # ── Anillo sensor RS (parámetros fijos, índice RS_IDX=0) ─────────────────
    _eval(ic, 'addelement("Double Bus Ring Resonator")')
    _eval(ic, f'set("name", "{RS_NAME}")')
    _try_eval(ic, 'set("x position", 220)')
    _try_eval(ic, 'set("y position", 0)')
    # Aplica todos los parámetros del índice RS_IDX usando la función existente
    # pero apuntando al nombre RS_NAME (no ring_name(RS_IDX+1))
    _eval(ic, f'setnamed("{RS_NAME}", "length",                   '
              f'{RING_RADIUS_M[RS_IDX]*2.0*math.pi:.12e})')
    _eval(ic, f'setnamed("{RS_NAME}", "frequency",                '
              f'{SPEED_OF_LIGHT/RING_LAMBDA_RES_M[RS_IDX]:.12e})')
    _eval(ic, f'setnamed("{RS_NAME}", "effective index 1",        '
              f'{RING_NEFF_TE[RS_IDX]:.12f})')
    _eval(ic, f'setnamed("{RS_NAME}", "group index 1",            '
              f'{RING_NG_TE[RS_IDX]:.12f})')
    _eval(ic, f'setnamed("{RS_NAME}", "loss 1",                   '
              f'{RING_LOSS_DB_PER_M[RS_IDX]:.6f})')
    _eval(ic, f'setnamed("{RS_NAME}", "dispersion 1",             '
              f'{RING_D_TE_PS2_PER_KM[RS_IDX]*1e-15:.12e})')
    _eval(ic, f'setnamed("{RS_NAME}", "coupling coefficient 1 1", '
              f'{RING_KAPPA_INPUT_SQ[RS_IDX]:.12f})')
    _eval(ic, f'setnamed("{RS_NAME}", "coupling coefficient 1 2", '
              f'{RING_KAPPA_DROP_SQ[RS_IDX]:.12f})')
    _eval(ic, f'setnamed("{RS_NAME}", "configuration", "unidirectional")')
    log.info(f"  {RS_NAME} added  [sensor, R={RING_RADIUS_M[RS_IDX]*1e6:.4f} µm, FIJO]")

    # ── Anillo espectrómetro RE (radio inicial = RING_RADIUS_M[RE_IDX]) ──────
    _eval(ic, 'addelement("Double Bus Ring Resonator")')
    _eval(ic, f'set("name", "{RE_NAME}")')
    _try_eval(ic, 'set("x position", 440)')
    _try_eval(ic, 'set("y position", 0)')
    # Radio inicial: SWEEP_RADIUS_E_M[0] = RING_RADIUS_M[RE_IDX]
    _eval(ic, f'setnamed("{RE_NAME}", "length",                   '
              f'{SWEEP_CIRCUM_E_M[0]:.12e})')
    _eval(ic, f'setnamed("{RE_NAME}", "frequency",                '
              f'{SPEED_OF_LIGHT/RING_LAMBDA_RES_M[RE_IDX]:.12e})')
    _eval(ic, f'setnamed("{RE_NAME}", "effective index 1",        '
              f'{RING_NEFF_TE[RE_IDX]:.12f})')
    _eval(ic, f'setnamed("{RE_NAME}", "group index 1",            '
              f'{RING_NG_TE[RE_IDX]:.12f})')
    _eval(ic, f'setnamed("{RE_NAME}", "loss 1",                   '
              f'{RING_LOSS_DB_PER_M[RE_IDX]:.6f})')
    _eval(ic, f'setnamed("{RE_NAME}", "dispersion 1",             '
              f'{RING_D_TE_PS2_PER_KM[RE_IDX]*1e-15:.12e})')
    _eval(ic, f'setnamed("{RE_NAME}", "coupling coefficient 1 1", '
              f'{RING_KAPPA_INPUT_SQ[RE_IDX]:.12f})')
    _eval(ic, f'setnamed("{RE_NAME}", "coupling coefficient 1 2", '
              f'{RING_KAPPA_DROP_SQ[RE_IDX]:.12f})')
    _eval(ic, f'setnamed("{RE_NAME}", "configuration", "unidirectional")')
    log.info(f"  {RE_NAME} added  [espectrómetro, R_init={SWEEP_RADIUS_E_M[0]*1e6:.4f} µm]")

    # ── Conexiones ────────────────────────────────────────────────────────────
    def wire(a, pa, b, pb):
        _eval(ic, f'connect("{a}", "{pa}", "{b}", "{pb}")')

    wire(ONA2_NAME, "output",   RS_NAME,   "input")
    wire(RS_NAME,   "output 1", ONA2_NAME, f"input {ONA2_THROUGH_SENSOR}")
    wire(RS_NAME,   "output 2", RE_NAME,   "input")
    wire(RE_NAME,   "output 2", ONA2_NAME, f"input {ONA2_DROP_SPEC}")
    wire(RE_NAME,   "output 1", ONA2_NAME, f"input {ONA2_THROUGH_SPEC}")

    log.info(
        f"  2-ring circuit built: {RS_NAME} → {RE_NAME} → ONA_2ring  "
        f"(inputs: 1=sensor_thr, 2=spec_drop, 3=spec_thr)"
    )


def _update_re_radius(ic, circum_m: float) -> None:
    """Actualiza solo la circunferencia (length) de RE_SPEC en cada punto."""
    _eval(ic, f'setnamed("{RE_NAME}", "length", {circum_m:.12e})')


def _extract_2ring_results(ic) -> tuple:
    """
    Extrae espectros ONA_2ring después de run().

    Retorna
    -------
    wl_m              : (n_wl,)   longitudes de onda [m], ascendente
    T_sensor_thr_dB   : (n_wl,)   sensor through [dB]
    T_spec_drop_dB    : (n_wl,)   espectrómetro drop [dB]  ← MEDICIÓN
    T_spec_thr_dB     : (n_wl,)   espectrómetro through [dB]
    drop_power_dBm    : float      potencia integrada drop [dBm]
    """
    # Eje de frecuencias desde input 1
    raw_ref = ic.getresult(ONA2_NAME,
                           f"input {ONA2_THROUGH_SENSOR}/mode 1/transmission")
    f_arr  = np.asarray(raw_ref["frequency"]).flatten()
    sort_i = np.argsort(f_arr)[::-1]
    wl_m   = SPEED_OF_LIGHT / f_arr[sort_i]

    p_source_W = 10.0 ** (ONA_POWER_DBM / 10.0) * 1e-3

    def _T_lin(port_n: int) -> np.ndarray:
        raw = ic.getresult(ONA2_NAME,
                           f"input {port_n}/mode 1/transmission")
        return np.abs(np.asarray(raw["TE transmission"]).flatten()[sort_i])

    def _to_dB(T):
        return 10.0 * np.log10(np.where(T > 0, T, 1e-30))

    T_s_lin  = _T_lin(ONA2_THROUGH_SENSOR)
    T_ed_lin = _T_lin(ONA2_DROP_SPEC)
    T_et_lin = _T_lin(ONA2_THROUGH_SPEC)

    # Potencia integrada en el drop del espectrómetro
    mean_T2    = float(np.mean(T_ed_lin ** 2))
    p_drop_W   = p_source_W * mean_T2
    drop_p_dBm = 10.0 * np.log10(max(p_drop_W, 1e-40) * 1e3)

    return (wl_m,
            _to_dB(T_s_lin),
            _to_dB(T_ed_lin),
            _to_dB(T_et_lin),
            drop_p_dBm)


def _init_hdf5_2ring(wl_ref_m: np.ndarray) -> None:
    n_pts = RADIUS_SWEEP_N
    n_wl  = len(wl_ref_m)
    with h5py.File(HDF5_2RING_PATH, "w") as f:
        md = f.create_group("metadata")
        md.create_dataset("radius_sweep_m",   data=SWEEP_RADIUS_E_M)
        md.create_dataset("circum_sweep_m",   data=SWEEP_CIRCUM_E_M)
        md.create_dataset("wavelengths_m",    data=wl_ref_m)
        md.attrs["version_name"]           = VERSION_2RING
        md.attrs["sweep_variable"]         = "RING_E radius (circumference)"
        md.attrs["sweep_n_points"]         = RADIUS_SWEEP_N
        md.attrs["radius_start_m"]         = RADIUS_SWEEP_START_M
        md.attrs["radius_stop_m"]          = RADIUS_SWEEP_STOP_M
        md.attrs["sensor_ring_radius_m"]   = RING_RADIUS_M[RS_IDX]
        md.attrs["sensor_ring_neff"]       = RING_NEFF_TE[RS_IDX]
        md.attrs["sensor_ring_ng"]         = RING_NG_TE[RS_IDX]
        md.attrs["sensor_ring_lambda_res"] = RING_LAMBDA_RES_M[RS_IDX]
        md.attrs["spec_ring_neff"]         = RING_NEFF_TE[RE_IDX]
        md.attrs["spec_ring_ng"]           = RING_NG_TE[RE_IDX]
        md.attrs["spec_ring_lambda_res"]   = RING_LAMBDA_RES_M[RE_IDX]
        md.attrs["ona_lambda_start_m"]     = ONA_LAMBDA_START_M
        md.attrs["ona_lambda_stop_m"]      = ONA_LAMBDA_STOP_M
        md.attrs["ona_n_points"]           = ONA_N_POINTS
        md.attrs["ona_power_dBm"]          = ONA_POWER_DBM
        md.attrs["power_calc_method"]      = (
            "P_det[W] = P_source[W] * mean(|T_drop(lambda)|^2) over ONA band"
        )
        md.attrs["timestamp_start"]        = datetime.now().isoformat()

        rg = f.create_group("results")
        rg.create_dataset("T_sensor_thr_dB",
                          data=np.full((n_pts, n_wl), np.nan), chunks=(1, n_wl))
        rg.create_dataset("T_spec_drop_dB",
                          data=np.full((n_pts, n_wl), np.nan), chunks=(1, n_wl))
        rg.create_dataset("T_spec_thr_dB",
                          data=np.full((n_pts, n_wl), np.nan), chunks=(1, n_wl))
        rg.create_dataset("drop_power_dBm",
                          data=np.full(n_pts, np.nan), chunks=(1,))

        f.create_group("flags").create_dataset(
            "computed", data=np.zeros(n_pts, dtype=bool), chunks=(1,))

    log.info(f"HDF5 2-ring initialised ({n_wl} wl pts) → {HDF5_2RING_PATH}")


def run_radius_sweep(hide_gui: bool = False) -> dict:
    n_pts = RADIUS_SWEEP_N

    wl_m = T_s_thr = T_e_drop = T_e_thr = p_drop = None
    computed   = np.zeros(n_pts, dtype=bool)
    hdf5_ready = False

    # ── Resume desde caché ────────────────────────────────────────────────────
    if HDF5_2RING_PATH.exists():
        log.info(f"Caché 2-ring encontrado → {HDF5_2RING_PATH}")
        try:
            with h5py.File(HDF5_2RING_PATH, "r") as f:
                wl_m    = f["metadata/wavelengths_m"][:]
                T_s_thr = f["results/T_sensor_thr_dB"][:]
                T_e_drop= f["results/T_spec_drop_dB"][:]
                T_e_thr = f["results/T_spec_thr_dB"][:]
                p_drop  = f["results/drop_power_dBm"][:]
                computed[:] = f["flags/computed"][:]
            hdf5_ready = True
            n_cached   = int(computed.sum())
            log.info(f"  Cached: {n_cached}/{n_pts}  |  Remaining: {n_pts-n_cached}")
            if n_pts - n_cached == 0:
                log.info("  Todos los puntos en caché — sin lanzar INTERCONNECT.")
                return _pack_2ring(wl_m, T_s_thr, T_e_drop, T_e_thr, p_drop, computed)
        except Exception as exc:
            log.warning(f"Caché ilegible ({exc}). Empezando desde cero.")
            wl_m = T_s_thr = T_e_drop = T_e_thr = p_drop = None
            computed[:] = False
            hdf5_ready  = False
    else:
        log.info("Sin caché — sweep desde cero.")

    log.info("Lanzando INTERCONNECT para sweep de radio …")
    ic         = lumapi.INTERCONNECT(hide=hide_gui)
    runs_done  = 0
    runs_total = int((~computed).sum())
    t_start    = time.time()

    try:
        _build_2ring_circuit(ic)
        log.info(f"Circuito 2-ring listo — {runs_total} puntos por computar …")

        for s_idx in range(n_pts):
            if computed[s_idx]:
                continue

            circum_val = float(SWEEP_CIRCUM_E_M[s_idx])
            radius_val = float(SWEEP_RADIUS_E_M[s_idx])

            _eval(ic, "switchtodesign")
            _update_re_radius(ic, circum_val)

            # ── Run ───────────────────────────────────────────────────────────
            try:
                _eval(ic, "run")
            except ICScriptError as exc:
                log.warning(f"  RUN FAILED  pt={s_idx:3d}  R={radius_val*1e6:.5f} µm  → {exc}")
                computed[s_idx] = True
                if hdf5_ready:
                    with h5py.File(HDF5_2RING_PATH, "r+") as hf:
                        hf["flags/computed"][s_idx] = True
                        hf.flush()
                continue

            # ── Extract ───────────────────────────────────────────────────────
            try:
                wl_i, ts_i, ted_i, tet_i, pd_i = _extract_2ring_results(ic)
            except Exception as exc:
                log.warning(f"  EXTRACT FAILED  pt={s_idx:3d}: {exc}")
                computed[s_idx] = True
                continue

            # ── Inicializar arrays en el primer punto válido ──────────────────
            if wl_m is None:
                n_wl    = len(wl_i)
                wl_m    = wl_i
                T_s_thr = np.full((n_pts, n_wl), np.nan)
                T_e_drop= np.full((n_pts, n_wl), np.nan)
                T_e_thr = np.full((n_pts, n_wl), np.nan)
                p_drop  = np.full(n_pts, np.nan)
                if not hdf5_ready:
                    _init_hdf5_2ring(wl_i)
                    hdf5_ready = True

            T_s_thr [s_idx, :] = ts_i
            T_e_drop[s_idx, :] = ted_i
            T_e_thr [s_idx, :] = tet_i
            p_drop  [s_idx]    = pd_i
            computed[s_idx]    = True

            # ── Flush HDF5 ────────────────────────────────────────────────────
            with h5py.File(HDF5_2RING_PATH, "r+") as hf:
                hf["results/T_sensor_thr_dB"][s_idx, :] = ts_i
                hf["results/T_spec_drop_dB"] [s_idx, :] = ted_i
                hf["results/T_spec_thr_dB"]  [s_idx, :] = tet_i
                hf["results/drop_power_dBm"] [s_idx]    = pd_i
                hf["flags/computed"]         [s_idx]    = True
                hf.flush()

            runs_done += 1
            if runs_done % 10 == 0 or runs_done == runs_total:
                elapsed = time.time() - t_start
                rate    = runs_done / elapsed if elapsed > 0 else 1e-9
                eta     = (runs_total - runs_done) / rate
                log.info(
                    f"  [{runs_done:3d}/{runs_total}]  "
                    f"R={radius_val*1e6:.5f} µm  │  "
                    f"P_drop={pd_i:.2f} dBm  │  "
                    f"{rate:.2f} sim/s  │  ETA {eta:.0f} s"
                )

        if hdf5_ready:
            with h5py.File(HDF5_2RING_PATH, "r+") as hf:
                hf["metadata"].attrs["timestamp_end"]  = datetime.now().isoformat()
                hf["metadata"].attrs["runs_completed"] = int(computed.sum())

    finally:
        try:
            ic.close()
        except Exception:
            pass
        log.info("INTERCONNECT 2-ring session closed.")

    elapsed = time.time() - t_start
    log.info(
        f"Sweep radio listo │ {runs_done} runs │ "
        f"total={elapsed:.1f} s │ avg={elapsed/max(runs_done,1):.2f} s/sim"
    )
    return _pack_2ring(wl_m, T_s_thr, T_e_drop, T_e_thr, p_drop, computed)


def _pack_2ring(wl, ts, ted, tet, pd, comp) -> dict:
    return dict(
        radius_sweep_m  = SWEEP_RADIUS_E_M,
        circum_sweep_m  = SWEEP_CIRCUM_E_M,
        wavelengths_m   = wl,
        T_sensor_thr_dB = ts,
        T_spec_drop_dB  = ted,
        T_spec_thr_dB   = tet,
        drop_power_dBm  = pd,
        computed        = comp,
    )


def _rs_mask() -> np.ndarray:
    return radius_sweep_results["computed"].astype(bool)


def plot_rs_drop_power_vs_radius(figsize=(10, 5), save: bool = True) -> plt.Figure:
    """
    Potencia integrada en el drop de RE_SPEC vs radio (y circunferencia).
    Eje X inferior: radio [µm].  Eje X superior: circunferencia [µm].
    """
    mask  = _rs_mask()
    R_v   = rs_radius_m[mask]  * 1e6    # µm
    C_v   = rs_circum_m[mask]  * 1e6    # µm
    P_v   = rs_p_drop[mask]             # dBm

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(R_v, P_v, color="#2166ac", lw=1.8,
            marker="o", ms=3, alpha=0.9, label="$P_{drop}$ espectrómetro")

    # Marcar radio de diseño del espectrómetro
    R_design_um = RING_RADIUS_M[RE_IDX] * 1e6
    ax.axvline(R_design_um, color="#d6604d", lw=1.2, ls="--",
               label=f"Radio diseño = {R_design_um:.4f} µm")

    ax.set_xlabel("Radio del anillo espectrómetro  $R_E$  (µm)", fontsize=12)
    ax.set_ylabel("Potencia en detector  (dBm)", fontsize=12)
    ax.set_title(
        "Potencia drop del espectrómetro vs radio  [2-ring circuit]\n"
        r"$P_{det} = P_{src} \cdot \langle|T_{drop}(\lambda)|^2\rangle_\lambda$",
        fontsize=12,
    )

    # Eje X superior con circunferencia
    ax2 = ax.twiny()
    ax2.set_xlim(
        ax.get_xlim()[0] * 2.0 * math.pi,
        ax.get_xlim()[1] * 2.0 * math.pi,
    )
    ax2.set_xlabel("Circunferencia  $L_E = 2\\pi R_E$  (µm)", fontsize=10)
    ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax.legend(framealpha=0.9, fontsize=9)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig.tight_layout()

    if save:
        fig.savefig(FIGURES_DIR / "rs_drop_power_vs_radius.png")
        fig.savefig(FIGURES_DIR / "rs_drop_power_vs_radius.pdf")
        log.info("Saved → rs_drop_power_vs_radius.png/pdf")
    return fig


def plot_rs_drop_spectra_vs_radius(n_curves: int = 200,
                                    figsize=(10, 5), save: bool = True) -> plt.Figure:
    """
    Espectros de transmisión del drop de RE_SPEC para cada valor de radio.
    Colormap = radio del espectrómetro.
    """
    mask      = _rs_mask()
    valid_idx = np.where(mask)[0]
    n_sel     = min(n_curves, len(valid_idx))
    sel_idx   = valid_idx[
        np.round(np.linspace(0, len(valid_idx) - 1, n_sel)).astype(int)
    ]
    R_sel = rs_radius_m[sel_idx] * 1e6

    cmap = plt.get_cmap("plasma")
    norm = Normalize(vmin=R_sel.min(), vmax=R_sel.max())
    sm   = ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])

    fig, ax = plt.subplots(figsize=figsize)
    for idx in sel_idx:
        ax.plot(rs_wl_nm, rs_T_spc_drp[idx],
                color=cmap(norm(rs_radius_m[idx] * 1e6)),
                lw=0.8, alpha=0.70)
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("$R_E$  (µm)", fontsize=10)
    ax.set_xlabel("Longitud de onda  (nm)")
    ax.set_ylabel("Transmisión drop  (dB)")
    ax.set_title(
        f"Espectros drop del espectrómetro vs radio  [{n_sel} curvas]\n"
        "Color = $R_E$ [µm]  —  2-ring circuit"
    )
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "rs_drop_spectra_vs_radius.png")
        fig.savefig(FIGURES_DIR / "rs_drop_spectra_vs_radius.pdf")
        log.info("Saved → rs_drop_spectra_vs_radius.png/pdf")
    return fig


def plot_rs_drop_heatmap(figsize=(10, 4), cmap_name: str = "inferno",
                          save: bool = True) -> plt.Figure:
    """
    Heatmap:  eje X = longitud de onda [nm],  eje Y = radio [µm],
    color = transmisión drop [dB].
    """
    mask   = _rs_mask()
    R_v    = rs_radius_m[mask] * 1e6
    spec_v = rs_T_spc_drp[mask, :]

    vmin = np.nanpercentile(spec_v, 2)
    vmax = np.nanpercentile(spec_v, 98)

    fig, ax = plt.subplots(figsize=figsize)
    img = ax.pcolormesh(rs_wl_nm, R_v, spec_v,
                        cmap=cmap_name, vmin=vmin, vmax=vmax, shading="auto")
    cbar = fig.colorbar(img, ax=ax, pad=0.01)
    cbar.set_label("Transmisión drop  (dB)", fontsize=10)
    ax.set_xlabel("Longitud de onda  (nm)")
    ax.set_ylabel("Radio espectrómetro  $R_E$  (µm)")
    ax.set_title(
        "Heatmap espectral drop del espectrómetro  [2-ring circuit]\n"
        "Eje Y = $R_E$  —  barrido de radio"
    )
    fig.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "rs_drop_heatmap_radius_vs_wl.png", dpi=200)
        fig.savefig(FIGURES_DIR / "rs_drop_heatmap_radius_vs_wl.pdf")
        log.info("Saved → rs_drop_heatmap_radius_vs_wl.png/pdf")
    return fig


def plot_rs_sensor_through(n_curves: int = 200,
                            figsize=(10, 5), save: bool = True) -> plt.Figure:
    """Espectros sensor through para cada radio del espectrómetro."""
    mask      = _rs_mask()
    valid_idx = np.where(mask)[0]
    n_sel     = min(n_curves, len(valid_idx))
    sel_idx   = valid_idx[
        np.round(np.linspace(0, len(valid_idx) - 1, n_sel)).astype(int)
    ]
    R_sel = rs_radius_m[sel_idx] * 1e6

    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=R_sel.min(), vmax=R_sel.max())
    sm   = ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])

    fig, ax = plt.subplots(figsize=figsize)
    for idx in sel_idx:
        ax.plot(rs_wl_nm, rs_T_sen_thr[idx],
                color=cmap(norm(rs_radius_m[idx] * 1e6)),
                lw=0.8, alpha=0.70)
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("$R_E$  (µm)", fontsize=10)
    ax.set_xlabel("Longitud de onda  (nm)")
    ax.set_ylabel("Transmisión sensor through  (dB)")
    ax.set_title(
        f"Sensor through vs radio del espectrómetro  [{n_sel} curvas]\n"
        "Color = $R_E$ [µm]  —  2-ring circuit"
    )
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "rs_sensor_through_vs_radius.png")
        fig.savefig(FIGURES_DIR / "rs_sensor_through_vs_radius.pdf")
        log.info("Saved → rs_sensor_through_vs_radius.png/pdf")
    return fig

def run(state=None):
    """Execute this step. `state` carries the shared namespace
    between steps (the notebook's old kernel globals + bridge).
    Returns the updated `state`."""
    state = {} if state is None else state
    global HDF5_2RING_PATH, ONA2_DROP_SPEC, ONA2_NAME, ONA2_N_INPUTS, ONA2_THROUGH_SENSOR, ONA2_THROUGH_SPEC, RADIUS_SWEEP_N, RADIUS_SWEEP_START_M, RADIUS_SWEEP_STOP_M, RE_IDX, RE_NAME, RS_IDX, RS_NAME, SWEEP_CIRCUM_E_M, SWEEP_CIRCUM_E_UM, SWEEP_RADIUS_E_M, VERSION_2RING, fig_rs1, fig_rs2, fig_rs3, fig_rs4, radius_sweep_results, rs_T_sen_thr, rs_T_spc_drp, rs_T_spc_thr, rs_circum_m, rs_circum_um, rs_computed, rs_p_drop, rs_radius_m, rs_wl_m, rs_wl_nm
    globals()['lumapi'] = import_lumapi()
    globals().update(state)

    RADIUS_SWEEP_START_M = RING_RADIUS_M[RE_IDX]          # 19.1818 µm — valor de diseño
    RADIUS_SWEEP_STOP_M  = RING_RADIUS_M[RE_IDX] * 1.005  # +0.5 % → ~19.278 µm
    RADIUS_SWEEP_N       = 200
    SWEEP_RADIUS_E_M = np.linspace(
        RADIUS_SWEEP_START_M,
        RADIUS_SWEEP_STOP_M,
        RADIUS_SWEEP_N,
    )
    SWEEP_CIRCUM_E_M  = SWEEP_RADIUS_E_M * 2.0 * math.pi
    SWEEP_CIRCUM_E_UM = SWEEP_CIRCUM_E_M * 1e6
    RS_NAME = "RS_SENSOR"      # anillo sensor
    RE_NAME = "RE_SPEC"        # anillo espectrómetro (swept)
    ONA2_NAME = "ONA_2ring"    # ONA dedicado para este sweep
    ONA2_N_INPUTS         = 3
    ONA2_THROUGH_SENSOR   = 1
    ONA2_DROP_SPEC        = 2   # ← potencia de interés
    ONA2_THROUGH_SPEC     = 3
    HDF5_2RING_PATH = DATA_DIR / "ICNT_2Ring_RadiusSweep_V1.h5"
    VERSION_2RING = "ICNT_2Ring_RadiusSweep_V1"
    print("=" * 72)
    print("  SWEEP DE RADIO — 2 anillos (sensor + espectrómetro)  [CELL 8]")
    print("=" * 72)
    print(f"  Sensor  (RS) : RING_RADIUS_M[{RS_IDX}] = {RING_RADIUS_M[RS_IDX]*1e6:.4f} µm  [FIJO]")
    print(f"                 neff={RING_NEFF_TE[RS_IDX]:.6f}  ng={RING_NG_TE[RS_IDX]:.6f}")
    print(f"                 λ_res={RING_LAMBDA_RES_M[RS_IDX]*1e9:.4f} nm")
    print(f"  Espec.  (RE) : radio barrido de {RADIUS_SWEEP_START_M*1e6:.4f} → "
          f"{RADIUS_SWEEP_STOP_M*1e6:.4f} µm  ({RADIUS_SWEEP_N} pts)")
    print(f"                 Δcircunf. = {(SWEEP_CIRCUM_E_M[-1]-SWEEP_CIRCUM_E_M[0])*1e9:.2f} nm")
    print(f"                 neff={RING_NEFF_TE[RE_IDX]:.6f}  ng={RING_NG_TE[RE_IDX]:.6f}  [fijos]")
    print(f"                 λ_res={RING_LAMBDA_RES_M[RE_IDX]*1e9:.4f} nm  [fijo]")
    print(f"  ONA          : λ {ONA_LAMBDA_START_M*1e9:.2f}–{ONA_LAMBDA_STOP_M*1e9:.2f} nm  "
          f"│  {ONA_N_POINTS} pts  │  {ONA2_N_INPUTS} inputs")
    print(f"  HDF5         : {HDF5_2RING_PATH}")
    print("=" * 72)
    radius_sweep_results = run_radius_sweep(hide_gui=False)
    rs_wl_m      = radius_sweep_results["wavelengths_m"]
    rs_radius_m  = radius_sweep_results["radius_sweep_m"]
    rs_circum_m  = radius_sweep_results["circum_sweep_m"]
    rs_T_sen_thr = radius_sweep_results["T_sensor_thr_dB"]
    rs_T_spc_drp = radius_sweep_results["T_spec_drop_dB"]
    rs_T_spc_thr = radius_sweep_results["T_spec_thr_dB"]
    rs_p_drop    = radius_sweep_results["drop_power_dBm"]
    rs_computed  = radius_sweep_results["computed"]
    rs_wl_nm     = rs_wl_m * 1e9 if rs_wl_m is not None else None
    rs_circum_um = rs_circum_m * 1e6
    print(f"\n  Sweep radio completo — {rs_computed.sum()} / {RADIUS_SWEEP_N} pts")
    if rs_wl_m is not None:
        print(f"  T_sensor_thr_dB  shape : {rs_T_sen_thr.shape}")
        print(f"  T_spec_drop_dB   shape : {rs_T_spc_drp.shape}")
        print(f"  drop_power_dBm   shape : {rs_p_drop.shape}")
    print(f"  HDF5                   : {HDF5_2RING_PATH}")
    if rs_wl_m is not None and rs_computed.sum() > 0:
        fig_rs1 = plot_rs_drop_power_vs_radius()
        fig_rs2 = plot_rs_drop_spectra_vs_radius(n_curves=200)
        fig_rs3 = plot_rs_drop_heatmap()
        fig_rs4 = plot_rs_sensor_through(n_curves=200)
        plt.show()
        print(f"\n  Figuras sweep radio → {FIGURES_DIR}")
        print(f"  HDF5              → {HDF5_2RING_PATH}")
    else:
        print("  Sin datos disponibles para graficar. Ejecuta run_radius_sweep().")

    state.update({k: globals().get(k) for k in [
        'HDF5_2RING_PATH', 'ONA2_DROP_SPEC', 'ONA2_NAME', 'ONA2_N_INPUTS', 'ONA2_THROUGH_SENSOR', 'ONA2_THROUGH_SPEC',
        'RADIUS_SWEEP_N', 'RADIUS_SWEEP_START_M', 'RADIUS_SWEEP_STOP_M', 'RE_IDX', 'RE_NAME', 'RS_IDX',
        'RS_NAME', 'SWEEP_CIRCUM_E_M', 'SWEEP_CIRCUM_E_UM', 'SWEEP_RADIUS_E_M', 'VERSION_2RING', 'fig_rs1',
        'fig_rs2', 'fig_rs3', 'fig_rs4', 'radius_sweep_results', 'rs_T_sen_thr', 'rs_T_spc_drp',
        'rs_T_spc_thr', 'rs_circum_m', 'rs_circum_um', 'rs_computed', 'rs_p_drop', 'rs_radius_m',
        'rs_wl_m', 'rs_wl_nm',
    ] if k in globals()})
    return state


if __name__ == "__main__":
    run()
