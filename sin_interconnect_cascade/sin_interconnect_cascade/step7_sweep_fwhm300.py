"""
step7_sweep_fwhm300.py — full cascade sweep with the FWHM~300 pm coupling variant (kappa_03).

Re-runs the 14-ring cascade neff sweep with the narrower FWHM~300 pm
couplers (kappa_03) and its own HDF5 cache, then renders the matching
neff- and n_cladding-axis figures. Engine primitives are imported
from step 1; only the coupling arrays and dataset names differ.
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
    ICScriptError, _eval, _try_eval, _extract_results, _update_ring1_neff_ng, drop_ona_input, ring_name,
)
from .step6_kappa_variants import (
    RING_KAPPA_INPUT_SQ_03, RING_KAPPA_DROP_SQ_03,
)

from .config import *  # shared platform constants & paths
from .lumerical_session import import_lumapi
from .plotting import apply_style, save_fig, make_colorbar
from . import plotting as _plot

apply_style()

# ─────────────────────────────────────────────────────────
#  Module constants / design knobs for this step
# ─────────────────────────────────────────────────────────
VERSION_NAME_03  = "ICNT_14Ring_Cascade_UniDir_neff_sweep_V3_kappa03"
HDF5_PATH_03     = DATA_DIR / f"{VERSION_NAME_03}.h5"
FIGURES_DIR_03   = DATA_DIR / "figures_kappa03"

def _apply_ring_params_03(ic, ring_idx: int,
                           neff_override=None, ng_override=None) -> None:
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
    _eval(ic, f'setnamed("{name}", "coupling coefficient 1 1", {RING_KAPPA_INPUT_SQ_03[ring_idx]:.12f})')
    _eval(ic, f'setnamed("{name}", "coupling coefficient 1 2", {RING_KAPPA_DROP_SQ_03[ring_idx]:.12f})')
    _eval(ic, f'setnamed("{name}", "configuration", "unidirectional")')


def _build_circuit_03(ic) -> None:
    _eval(ic, "switchtodesign")
    _try_eval(ic, "selectall")
    _try_eval(ic, "delete")

    pwr_W   = 10.0 ** (ONA_POWER_DBM / 10.0) * 1e-3
    f_start = SPEED_OF_LIGHT / ONA_LAMBDA_STOP_M
    f_stop  = SPEED_OF_LIGHT / ONA_LAMBDA_START_M

    # ONA — 15 puertos de entrada
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

    # Anillos con kappa_03
    for i in range(N_RINGS):
        rn = ring_name(i + 1)
        _eval(ic, 'addelement("Double Bus Ring Resonator")')
        _eval(ic, f'set("name", "{rn}")')
        _try_eval(ic, f'set("x position", {float((i + 1) * 220)})')
        _try_eval(ic, f'set("y position", 0)')
        _apply_ring_params_03(ic, ring_idx=i)
        log.info(
            f"  [03] RING_{i+1:2d} — "
            f"κ²_in={RING_KAPPA_INPUT_SQ_03[i]:.6f}  "
            f"κ²_dr={RING_KAPPA_DROP_SQ_03[i]:.6f}"
        )

    # Conexiones — topología V3 idéntica a _build_circuit
    def wire(elem_a, port_a, elem_b, port_b):
        _eval(ic, f'connect("{elem_a}", "{port_a}", "{elem_b}", "{port_b}")')

    wire(ONA_NAME,        "output",   ring_name(1), "input")
    wire(ring_name(1),    "output 1", ONA_NAME,     f"input {THROUGH_SENSOR_INPUT}")
    wire(ring_name(1),    "output 2", ring_name(2), "input")
    for i in range(2, N_RINGS):
        wire(ring_name(i), "output 1", ring_name(i + 1), "input")
        wire(ring_name(i), "output 2", ONA_NAME, f"input {drop_ona_input(i)}")
    wire(ring_name(N_RINGS), "output 2", ONA_NAME, f"input {drop_ona_input(N_RINGS)}")
    wire(ring_name(N_RINGS), "output 1", ONA_NAME, f"input {THROUGH_FINAL_INPUT}")

    log.info("V3-kappa03 circuit built.")


def _init_hdf5_03(wl_ref_m: np.ndarray) -> None:
    n_pts = SWEEP_N_POINTS
    n_wl  = len(wl_ref_m)
    with h5py.File(HDF5_PATH_03, "w") as f:
        md = f.create_group("metadata")
        md.create_dataset("neff_sweep",    data=SWEEP_NEFF)
        md.create_dataset("ng_sweep",      data=SWEEP_NG)
        md.create_dataset("wavelengths_m", data=wl_ref_m)
        md.attrs["version_name"]        = VERSION_NAME_03
        md.attrs["kappa_variant"]       = "kappa_03"
        md.attrs["n_rings"]             = N_RINGS
        md.attrs["n_drops"]             = N_DROPS
        md.attrs["sweep_n_points"]      = SWEEP_N_POINTS
        md.attrs["ring_model"]          = "Double Bus Ring Resonator"
        md.attrs["ring_configuration"]  = "unidirectional"
        md.attrs["topology"]            = (
            "V3-kappa03: ONA multiport, no OPMs. "
            "ONA input 1=RING_1 through, input k=RING_k drop (k=2-14), "
            "input 15=RING_14 through"
        )
        md.attrs["ona_lambda_start_m"]  = ONA_LAMBDA_START_M
        md.attrs["ona_lambda_stop_m"]   = ONA_LAMBDA_STOP_M
        md.attrs["ona_n_points"]        = ONA_N_POINTS
        md.attrs["ona_power_dBm"]       = ONA_POWER_DBM
        md.attrs["ona_n_input_ports"]   = ONA_N_INPUT_PORTS
        md.attrs["power_calc_method"]   = (
            "P_det[W] = P_source[W] * mean(|T_drop(lambda)|^2) over ONA band"
        )
        md.attrs["timestamp_start"]     = datetime.now().isoformat()
        for i in range(N_RINGS):
            p = f"ring{i+1}_"
            md.attrs[p + "kappa_input_sq"] = RING_KAPPA_INPUT_SQ_03[i]
            md.attrs[p + "kappa_drop_sq"]  = RING_KAPPA_DROP_SQ_03[i]
            md.attrs[p + "radius_m"]       = RING_RADIUS_M[i]
            md.attrs[p + "lambda_res_m"]   = RING_LAMBDA_RES_M[i]
            md.attrs[p + "neff_TE"]        = RING_NEFF_TE[i]
            md.attrs[p + "ng_TE"]          = RING_NG_TE[i]
            md.attrs[p + "loss_dB_per_m"]  = RING_LOSS_DB_PER_M[i]

        rg = f.create_group("results")
        rg.create_dataset("T_sensor_through_dB",
                          data=np.full((n_pts, n_wl), np.nan), chunks=(1, n_wl))
        rg.create_dataset("T_final_through_dB",
                          data=np.full((n_pts, n_wl), np.nan), chunks=(1, n_wl))
        rg.create_dataset("T_drop_dB",
                          data=np.full((n_pts, N_DROPS, n_wl), np.nan),
                          chunks=(1, 1, n_wl))
        rg.create_dataset("drop_power_dBm",
                          data=np.full((n_pts, N_DROPS), np.nan),
                          chunks=(1, N_DROPS))
        f.create_group("flags").create_dataset(
            "computed", data=np.zeros(n_pts, dtype=bool), chunks=(1,))

    log.info(f"HDF5 kappa03 inicializado → {HDF5_PATH_03}")


def run_interconnect_sweep_03(hide_gui: bool = False):
    n_pts = SWEEP_N_POINTS
    wavelengths_m = T_sensor_thr = T_final_thr = T_drop = p_drop = None
    computed_03   = np.zeros(n_pts, dtype=bool)
    hdf5_ready    = False

    if HDF5_PATH_03.exists():
        log.info(f"[03] Caché encontrado → {HDF5_PATH_03}")
        try:
            with h5py.File(HDF5_PATH_03, "r") as f:
                wavelengths_m = f["metadata/wavelengths_m"][:]
                T_sensor_thr  = f["results/T_sensor_through_dB"][:]
                T_final_thr   = f["results/T_final_through_dB"][:]
                T_drop        = f["results/T_drop_dB"][:]
                p_drop        = f["results/drop_power_dBm"][:]
                computed_03[:]= f["flags/computed"][:]
            hdf5_ready = True
            n_cached   = int(computed_03.sum())
            log.info(f"[03] Cached: {n_cached}/{n_pts}  |  Pending: {n_pts - n_cached}")
            if n_pts - n_cached == 0:
                log.info("[03] Todos los puntos en caché — sin lanzar INTERCONNECT.")
                return _pack_results_03(
                    wavelengths_m, T_sensor_thr, T_final_thr, T_drop, p_drop, computed_03)
        except Exception as exc:
            log.warning(f"[03] Caché ilegible ({exc}). Empezando desde cero.")
            wavelengths_m = T_sensor_thr = T_final_thr = T_drop = p_drop = None
            computed_03[:] = False
            hdf5_ready = False
    else:
        log.info("[03] Sin caché — sweep desde cero.")

    log.info("[03] Lanzando INTERCONNECT …")
    ic         = lumapi.INTERCONNECT(hide=hide_gui)
    runs_done  = 0
    runs_total = int((~computed_03).sum())
    t_start    = time.time()

    try:
        _build_circuit_03(ic)
        log.info(f"[03] Circuito listo — {runs_total} puntos por computar …")

        for s_idx in range(n_pts):
            if computed_03[s_idx]:
                continue

            neff_val = float(SWEEP_NEFF[s_idx])
            ng_val   = float(SWEEP_NG[s_idx])

            _eval(ic, "switchtodesign")
            _update_ring1_neff_ng(ic, neff_val, ng_val)   # reutiliza la función de Cell 3

            try:
                _eval(ic, "run")
            except ICScriptError as exc:
                log.warning(f"  [03] RUN FAILED pt={s_idx:3d} neff={neff_val:.6f} → {exc}")
                computed_03[s_idx] = True
                if hdf5_ready:
                    with h5py.File(HDF5_PATH_03, "r+") as hf:
                        hf["flags/computed"][s_idx] = True; hf.flush()
                continue

            try:
                wl_m, t_sen, t_fin, t_drp, p_drp = _extract_results(ic)  # reutiliza Cell 3
            except Exception as exc:
                log.warning(f"  [03] EXTRACT FAILED pt={s_idx:3d}: {exc}")
                computed_03[s_idx] = True
                continue

            if wavelengths_m is None:
                n_wl          = len(wl_m)
                wavelengths_m = wl_m
                T_sensor_thr  = np.full((n_pts, n_wl),          np.nan)
                T_final_thr   = np.full((n_pts, n_wl),          np.nan)
                T_drop        = np.full((n_pts, N_DROPS, n_wl), np.nan)
                p_drop        = np.full((n_pts, N_DROPS),        np.nan)
                if not hdf5_ready:
                    _init_hdf5_03(wl_m)
                    hdf5_ready = True

            T_sensor_thr[s_idx, :]    = t_sen
            T_final_thr [s_idx, :]    = t_fin
            T_drop      [s_idx, :, :] = t_drp
            p_drop      [s_idx, :]    = p_drp
            computed_03 [s_idx]        = True

            with h5py.File(HDF5_PATH_03, "r+") as hf:
                hf["results/T_sensor_through_dB"][s_idx, :]    = t_sen
                hf["results/T_final_through_dB"] [s_idx, :]    = t_fin
                hf["results/T_drop_dB"]          [s_idx, :, :] = t_drp
                hf["results/drop_power_dBm"]     [s_idx, :]    = p_drp
                hf["flags/computed"]             [s_idx]        = True
                hf.flush()

            runs_done += 1
            if runs_done % 5 == 0 or runs_done == runs_total:
                elapsed = time.time() - t_start
                rate    = runs_done / elapsed if elapsed > 0 else 1e-9
                eta     = (runs_total - runs_done) / rate
                log.info(
                    f"  [03][{runs_done:3d}/{runs_total}]  "
                    f"neff={neff_val:.6f}  ng={ng_val:.6f}  │  "
                    f"{rate:.2f} sim/s  │  ETA {eta:5.0f} s"
                )

        if hdf5_ready:
            with h5py.File(HDF5_PATH_03, "r+") as hf:
                hf["metadata"].attrs["timestamp_end"]  = datetime.now().isoformat()
                hf["metadata"].attrs["runs_completed"] = int(computed_03.sum())

    finally:
        try: ic.close()
        except Exception: pass
        log.info("[03] INTERCONNECT session closed.")

    elapsed = time.time() - t_start
    log.info(
        f"[03] Sweep kappa03 listo │ {runs_done} runs │ "
        f"total={elapsed:.1f} s │ avg={elapsed/max(runs_done,1):.2f} s/sim"
    )
    return _pack_results_03(
        wavelengths_m, T_sensor_thr, T_final_thr, T_drop, p_drop, computed_03)


def _pack_results_03(wl, t_sen, t_fin, t_drop, p_drop, comp):
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


def _valid_mask_03():
    return sweep_results_03["computed"].astype(bool)


def plot_03_drop_power_vs_neff(figsize=(11, 6), save=True):
    mask   = _valid_mask_03()
    neff_v = sweep_results_03["neff_sweep"][mask]
    p_v    = sweep_results_03["drop_power_dBm"][mask, :]
    cmap   = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=figsize)
    for k in range(N_DROPS):
        ax.plot(neff_v, p_v[:, k], color=cmap(k / N_DROPS),
                lw=1.5, marker="o", ms=2.5, alpha=0.85, label=DROP_LABELS[k])
    ax.set_xlabel("Ring 1 — $n_{eff}$ (TE)  [sensor]", fontsize=12)
    ax.set_ylabel("Detector power  (dBm)", fontsize=12)
    ax.set_title(
        "Integrated Drop Power vs Sensor $n_{eff}$  [V3 — kappa_03]\n"
        r"$P_{det} = P_{src} \cdot \langle|T_{drop}(\lambda)|^2\rangle_\lambda$"
        "   —   13 spectrometer rings", fontsize=12)
    ax.legend(ncol=3, framealpha=0.88, fontsize=8, loc="best")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR_03 / "drop_power_vs_neff_all.png")
        fig.savefig(FIGURES_DIR_03 / "drop_power_vs_neff_all.pdf")
    return fig


def plot_03_sensor_through_sweep(n_curves=200, figsize=(10, 5),
                                  cmap_name="plasma", save=True):
    neff_arr  = sweep_results_03["neff_sweep"]
    wl_nm     = sweep_results_03["wavelengths_m"] * 1e9
    T_data    = sweep_results_03["T_sensor_through_dB"]
    mask      = _valid_mask_03()
    valid_idx = np.where(mask)[0]
    n_sel     = min(n_curves, len(valid_idx))
    sel_idx   = valid_idx[np.round(np.linspace(0, len(valid_idx)-1, n_sel)).astype(int)]
    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=neff_arr[sel_idx].min(), vmax=neff_arr[sel_idx].max())
    sm   = ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    fig, ax = plt.subplots(figsize=figsize)
    for idx in sel_idx:
        ax.plot(wl_nm, T_data[idx], color=cmap(norm(neff_arr[idx])), lw=0.8, alpha=0.70)
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Ring 1 — $n_{eff}$ (TE)", fontsize=10)
    ax.set_xlabel("Wavelength  (nm)"); ax.set_ylabel("Transmission  (dB)")
    ax.set_title(f"Sensor Through Spectrum — kappa_03\n({n_sel} curves, colour = $n_{{eff,1}}$)")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR_03 / f"sensor_through_sweep_{n_sel}curves.png")
        fig.savefig(FIGURES_DIR_03 / f"sensor_through_sweep_{n_sel}curves.pdf")
    return fig


def plot_03_final_through_sweep(n_curves=200, figsize=(10, 5),
                                 cmap_name="viridis", save=True):
    neff_arr  = sweep_results_03["neff_sweep"]
    wl_nm     = sweep_results_03["wavelengths_m"] * 1e9
    T_data    = sweep_results_03["T_final_through_dB"]
    mask      = _valid_mask_03()
    valid_idx = np.where(mask)[0]
    n_sel     = min(n_curves, len(valid_idx))
    sel_idx   = valid_idx[np.round(np.linspace(0, len(valid_idx)-1, n_sel)).astype(int)]
    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=neff_arr[sel_idx].min(), vmax=neff_arr[sel_idx].max())
    sm   = ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    fig, ax = plt.subplots(figsize=figsize)
    for idx in sel_idx:
        ax.plot(wl_nm, T_data[idx], color=cmap(norm(neff_arr[idx])), lw=0.8, alpha=0.70)
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Ring 1 — $n_{eff}$ (TE)", fontsize=10)
    ax.set_xlabel("Wavelength  (nm)"); ax.set_ylabel("Transmission  (dB)")
    ax.set_title(f"Cascade Final Through — kappa_03\n({n_sel} curves, colour = $n_{{eff,1}}$)")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR_03 / f"final_through_sweep_{n_sel}curves.png")
        fig.savefig(FIGURES_DIR_03 / f"final_through_sweep_{n_sel}curves.pdf")
    return fig


def plot_03_drop_spectrum_heatmap(drop_k=1, figsize=(10, 3.5),
                                   cmap_name="inferno",
                                   vmin_dB=None, vmax_dB=None, save=True):
    assert 1 <= drop_k <= N_DROPS
    ring_label = DROP_LABELS[drop_k - 1]
    neff_arr   = sweep_results_03["neff_sweep"]
    wl_nm      = sweep_results_03["wavelengths_m"] * 1e9
    t_drop     = sweep_results_03["T_drop_dB"]
    mask       = _valid_mask_03()
    neff_v     = neff_arr[mask]
    spec_v     = t_drop[mask, drop_k - 1, :]
    _vmin = vmin_dB if vmin_dB is not None else np.nanpercentile(spec_v, 2)
    _vmax = vmax_dB if vmax_dB is not None else np.nanpercentile(spec_v, 98)
    fig, ax = plt.subplots(figsize=figsize)
    img = ax.pcolormesh(wl_nm, neff_v, spec_v,
                        cmap=cmap_name, vmin=_vmin, vmax=_vmax, shading="auto")
    cbar = fig.colorbar(img, ax=ax, pad=0.01)
    cbar.set_label("Transmission  (dB)", fontsize=10)
    ax.set_xlabel("Wavelength  (nm)"); ax.set_ylabel("Ring 1 — $n_{eff}$ (TE)")
    ax.set_title(
        f"Drop Spectrum Heatmap — {ring_label}  [kappa_03]\n"
        "neff sweep on sensor ring (RING_1)")
    fig.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR_03 / f"drop{drop_k}_spectrum_heatmap.png")
        fig.savefig(FIGURES_DIR_03 / f"drop{drop_k}_spectrum_heatmap.pdf")
    return fig


def plot_03_all_drop_heatmaps(ncols=4, cmap_name="inferno", save=True):
    neff_arr = sweep_results_03["neff_sweep"]
    wl_nm    = sweep_results_03["wavelengths_m"] * 1e9
    t_drop   = sweep_results_03["T_drop_dB"]
    mask     = _valid_mask_03()
    neff_v   = neff_arr[mask]
    nrows    = math.ceil(N_DROPS / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 3.8, nrows * 2.8),
                              sharex=True, sharey=True)
    axes_flat = axes.flatten()
    all_vals  = t_drop[mask].ravel()
    vmin = np.nanpercentile(all_vals, 2); vmax = np.nanpercentile(all_vals, 98)
    im = None
    for k in range(1, N_DROPS + 1):
        ax   = axes_flat[k - 1]
        spec = t_drop[mask, k - 1, :]
        im   = ax.pcolormesh(wl_nm, neff_v, spec,
                             cmap=cmap_name, vmin=vmin, vmax=vmax, shading="auto")
        ax.set_title(f"{DROP_LABELS[k-1]} drop", fontsize=8)
        if k % ncols == 1: ax.set_ylabel("$n_{eff,1}$", fontsize=7)
        if k > (nrows - 1) * ncols: ax.set_xlabel("λ (nm)", fontsize=7)
        ax.tick_params(labelsize=6)
    for ax in axes_flat[N_DROPS:]: ax.set_visible(False)
    if im is not None:
        fig.colorbar(im, ax=axes_flat[:N_DROPS], shrink=0.6, pad=0.02,
                     label="Transmission (dB)", fraction=0.015)
    fig.suptitle(
        "All Drop Spectra  [V3 — kappa_03]\nneff sweep on sensor ring (RING_1)",
        fontsize=11)
    fig.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR_03 / "all_drop_heatmaps_grid.png", dpi=200)
        fig.savefig(FIGURES_DIR_03 / "all_drop_heatmaps_grid.pdf")
    return fig


def plot_03_power_heatmap_drops_vs_neff(figsize=(10, 5),
                                         cmap_name="plasma", save=True):
    neff_arr = sweep_results_03["neff_sweep"]
    p_drop   = sweep_results_03["drop_power_dBm"]
    mask     = _valid_mask_03()
    neff_v   = neff_arr[mask]
    p_v      = p_drop[mask, :].T
    fig, ax  = plt.subplots(figsize=figsize)
    img = ax.pcolormesh(neff_v, np.arange(1, N_DROPS + 1), p_v,
                        cmap=cmap_name, shading="auto")
    cbar = fig.colorbar(img, ax=ax, pad=0.02)
    cbar.set_label("Detector power  (dBm)", fontsize=10)
    ax.set_xlabel("Ring 1 — $n_{eff}$ (TE)  [sensor]", fontsize=11)
    ax.set_ylabel("Spectrometer ring index  (1=RING_2 … 13=RING_14)", fontsize=10)
    ax.set_yticks(np.arange(1, N_DROPS + 1))
    ax.set_yticklabels(DROP_LABELS, fontsize=7)
    ax.set_title(
        "Detector Power Heatmap — All Drops vs $n_{eff}$  [kappa_03]\n"
        r"Colour: $P_{det}$ [dBm]  per spectrometer ring", fontsize=12)
    fig.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR_03 / "power_heatmap_drops_vs_neff.png", dpi=200)
        fig.savefig(FIGURES_DIR_03 / "power_heatmap_drops_vs_neff.pdf")
    return fig


def plot_03_resonance_tracking(figsize=(7, 4.5), save=True):
    neff_arr = sweep_results_03["neff_sweep"]
    wl_nm    = sweep_results_03["wavelengths_m"] * 1e9
    T_data   = sweep_results_03["T_sensor_through_dB"]
    mask     = _valid_mask_03()
    neff_v   = neff_arr[mask]
    T_v      = T_data[mask, :]
    dip_idx  = np.argmin(T_v, axis=1)
    lam_dip  = wl_nm[dip_idx]
    coeffs   = np.polyfit(neff_v, lam_dip, 1)
    sens     = coeffs[0]
    fig, ax  = plt.subplots(figsize=figsize)
    ax.scatter(neff_v, lam_dip, s=20, zorder=5, color="#2166ac", label="Resonance dip")
    ax.plot(neff_v, np.poly1d(coeffs)(neff_v), "r--", lw=1.5,
            label=f"Linear fit   ∂λ/∂n = {sens:.3f} nm / RIU")
    ax.set_xlabel("Ring 1 — $n_{eff}$ (TE)")
    ax.set_ylabel("Resonance wavelength  (nm)")
    ax.set_title(
        f"Resonance Tracking — kappa_03\nSensitivity: {sens:.3f} nm / RIU")
    ax.legend(framealpha=0.9)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR_03 / "resonance_tracking_sensor.png")
        fig.savefig(FIGURES_DIR_03 / "resonance_tracking_sensor.pdf")
    log.info(f"[03] Resonance sensitivity: {sens:.4f} nm/RIU")
    return fig


def plot_03_drop_power_vs_ncladding(figsize=(11, 6), save=True):
    mask = _valid_mask_03()
    nc_v = n_cladding[mask]
    p_v  = sweep_results_03["drop_power_dBm"][mask, :]
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=figsize)
    for k in range(N_DROPS):
        ax.plot(nc_v, p_v[:, k], color=cmap(k / N_DROPS),
                lw=1.5, marker="o", ms=2.5, alpha=0.85, label=DROP_LABELS[k])
    ax.set_xlabel("Índice de refracción del cladding  $n_{clad}$", fontsize=12)
    ax.set_ylabel("Potencia en el detector  (dBm)", fontsize=12)
    ax.set_title(
        "Potencia integrada en drop vs $n_{clad}$  [kappa_03]\n"
        r"$P_{det} = P_{src} \cdot \langle|T_{drop}(\lambda)|^2\rangle_\lambda$"
        "   —   13 anillos espectrómetros", fontsize=12)
    ax.legend(ncol=3, framealpha=0.88, fontsize=8, loc="best")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR_03 / "nc_drop_power_vs_ncladding_all.png")
        fig.savefig(FIGURES_DIR_03 / "nc_drop_power_vs_ncladding_all.pdf")
    return fig


def plot_03_sensor_through_sweep_nc(n_curves=200, figsize=(10, 5),
                                     cmap_name="plasma", save=True):
    wl_nm     = sweep_results_03["wavelengths_m"] * 1e9
    T_data    = sweep_results_03["T_sensor_through_dB"]
    mask      = _valid_mask_03()
    valid_idx = np.where(mask)[0]
    n_sel     = min(n_curves, len(valid_idx))
    sel_idx   = valid_idx[np.round(np.linspace(0, len(valid_idx)-1, n_sel)).astype(int)]
    nc_sel    = n_cladding[sel_idx]
    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=nc_sel.min(), vmax=nc_sel.max())
    sm   = ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    fig, ax = plt.subplots(figsize=figsize)
    for idx in sel_idx:
        ax.plot(wl_nm, T_data[idx], color=cmap(norm(n_cladding[idx])), lw=0.8, alpha=0.70)
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("$n_{clad}$", fontsize=10)
    ax.set_xlabel("Longitud de onda  (nm)"); ax.set_ylabel("Transmisión  (dB)")
    ax.set_title(f"Espectro sensor through — kappa_03\n({n_sel} curvas, color = $n_{{clad}}$)")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR_03 / f"nc_sensor_through_sweep_{n_sel}curves.png")
        fig.savefig(FIGURES_DIR_03 / f"nc_sensor_through_sweep_{n_sel}curves.pdf")
    return fig


def plot_03_final_through_sweep_nc(n_curves=200, figsize=(10, 5),
                                    cmap_name="viridis", save=True):
    wl_nm     = sweep_results_03["wavelengths_m"] * 1e9
    T_data    = sweep_results_03["T_final_through_dB"]
    mask      = _valid_mask_03()
    valid_idx = np.where(mask)[0]
    n_sel     = min(n_curves, len(valid_idx))
    sel_idx   = valid_idx[np.round(np.linspace(0, len(valid_idx)-1, n_sel)).astype(int)]
    nc_sel    = n_cladding[sel_idx]
    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=nc_sel.min(), vmax=nc_sel.max())
    sm   = ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    fig, ax = plt.subplots(figsize=figsize)
    for idx in sel_idx:
        ax.plot(wl_nm, T_data[idx], color=cmap(norm(n_cladding[idx])), lw=0.8, alpha=0.70)
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("$n_{clad}$", fontsize=10)
    ax.set_xlabel("Longitud de onda  (nm)"); ax.set_ylabel("Transmisión  (dB)")
    ax.set_title(f"Through final de cascada — kappa_03\n({n_sel} curvas, color = $n_{{clad}}$)")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR_03 / f"nc_final_through_sweep_{n_sel}curves.png")
        fig.savefig(FIGURES_DIR_03 / f"nc_final_through_sweep_{n_sel}curves.pdf")
    return fig


def plot_03_drop_spectrum_heatmap_nc(drop_k=1, figsize=(10, 3.5),
                                      cmap_name="inferno",
                                      vmin_dB=None, vmax_dB=None, save=True):
    assert 1 <= drop_k <= N_DROPS
    ring_label = DROP_LABELS[drop_k - 1]
    wl_nm  = sweep_results_03["wavelengths_m"] * 1e9
    t_drop = sweep_results_03["T_drop_dB"]
    mask   = _valid_mask_03()
    nc_v   = n_cladding[mask]
    spec_v = t_drop[mask, drop_k - 1, :]
    _vmin  = vmin_dB if vmin_dB is not None else np.nanpercentile(spec_v, 2)
    _vmax  = vmax_dB if vmax_dB is not None else np.nanpercentile(spec_v, 98)
    fig, ax = plt.subplots(figsize=figsize)
    img = ax.pcolormesh(wl_nm, nc_v, spec_v,
                        cmap=cmap_name, vmin=_vmin, vmax=_vmax, shading="auto")
    cbar = fig.colorbar(img, ax=ax, pad=0.01)
    cbar.set_label("Transmisión  (dB)", fontsize=10)
    ax.set_xlabel("Longitud de onda  (nm)"); ax.set_ylabel("$n_{clad}$")
    ax.set_title(
        f"Heatmap espectral drop — {ring_label}  [kappa_03]\n"
        f"Barrido de $n_{{clad}}$ sobre RING_1")
    fig.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR_03 / f"nc_drop{drop_k}_spectrum_heatmap.png")
        fig.savefig(FIGURES_DIR_03 / f"nc_drop{drop_k}_spectrum_heatmap.pdf")
    return fig


def plot_03_all_drop_heatmaps_nc(ncols=4, cmap_name="inferno", save=True):
    wl_nm  = sweep_results_03["wavelengths_m"] * 1e9
    t_drop = sweep_results_03["T_drop_dB"]
    mask   = _valid_mask_03()
    nc_v   = n_cladding[mask]
    nrows  = math.ceil(N_DROPS / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 3.8, nrows * 2.8),
                              sharex=True, sharey=True)
    axes_flat = axes.flatten()
    all_vals  = t_drop[mask].ravel()
    vmin = np.nanpercentile(all_vals, 2); vmax = np.nanpercentile(all_vals, 98)
    im = None
    for k in range(1, N_DROPS + 1):
        ax   = axes_flat[k - 1]
        spec = t_drop[mask, k - 1, :]
        im   = ax.pcolormesh(wl_nm, nc_v, spec,
                             cmap=cmap_name, vmin=vmin, vmax=vmax, shading="auto")
        ax.set_title(f"{DROP_LABELS[k-1]} drop", fontsize=8)
        if k % ncols == 1: ax.set_ylabel("$n_{clad}$", fontsize=7)
        if k > (nrows - 1) * ncols: ax.set_xlabel("λ (nm)", fontsize=7)
        ax.tick_params(labelsize=6)
    for ax in axes_flat[N_DROPS:]: ax.set_visible(False)
    if im is not None:
        fig.colorbar(im, ax=axes_flat[:N_DROPS], shrink=0.6, pad=0.02,
                     label="Transmisión (dB)", fraction=0.015)
    fig.suptitle(
        "Espectros drop — 13 anillos  [kappa_03]\nBarrido de $n_{clad}$ sobre RING_1",
        fontsize=11)
    fig.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR_03 / "nc_all_drop_heatmaps_grid.png", dpi=200)
        fig.savefig(FIGURES_DIR_03 / "nc_all_drop_heatmaps_grid.pdf")
    return fig


def plot_03_power_heatmap_drops_vs_ncladding(figsize=(10, 5),
                                              cmap_name="plasma", save=True):
    mask = _valid_mask_03()
    nc_v = n_cladding[mask]
    p_v  = sweep_results_03["drop_power_dBm"][mask, :].T
    fig, ax = plt.subplots(figsize=figsize)
    img = ax.pcolormesh(nc_v, np.arange(1, N_DROPS + 1), p_v,
                        cmap=cmap_name, shading="auto")
    cbar = fig.colorbar(img, ax=ax, pad=0.02)
    cbar.set_label("Potencia en detector  (dBm)", fontsize=10)
    ax.set_xlabel("Índice de refracción del cladding  $n_{clad}$", fontsize=11)
    ax.set_ylabel("Índice de anillo espectrómetro  (1=RING_2 … 13=RING_14)", fontsize=10)
    ax.set_yticks(np.arange(1, N_DROPS + 1)); ax.set_yticklabels(DROP_LABELS, fontsize=7)
    ax.set_title(
        "Heatmap de potencia — Todos los drops vs $n_{clad}$  [kappa_03]\n"
        r"Color: $P_{det}$ [dBm]  por anillo espectrómetro", fontsize=12)
    fig.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR_03 / "nc_power_heatmap_drops_vs_ncladding.png", dpi=200)
        fig.savefig(FIGURES_DIR_03 / "nc_power_heatmap_drops_vs_ncladding.pdf")
    return fig


def plot_03_resonance_tracking_nc(figsize=(7, 4.5), save=True):
    wl_nm  = sweep_results_03["wavelengths_m"] * 1e9
    T_data = sweep_results_03["T_sensor_through_dB"]
    mask   = _valid_mask_03()
    nc_v   = n_cladding[mask]
    T_v    = T_data[mask, :]
    dip_idx = np.argmin(T_v, axis=1)
    lam_dip = wl_nm[dip_idx]
    coeffs  = np.polyfit(nc_v, lam_dip, 1)
    sens    = coeffs[0]
    r2      = 1.0 - np.sum((lam_dip - np.poly1d(coeffs)(nc_v))**2) / \
                    np.sum((lam_dip - lam_dip.mean())**2)
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(nc_v, lam_dip, s=20, zorder=5,
               color="#2166ac", label="Mínimo de transmisión (dip)")
    ax.plot(nc_v, np.poly1d(coeffs)(nc_v), "r--", lw=1.5,
            label=(f"Ajuste lineal\n"
                   f"$S = \\partial\\lambda / \\partial n_{{clad}}$ "
                   f"= {sens:.2f} nm/RIU\n"
                   f"$R^2$ = {r2:.6f}"))
    ax.set_xlabel("Índice de refracción del cladding  $n_{clad}$", fontsize=12)
    ax.set_ylabel("Longitud de onda de resonancia  (nm)", fontsize=12)
    ax.set_title(
        f"Tracking de resonancia — kappa_03\nSensibilidad: {sens:.2f} nm/RIU  ($n_{{clad}}$)",
        fontsize=12)
    ax.legend(framealpha=0.9, fontsize=9)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR_03 / "nc_resonance_tracking_vs_ncladding.png")
        fig.savefig(FIGURES_DIR_03 / "nc_resonance_tracking_vs_ncladding.pdf")
    log.info(f"[03] Sensibilidad (n_cladding): {sens:.4f} nm/RIU   R²={r2:.8f}")
    return fig

def run(state=None):
    """Execute this step. `state` carries the shared namespace
    between steps (the notebook's old kernel globals + bridge).
    Returns the updated `state`."""
    state = {} if state is None else state
    global FIGURES_DIR_03, HDF5_PATH_03, VERSION_NAME_03, computed_03, f03_1, f03_10, f03_11, f03_12, f03_13, f03_14, f03_15, f03_16, f03_2, f03_3, f03_4, f03_5, f03_6, f03_7, f03_8, f03_9, sweep_results_03, wl_03, wl_nm_03
    globals()['lumapi'] = import_lumapi()
    globals().update(state)

    FIGURES_DIR_03.mkdir(parents=True, exist_ok=True)
    print(f"  Versión kappa_03   : {VERSION_NAME_03}")
    print(f"  HDF5               : {HDF5_PATH_03}")
    print(f"  Figuras            : {FIGURES_DIR_03}")
    sweep_results_03 = run_interconnect_sweep_03(hide_gui=False)
    wl_03          = sweep_results_03["wavelengths_m"]
    wl_nm_03       = wl_03 * 1e9 if wl_03 is not None else None
    computed_03    = sweep_results_03["computed"]
    print(f"\n  [kappa_03] Sweep completo — {computed_03.sum()}/{SWEEP_N_POINTS} pts")
    if wl_03 is not None:
        print(f"  T_sensor_through_dB shape : {sweep_results_03['T_sensor_through_dB'].shape}")
        print(f"  T_drop_dB           shape : {sweep_results_03['T_drop_dB'].shape}")
        print(f"  drop_power_dBm      shape : {sweep_results_03['drop_power_dBm'].shape}")
    print(f"  HDF5                      : {HDF5_PATH_03}")
    if wl_03 is not None and computed_03.sum() > 0:
        # Eje neff
        f03_1  = plot_03_drop_power_vs_neff()
        f03_2  = plot_03_sensor_through_sweep(n_curves=200)
        f03_3  = plot_03_final_through_sweep(n_curves=200)
        f03_4  = plot_03_drop_spectrum_heatmap(drop_k=1)
        f03_5  = plot_03_drop_spectrum_heatmap(drop_k=13)
        f03_6  = plot_03_all_drop_heatmaps()
        f03_7  = plot_03_power_heatmap_drops_vs_neff()
        f03_8  = plot_03_resonance_tracking()
        # Eje n_cladding
        f03_9  = plot_03_drop_power_vs_ncladding()
        f03_10 = plot_03_sensor_through_sweep_nc(n_curves=200)
        f03_11 = plot_03_final_through_sweep_nc(n_curves=200)
        f03_12 = plot_03_drop_spectrum_heatmap_nc(drop_k=1)
        f03_13 = plot_03_drop_spectrum_heatmap_nc(drop_k=13)
        f03_14 = plot_03_all_drop_heatmaps_nc()
        f03_15 = plot_03_power_heatmap_drops_vs_ncladding()
        f03_16 = plot_03_resonance_tracking_nc()
        plt.show()
        print(f"\n  [kappa_03] Figuras → {FIGURES_DIR_03}")
        print(f"  [kappa_03] HDF5   → {HDF5_PATH_03}")
    else:
        print("  [kappa_03] Sin datos disponibles — ejecuta run_interconnect_sweep_03().")

    state.update({k: globals().get(k) for k in [
        'FIGURES_DIR_03', 'HDF5_PATH_03', 'VERSION_NAME_03', 'computed_03', 'f03_1', 'f03_10',
        'f03_11', 'f03_12', 'f03_13', 'f03_14', 'f03_15', 'f03_16',
        'f03_2', 'f03_3', 'f03_4', 'f03_5', 'f03_6', 'f03_7',
        'f03_8', 'f03_9', 'sweep_results_03', 'wl_03', 'wl_nm_03',
    ] if k in globals()})
    return state


if __name__ == "__main__":
    run()
