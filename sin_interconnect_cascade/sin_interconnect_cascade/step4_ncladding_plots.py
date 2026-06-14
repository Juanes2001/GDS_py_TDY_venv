"""
step4_ncladding_plots.py — drop-power / through-port figures on the n_cladding axis.

Presentation layer for step 3: draws the drop-power-vs-n_cladding,
through-port and heatmap figures, reusing the cached sweep results
through get_results()/_valid_mask().
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
    get_results, _valid_mask,
)

from .config import *  # shared platform constants & paths
from .plotting import apply_style, save_fig, make_colorbar
from . import plotting as _plot

# ── Data-contract inputs (injected by main.py via `state`) ──────────
n_cladding = None

apply_style()

# ─────────────────────────────────────────────────────────
#  Module constants / design knobs for this step
# ─────────────────────────────────────────────────────────


def plot_drop_power_vs_ncladding(results=None, figsize=(11, 6),
                                 save: bool = True) -> plt.Figure:
    """
    Versión n_cladding de plot_drop_power_vs_neff (Cell 4, Plot 1).
    Eje X: índice de refracción del cladding (1.33 → 1.37).
    Eje Y: potencia integrada en el detector [dBm].
    13 curvas, una por anillo espectrómetro (RING_2..14).
    """
    if results is None:
        results = get_results()
    mask   = _valid_mask(results)
    nc_v   = n_cladding[mask]
    p_v    = results["drop_power_dBm"][mask, :]   # (n_valid, N_DROPS)

    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=figsize)
    for k in range(N_DROPS):
        ax.plot(nc_v, p_v[:, k],
                color=cmap(k / N_DROPS), lw=1.5,
                marker="o", ms=2.5, alpha=0.85,
                label=DROP_LABELS[k])

    ax.set_xlabel("Índice de refracción del cladding  $n_{clad}$", fontsize=12)
    ax.set_ylabel("Potencia en el detector  (dBm)", fontsize=12)
    ax.set_title(
        "Potencia integrada en drop vs $n_{clad}$  [V3 — ONA multiport]\n"
        r"$P_{det} = P_{src} \cdot \langle|T_{drop}(\lambda)|^2\rangle_\lambda$"
        "   —   13 anillos espectrómetros",
        fontsize=12,
    )
    ax.legend(ncol=3, framealpha=0.88, fontsize=8, loc="best")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "nc_drop_power_vs_ncladding_all.png")
        fig.savefig(FIGURES_DIR / "nc_drop_power_vs_ncladding_all.pdf")
        log.info("Saved → nc_drop_power_vs_ncladding_all.png/pdf")
    return fig


def plot_sensor_through_sweep_nc(results=None, n_curves: int = 200,
                                  figsize=(10, 5), cmap_name: str = "plasma",
                                  save: bool = True) -> plt.Figure:
    """
    Versión n_cladding de plot_sensor_through_sweep (Cell 4, Plot 2).
    Espectros de transmisión ONA input 1 (RING_1 through).
    Colormap codifica n_cladding en lugar de neff.
    """
    if results is None:
        results = get_results()
    wl_nm    = results["wavelengths_m"] * 1e9
    T_data   = results["T_sensor_through_dB"]
    mask     = _valid_mask(results)
    valid_idx = np.where(mask)[0]
    n_sel    = min(n_curves, len(valid_idx))
    sel_idx  = valid_idx[
        np.round(np.linspace(0, len(valid_idx) - 1, n_sel)).astype(int)
    ]
    nc_sel   = n_cladding[sel_idx]

    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=nc_sel.min(), vmax=nc_sel.max())
    sm   = ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])

    fig, ax = plt.subplots(figsize=figsize)
    for idx in sel_idx:
        ax.plot(wl_nm, T_data[idx],
                color=cmap(norm(n_cladding[idx])),
                lw=0.8, alpha=0.70)
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("$n_{clad}$", fontsize=10)
    ax.set_xlabel("Longitud de onda  (nm)")
    ax.set_ylabel("Transmisión  (dB)")
    ax.set_title(
        f"Espectro sensor through — ONA input 1  (RING_1)\n"
        f"({n_sel} curvas,  color = $n_{{clad}}$)"
    )
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig.tight_layout()
    if save:
        stem = f"nc_sensor_through_sweep_{n_sel}curves"
        fig.savefig(FIGURES_DIR / f"{stem}.png")
        fig.savefig(FIGURES_DIR / f"{stem}.pdf")
        log.info(f"Saved → {stem}.png/pdf")
    return fig


def plot_final_through_sweep_nc(results=None, n_curves: int = 200,
                                 figsize=(10, 5), cmap_name: str = "viridis",
                                 save: bool = True) -> plt.Figure:
    """
    Versión n_cladding de plot_final_through_sweep (Cell 4, Plot 3).
    Espectros ONA input 15 (RING_14 through), colormap = n_cladding.
    """
    if results is None:
        results = get_results()
    wl_nm    = results["wavelengths_m"] * 1e9
    T_data   = results["T_final_through_dB"]
    mask     = _valid_mask(results)
    valid_idx = np.where(mask)[0]
    n_sel    = min(n_curves, len(valid_idx))
    sel_idx  = valid_idx[
        np.round(np.linspace(0, len(valid_idx) - 1, n_sel)).astype(int)
    ]
    nc_sel   = n_cladding[sel_idx]

    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=nc_sel.min(), vmax=nc_sel.max())
    sm   = ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])

    fig, ax = plt.subplots(figsize=figsize)
    for idx in sel_idx:
        ax.plot(wl_nm, T_data[idx],
                color=cmap(norm(n_cladding[idx])),
                lw=0.8, alpha=0.70)
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("$n_{clad}$", fontsize=10)
    ax.set_xlabel("Longitud de onda  (nm)")
    ax.set_ylabel("Transmisión  (dB)")
    ax.set_title(
        f"Through final de cascada — ONA input 15  (RING_14)\n"
        f"({n_sel} curvas,  color = $n_{{clad}}$)"
    )
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig.tight_layout()
    if save:
        stem = f"nc_final_through_sweep_{n_sel}curves"
        fig.savefig(FIGURES_DIR / f"{stem}.png")
        fig.savefig(FIGURES_DIR / f"{stem}.pdf")
        log.info(f"Saved → {stem}.png/pdf")
    return fig


def plot_drop_spectrum_heatmap_nc(results=None, drop_k: int = 1,
                                   figsize=(10, 3.5), cmap_name: str = "inferno",
                                   vmin_dB=None, vmax_dB=None,
                                   save: bool = True) -> plt.Figure:
    """
    Versión n_cladding de plot_drop_spectrum_heatmap (Cell 4, Plot 4).
    Heatmap: eje Y = n_cladding (en lugar de neff),  eje X = longitud de onda.
    drop_k 1-based (1=RING_2 drop … 13=RING_14 drop).
    """
    if results is None:
        results = get_results()
    assert 1 <= drop_k <= N_DROPS
    ring_label = DROP_LABELS[drop_k - 1]

    wl_nm  = results["wavelengths_m"] * 1e9
    t_drop = results["T_drop_dB"]
    mask   = _valid_mask(results)
    nc_v   = n_cladding[mask]
    spec_v = t_drop[mask, drop_k - 1, :]

    _vmin = vmin_dB if vmin_dB is not None else np.nanpercentile(spec_v, 2)
    _vmax = vmax_dB if vmax_dB is not None else np.nanpercentile(spec_v, 98)

    fig, ax = plt.subplots(figsize=figsize)
    img = ax.pcolormesh(wl_nm, nc_v, spec_v,
                        cmap=cmap_name, vmin=_vmin, vmax=_vmax, shading="auto")
    cbar = fig.colorbar(img, ax=ax, pad=0.01)
    cbar.set_label("Transmisión  (dB)", fontsize=10)
    ax.set_xlabel("Longitud de onda  (nm)")
    ax.set_ylabel("$n_{clad}$")
    ax.set_title(
        f"Heatmap espectral drop — {ring_label}  (ONA input {drop_k + 1})\n"
        f"Barrido de $n_{{clad}}$ sobre anillo sensor (RING_1)"
    )
    fig.tight_layout()
    if save:
        fname = f"nc_drop{drop_k}_spectrum_heatmap"
        fig.savefig(FIGURES_DIR / f"{fname}.png")
        fig.savefig(FIGURES_DIR / f"{fname}.pdf")
        log.info(f"Saved → {fname}.png/pdf")
    return fig


def plot_all_drop_heatmaps_nc(results=None, ncols: int = 4,
                               cmap_name: str = "inferno",
                               save: bool = True) -> plt.Figure:
    """
    Versión n_cladding de plot_all_drop_heatmaps (Cell 4, Plot 5).
    Grid 4×4 con los 13 heatmaps; eje Y = n_cladding.
    """
    if results is None:
        results = get_results()
    wl_nm  = results["wavelengths_m"] * 1e9
    t_drop = results["T_drop_dB"]
    mask   = _valid_mask(results)
    nc_v   = n_cladding[mask]

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
        ax   = axes_flat[k - 1]
        spec = t_drop[mask, k - 1, :]
        im   = ax.pcolormesh(wl_nm, nc_v, spec,
                             cmap=cmap_name, vmin=vmin, vmax=vmax, shading="auto")
        ax.set_title(f"{DROP_LABELS[k-1]} drop", fontsize=8)
        if k % ncols == 1:
            ax.set_ylabel("$n_{clad}$", fontsize=7)
        if k > (nrows - 1) * ncols:
            ax.set_xlabel("λ (nm)", fontsize=7)
        ax.tick_params(labelsize=6)

    for ax in axes_flat[N_DROPS:]:
        ax.set_visible(False)

    if im is not None:
        fig.colorbar(im, ax=axes_flat[:N_DROPS], shrink=0.6, pad=0.02,
                     label="Transmisión (dB)", fraction=0.015)
    fig.suptitle(
        "Espectros drop — 13 anillos espectrómetros  [V3 — ONA multiport]\n"
        "Barrido de $n_{clad}$ sobre anillo sensor (RING_1)",
        fontsize=11,
    )
    fig.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "nc_all_drop_heatmaps_grid.png", dpi=200)
        fig.savefig(FIGURES_DIR / "nc_all_drop_heatmaps_grid.pdf")
        log.info("Saved → nc_all_drop_heatmaps_grid.png/pdf")
    return fig


def plot_power_heatmap_drops_vs_ncladding(results=None, figsize=(10, 5),
                                           cmap_name: str = "plasma",
                                           save: bool = True) -> plt.Figure:
    """
    Versión n_cladding de plot_power_heatmap_drops_vs_neff (Cell 4, Plot 6).
    Eje X = n_cladding,  eje Y = índice de anillo espectrómetro.
    Color = potencia integrada en el detector [dBm].
    """
    if results is None:
        results = get_results()
    mask  = _valid_mask(results)
    nc_v  = n_cladding[mask]
    p_v   = results["drop_power_dBm"][mask, :].T    # (N_DROPS, n_valid)

    fig, ax = plt.subplots(figsize=figsize)
    img = ax.pcolormesh(nc_v, np.arange(1, N_DROPS + 1), p_v,
                        cmap=cmap_name, shading="auto")
    cbar = fig.colorbar(img, ax=ax, pad=0.02)
    cbar.set_label("Potencia en detector  (dBm)", fontsize=10)
    ax.set_xlabel("Índice de refracción del cladding  $n_{clad}$", fontsize=11)
    ax.set_ylabel("Índice de anillo espectrómetro  (1=RING_2 … 13=RING_14)", fontsize=10)
    ax.set_yticks(np.arange(1, N_DROPS + 1))
    ax.set_yticklabels(DROP_LABELS, fontsize=7)
    ax.set_title(
        "Heatmap de potencia — Todos los drops vs $n_{clad}$  [V3]\n"
        r"Color: $P_{det}$ [dBm]  por anillo espectrómetro",
        fontsize=12,
    )
    fig.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "nc_power_heatmap_drops_vs_ncladding.png", dpi=200)
        fig.savefig(FIGURES_DIR / "nc_power_heatmap_drops_vs_ncladding.pdf")
        log.info("Saved → nc_power_heatmap_drops_vs_ncladding.png/pdf")
    return fig


def plot_resonance_tracking_nc(results=None, figsize=(7, 4.5),
                                save: bool = True) -> plt.Figure:
    """
    Versión n_cladding de plot_resonance_tracking (Cell 4, Plot 7).
    Eje X = n_cladding.  Sensibilidad reportada en nm / RIU de cladding.
    """
    if results is None:
        results = get_results()
    wl_nm  = results["wavelengths_m"] * 1e9
    T_data = results["T_sensor_through_dB"]
    mask   = _valid_mask(results)
    nc_v   = n_cladding[mask]
    T_v    = T_data[mask, :]
    dip_idx = np.argmin(T_v, axis=1)
    lam_dip = wl_nm[dip_idx]

    coeffs = np.polyfit(nc_v, lam_dip, 1)
    sens   = coeffs[0]   # nm / RIU_cladding
    r2     = 1.0 - np.sum((lam_dip - np.poly1d(coeffs)(nc_v))**2) / \
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
    ax.set_ylabel(r"\lambda  (nm)", fontsize=12)
    ax.set_title(
        f"Tracking de resonancia — Sensor through (ONA input 1)\n"
        f"Sensibilidad: {sens:.2f} nm/RIU  ($n_{{clad}}$)",
        fontsize=12,
    )
    ax.legend(framealpha=0.9, fontsize=9)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "nc_resonance_tracking_vs_ncladding.png")
        fig.savefig(FIGURES_DIR / "nc_resonance_tracking_vs_ncladding.pdf")
        log.info("Saved → nc_resonance_tracking_vs_ncladding.png/pdf")
    log.info(f"Sensibilidad (n_cladding): {sens:.4f} nm/RIU   R²={r2:.8f}")
    return fig

def run(state=None):
    """Execute this step. `state` carries the shared namespace
    between steps (the notebook's old kernel globals + bridge).
    Returns the updated `state`."""
    state = {} if state is None else state
    global _res, fig_nc1, fig_nc2, fig_nc3, fig_nc4, fig_nc5, fig_nc6, fig_nc7, fig_nc8, mask_nc, nclad_v, neff_v_nc
    globals().update(state)

    _res = get_results()
    mask_nc   = _valid_mask(_res)
    nclad_v   = n_cladding[mask_nc]          # n_cladding filtrado a puntos válidos
    neff_v_nc = _res["neff_sweep"][mask_nc]  # neff correspondiente (para colorbars)
    fig_nc1 = plot_drop_power_vs_ncladding(_res)
    fig_nc2 = plot_sensor_through_sweep_nc(_res, n_curves=200)
    fig_nc3 = plot_final_through_sweep_nc(_res, n_curves=200)
    fig_nc4 = plot_drop_spectrum_heatmap_nc(_res, drop_k=1)    # RING_2 drop
    fig_nc5 = plot_drop_spectrum_heatmap_nc(_res, drop_k=13)   # RING_14 drop
    fig_nc6 = plot_all_drop_heatmaps_nc(_res)
    fig_nc7 = plot_power_heatmap_drops_vs_ncladding(_res)
    fig_nc8 = plot_resonance_tracking_nc(_res)
    plt.show()
    print(f"\n  Figuras (n_cladding) → {FIGURES_DIR}")

    state.update({k: globals().get(k) for k in [
        '_res', 'fig_nc1', 'fig_nc2', 'fig_nc3', 'fig_nc4', 'fig_nc5',
        'fig_nc6', 'fig_nc7', 'fig_nc8', 'mask_nc', 'nclad_v', 'neff_v_nc',
    ] if k in globals()})
    return state


if __name__ == "__main__":
    run()
