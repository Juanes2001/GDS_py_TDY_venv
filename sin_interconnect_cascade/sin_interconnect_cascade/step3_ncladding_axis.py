"""
step3_ncladding_axis.py — re-express the sensor sweep on a cladding-index (n_cladding) axis.

Maps the swept sensor-ring n_eff back onto the physical aqueous
cladding index so every result can be read as a function of
n_cladding (1.33 -> 1.37), the quantity an experimentalist controls.
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

apply_style()

# ─────────────────────────────────────────────────────────
#  Module constants / design knobs for this step
# ─────────────────────────────────────────────────────────
N_CLADDING_START = 1.33
N_CLADDING_STOP  = 1.37
n_cladding = np.linspace(N_CLADDING_START, N_CLADDING_STOP, SWEEP_N_POINTS)
coeffs_nc   = np.polyfit(n_cladding, SWEEP_NEFF, 1)



def run(state=None):
    """Execute this step. `state` carries the shared namespace
    between steps (the notebook's old kernel globals + bridge).
    Returns the updated `state`."""
    state = {} if state is None else state
    global N_CLADDING_START, N_CLADDING_STOP, ax_nc, coeffs_nc, fig_nc, fit_neff, n_cladding, r2, slope_nc
    globals().update(state)

    slope_nc    = coeffs_nc[0]   # Δn_eff / Δn_cladding  [RIU/RIU]
    fit_neff    = np.poly1d(coeffs_nc)(n_cladding)
    r2          = 1.0 - np.sum((SWEEP_NEFF - fit_neff)**2) / \
                        np.sum((SWEEP_NEFF - SWEEP_NEFF.mean())**2)
    print("=" * 62)
    print("  n_cladding  ↔  n_eff (sensor ring)  —  Parametrización")
    print("=" * 62)
    print(f"  n_cladding  :  {N_CLADDING_START:.4f}  →  {N_CLADDING_STOP:.4f}   ({SWEEP_N_POINTS} pts)")
    print(f"  n_eff       :  {SWEEP_NEFF[0]:.8f}  →  {SWEEP_NEFF[-1]:.8f}")
    print(f"  Δn_eff      :  {SWEEP_NEFF[-1]-SWEEP_NEFF[0]:.6e}  RIU")
    print(f"  Δn_cladding :  {N_CLADDING_STOP-N_CLADDING_START:.4f}  RIU")
    print(f"  Sensibilidad:  dn_eff/dn_clad = {slope_nc:.6f}  RIU/RIU")
    print(f"  R²          :  {r2:.8f}")
    print("=" * 62)
    fig_nc, ax_nc = plt.subplots(figsize=(8, 5))
    ax_nc.plot(
        SWEEP_NEFF, n_cladding,
        color="#2166ac", lw=2.0, marker="o", ms=3.5,
        markevery=10, alpha=0.85,
        label="Datos simulados (MODE/FDTD)",
        zorder=4,
    )
    ax_nc.plot(
        fit_neff, n_cladding,
        color="#d6604d", lw=1.5, ls="--",
        label=(f"Ajuste lineal\n"
               f"$\\partial n_{{eff}} / \\partial n_{{clad}}$ = {slope_nc:.5f}  RIU/RIU\n"
               f"$R^2$ = {r2:.6f}"),
        zorder=3,
    )
    ax_nc.set_xlabel("Índice efectivo del anillo sensor  $n_{eff}$ (TE)", fontsize=12)
    ax_nc.set_ylabel("Índice de refracción del cladding  $n_{clad}$", fontsize=12)
    ax_nc.set_title(
        "Parametrización: $n_{clad}$ vs $n_{eff}$ — Anillo sensor  [SiN 400 nm × 1000 nm]\n"
        f"$n_{{clad}} \\in [{N_CLADDING_START},\\,{N_CLADDING_STOP}]$  —  {SWEEP_N_POINTS} puntos",
        fontsize=12,
    )
    ax_nc.legend(framealpha=0.92, fontsize=9, loc="upper left")
    ax_nc.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax_nc.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax_nc.annotate(
        f"$n_{{clad}}$ = {N_CLADDING_START:.2f}\n$n_{{eff}}$ = {SWEEP_NEFF[0]:.6f}",
        xy=(SWEEP_NEFF[0], n_cladding[0]),
        xytext=(SWEEP_NEFF[0] + 0.00015, n_cladding[0] + 0.001),
        fontsize=8, color="#2166ac",
        arrowprops=dict(arrowstyle="->", color="#2166ac", lw=0.8),
    )
    ax_nc.annotate(
        f"$n_{{clad}}$ = {N_CLADDING_STOP:.2f}\n$n_{{eff}}$ = {SWEEP_NEFF[-1]:.6f}",
        xy=(SWEEP_NEFF[-1], n_cladding[-1]),
        xytext=(SWEEP_NEFF[-1] - 0.00055, n_cladding[-1] - 0.003),
        fontsize=8, color="#2166ac",
        arrowprops=dict(arrowstyle="->", color="#2166ac", lw=0.8),
    )
    fig_nc.tight_layout()
    fig_nc.savefig(FIGURES_DIR / "ncladding_vs_neff_sensor.png", dpi=200)
    fig_nc.savefig(FIGURES_DIR / "ncladding_vs_neff_sensor.pdf")
    plt.show()
    print(f"\n  Guardada → ncladding_vs_neff_sensor.png/pdf  ({FIGURES_DIR})")

    state.update({k: globals().get(k) for k in [
        'N_CLADDING_START', 'N_CLADDING_STOP', 'ax_nc', 'coeffs_nc', 'fig_nc', 'fit_neff',
        'n_cladding', 'r2', 'slope_nc',
    ] if k in globals()})
    return state


if __name__ == "__main__":
    run()
