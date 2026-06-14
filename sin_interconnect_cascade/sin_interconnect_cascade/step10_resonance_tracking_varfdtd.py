"""
step10_resonance_tracking_varfdtd.py — resonance-tracking overlay of the cascade against the varFDTD reference.

Overlays the circuit-level resonance-tracking curves on the external
varFDTD dataset to validate that the compact INTERCONNECT model
reproduces the full-wave resonance shifts of the sensor ring.
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

apply_style()

# ─────────────────────────────────────────────────────────
#  Module constants / design knobs for this step
# ─────────────────────────────────────────────────────────
ALT_N_CLADDING = np.array([
    1.3300, 1.3321, 1.3342, 1.3363, 1.3384,
    1.3405, 1.3426, 1.3447, 1.3468, 1.3489,
    1.3511, 1.3532, 1.3553, 1.3574, 1.3595,
    1.3616, 1.3637, 1.3658, 1.3679, 1.3700,
])
ALT_LAMBDA_THROUGH_NM = np.array([
    1549.963, 1550.414, 1550.815, 1551.216, 1551.617,
    1552.019, 1552.420, 1552.822, 1553.225, 1553.677,
    1554.080, 1554.483, 1554.936, 1555.339, 1555.743,
    1556.197, 1556.601, 1557.056, 1557.511, 1557.915,
])
ALT_LAMBDA_RES_NM = ALT_LAMBDA_THROUGH_NM.copy()
coeffs_alt = np.polyfit(ALT_N_CLADDING, ALT_LAMBDA_RES_NM, 1)

def plot_resonance_tracking_overlay(
    results=None,
    figsize: tuple = (8, 5.5),
    save: bool = True,
    label_serie1: str = "INTCN  (200 pts)",
    label_serie2: str = "varFDTD (20 pts)",
    color_serie1: str = "#2166ac",
    color_serie2: str = "#e6550d",
    save_stem: str = "nc_resonance_tracking_overlay",
) -> "plt.Figure":
    """
    Superpone dos series de tracking de resonancia en una sola figura.

    Las etiquetas de cada serie se muestran mediante anotaciones con flecha
    directamente sobre cada recta de ajuste, incluyendo la sensibilidad S y R².
    No se usa leyenda convencional.

    Serie 1 — sweep principal (n_cladding × λ_dip extraída de
              T_sensor_through_dB, idéntico a plot_resonance_tracking_nc)

    Serie 2 — dataset alternativo de 20 puntos definido en
              ALT_N_CLADDING / ALT_LAMBDA_RES_NM.

    Parámetros
    ----------
    results      : dict devuelto por get_results(); si None se obtiene automáticamente.
    figsize      : tamaño de la figura en pulgadas.
    save         : si True, guarda PNG y PDF en FIGURES_DIR.
    label_serie1 : etiqueta de la primera recta (texto de la anotación).
    label_serie2 : etiqueta de la segunda recta (texto de la anotación).
    color_serie1 : color hexadecimal Serie 1.
    color_serie2 : color hexadecimal Serie 2.
    save_stem    : nombre base del fichero de salida (sin extensión).

    Retorna
    -------
    fig : matplotlib.figure.Figure
    """
    # ── Carga de resultados del sweep principal ───────────────────────────────
    if results is None:
        results = get_results()

    mask  = _valid_mask(results)
    nc_v  = n_cladding[mask]                           # (n_valid,)
    T_v   = results["T_sensor_through_dB"][mask, :]    # (n_valid, n_wl)
    wl_nm = results["wavelengths_m"] * 1e9             # (n_wl,)

    # ── Extracción del dip (misma lógica que plot_resonance_tracking_nc) ──────
    dip_idx = np.argmin(T_v, axis=1)
    lam_dip = wl_nm[dip_idx]

    # ── Ajuste lineal Serie 1 ─────────────────────────────────────────────────
    coeffs1 = np.polyfit(nc_v, lam_dip, 1)
    sens1   = coeffs1[0]
    fit1    = np.poly1d(coeffs1)(nc_v)
    r2_1    = 1.0 - (
        np.sum((lam_dip - fit1) ** 2) /
        np.sum((lam_dip - lam_dip.mean()) ** 2)
    )

    # ── Ajuste lineal Serie 2 ─────────────────────────────────────────────────
    coeffs2 = coeffs_alt
    sens2   = sens_alt
    r2_2    = r2_alt

    # Rango X unificado para extender la recta de ajuste de Serie 2
    nc_ext = np.array([
        min(nc_v.min(), ALT_N_CLADDING.min()),
        max(nc_v.max(), ALT_N_CLADDING.max()),
    ])

    # ── Figura ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)

    # — Serie 1: scatter + recta de ajuste ────────────────────────────────────
    ax.scatter(
        nc_v, lam_dip,
        s=20, zorder=6,
        color=color_serie1,
    )
    ax.plot(
        nc_v, fit1,
        ls="--", lw=1.8,
        color=color_serie1,
        zorder=5,
    )

    # — Serie 2: scatter + recta de ajuste ────────────────────────────────────
    ax.scatter(
        ALT_N_CLADDING, ALT_LAMBDA_RES_NM,
        s=55, marker="o",
        edgecolors=color_serie2,
        facecolors="none",
        linewidths=1.8,
        zorder=8,
    )
    ax.plot(
        nc_ext, np.poly1d(coeffs2)(nc_ext),
        ls="-.", lw=1.8,
        color=color_serie2,
        zorder=7,
    )

    # ── Anotaciones con flecha — Serie 1 (INTERCONNECT) ──────────────────────
    # Punto de anclaje: 30 % del recorrido de nc_v
    ann1_x = nc_v[int(len(nc_v) * 0.30)]
    ann1_y = float(np.poly1d(coeffs1)(ann1_x))
    # Texto desplazado hacia abajo-izquierda para no solapar la recta
    ann1_tx = ann1_x + 0.028
    ann1_ty = ann1_y - 0
    ax.annotate(
        f"{label_serie1}\n"
        f"$S = {sens1:.2f}$ nm/RIU\n"
        ,
        xy=(ann1_x, ann1_y),
        xytext=(ann1_tx, ann1_ty),
        fontsize=23,
        color=color_serie1,
        fontweight="bold",
        ha="right",
        va="top",
        arrowprops=dict(
            arrowstyle="->",
            color=color_serie1,
            lw=1.4,
            connectionstyle="arc3,rad=-0.25",
        ),
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor=color_serie1,
            alpha=0.88,
            linewidth=1.0,
        ),
        zorder=10,
    )

    # ── Anotaciones con flecha — Serie 2 (varFDTD) ───────────────────────────
    # Punto de anclaje: 68 % del recorrido del array alternativo
    ann2_x = ALT_N_CLADDING[int(len(ALT_N_CLADDING) * 0.68)]
    ann2_y = float(np.poly1d(coeffs2)(ann2_x))
    # Texto desplazado hacia arriba-derecha
    ann2_tx = ann2_x - 0.03
    ann2_ty = ann2_y + 0.55
    ax.annotate(
        f"{label_serie2}\n"
        f"$S = {sens2:.2f}$ nm/RIU\n"
        ,
        xy=(ann2_x, ann2_y),
        xytext=(ann2_tx, ann2_ty),
        fontsize=23,
        color=color_serie2,
        fontweight="bold",
        ha="left",
        va="bottom",
        arrowprops=dict(
            arrowstyle="->",
            color=color_serie2,
            lw=1.4,
            connectionstyle="arc3,rad=0.25",
        ),
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor=color_serie2,
            alpha=0.88,
            linewidth=1.0,
        ),
        zorder=10,
    )

    # ── Etiquetas y título ────────────────────────────────────────────────────
    ax.set_xlabel(
        "$n_{clad}$",
        fontsize=20,
    )
    ax.set_ylabel(
        r"$\lambda$ (nm)",
        fontsize=20,
    )
    ax.set_title(
        "Tracking de resonancia — Sensor through (ONA input 1)\n"
        "Comparación: sweep principal vs dataset alternativo",
        fontsize=12,
    )

    # ── Ticks menores (estilo heredado del notebook) ──────────────────────────
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    fig.tight_layout()

    # ── Guardado ──────────────────────────────────────────────────────────────
    if save:
        fig.savefig(FIGURES_DIR / f"{save_stem}.png", dpi=300, bbox_inches="tight")
        fig.savefig(FIGURES_DIR / f"{save_stem}.pdf",           bbox_inches="tight")
        log.info(f"Saved → {save_stem}.png/pdf  ({FIGURES_DIR})")

    log.info(f"Sensibilidad Serie 1 : {sens1:.4f} nm/RIU   R²={r2_1:.8f}")
    log.info(f"Sensibilidad Serie 2 : {sens2:.4f} nm/RIU   R²={r2_2:.8f}")
    log.info(f"Δ sensibilidad       : {sens2 - sens1:+.4f} nm/RIU")

    return fig

def run(state=None):
    """Execute this step. `state` carries the shared namespace
    between steps (the notebook's old kernel globals + bridge).
    Returns the updated `state`."""
    state = {} if state is None else state
    global ALT_LAMBDA_RES_NM, ALT_LAMBDA_THROUGH_NM, ALT_N_CLADDING, _res_d, coeffs_alt, fig_overlay, fit_alt, r2_alt, sens_alt
    globals().update(state)

    sens_alt    = coeffs_alt[0]                               # nm/RIU
    fit_alt     = np.poly1d(coeffs_alt)(ALT_N_CLADDING)
    r2_alt      = 1.0 - (
        np.sum((ALT_LAMBDA_RES_NM - fit_alt) ** 2) /
        np.sum((ALT_LAMBDA_RES_NM - ALT_LAMBDA_RES_NM.mean()) ** 2)
    )
    print("=" * 65)
    print("  CELL D — Resonance Tracking Overlay  │  2 series")
    print("=" * 65)
    print(f"  Dataset alternativo  :  {len(ALT_N_CLADDING)} puntos")
    print(f"  n_cladding range     :  {ALT_N_CLADDING[0]:.4f}  →  {ALT_N_CLADDING[-1]:.4f}")
    print(f"  λ_res range          :  {ALT_LAMBDA_RES_NM[0]:.3f}  →  {ALT_LAMBDA_RES_NM[-1]:.3f} nm")
    print(f"  Sensibilidad (alt)   :  S = {sens_alt:.2f} nm/RIU    R² = {r2_alt:.6f}")
    print("=" * 65)
    _res_d = get_results()
    fig_overlay = plot_resonance_tracking_overlay(_res_d)
    plt.show()
    print()
    print(f"  Figura guardada → {FIGURES_DIR / 'nc_resonance_tracking_overlay.png'}")
    print(f"                    {FIGURES_DIR / 'nc_resonance_tracking_overlay.pdf'}")

    state.update({k: globals().get(k) for k in [
        'ALT_LAMBDA_RES_NM', 'ALT_LAMBDA_THROUGH_NM', 'ALT_N_CLADDING', '_res_d', 'coeffs_alt', 'fig_overlay',
        'fit_alt', 'r2_alt', 'sens_alt',
    ] if k in globals()})
    return state


if __name__ == "__main__":
    run()
