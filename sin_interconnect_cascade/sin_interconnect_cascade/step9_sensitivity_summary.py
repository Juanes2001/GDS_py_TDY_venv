"""
step9_sensitivity_summary.py — FWHM x ring comparison and the dB/RIU power-sensitivity summary figure.

The headline sensor-performance step: for every ring and every FWHM
variant it locates the operating crossing point and computes the
local power sensitivity dP_det/dn_cladding (dB/RIU), collecting them
into a single comparison figure. Consumes the cached sweep_results
of steps 1/7/8 via `state`.
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
dk = None
ks = None
sweep_results = None
sweep_results_02 = None
sweep_results_03 = None

apply_style()

# ─────────────────────────────────────────────────────────
#  Module constants / design knobs for this step
# ─────────────────────────────────────────────────────────
COMPARE_DROP_INDICES = [3, 4]
RING_COLORS = {
    3: "#2166ac",    # RING_4 — azul
    4: "#1a9641",    # RING_5 — verde
}

def _get_nc_power(results: dict, drop_k: int):
    """Retorna (nc_valid, p_valid) filtrando solo puntos computados."""
    mask = results["computed"].astype(bool)
    nc_v = n_cladding[mask]
    p_v  = results["drop_power_dBm"][mask, drop_k - 1]
    return nc_v, p_v


def _check_results_available() -> bool:
    """Verifica que los tres conjuntos de resultados tienen datos válidos."""
    ok = True
    for name, res in [
        ("sweep_results   (κ_500)", sweep_results),
        ("sweep_results_03 (κ_300)", sweep_results_03),
        ("sweep_results_02 (κ_100)", sweep_results_02),
    ]:
        wl     = res.get("wavelengths_m")
        n_comp = int(res["computed"].sum())
        if wl is None or n_comp == 0:
            log.warning(f"  [CELL C] Sin datos: {name}")
            ok = False
        else:
            log.info(f"  [CELL C] OK — {name}  ({n_comp} pts computados)")
    return ok


def _find_best_crossing(nc_arr, p_a, p_b):
    """
    Encuentra el cruce más prominente entre las curvas p_a y p_b
    (ambas definidas en nc_arr con los mismos puntos).

    "Más prominente" = mayor potencia media en el entorno del cruce,
    que corresponde al punto de operación con mejor SNR.

    Física:
      El cruce entre P_RING4 y P_RING5 en κ_500 es el "punto de cuadratura"
      donde el sensor tiene máxima sensibilidad diferencial de intensidad.
      En ese punto la señal diferencial (P4 − P5) cambia más rápidamente
      con n_cladding, aunque su valor absoluto sea cero.

    Retorna
    -------
    nc_cross : float | None   n_cladding en el cruce
    p_cross  : float | None   P_det [dBm] interpolada en el cruce
    """
    diff      = p_a - p_b
    best_nc   = None
    best_p    = None
    best_pavg = -np.inf

    for i in range(len(diff) - 1):
        if diff[i] * diff[i + 1] < 0:                  # cambio de signo → cruce
            frac    = abs(diff[i]) / (abs(diff[i]) + abs(diff[i + 1]) + 1e-30)
            nc_x    = float(nc_arr[i]  + frac * (nc_arr[i + 1]  - nc_arr[i]))
            p4_x    = float(p_a[i]     + frac * (p_a[i + 1]     - p_a[i]))
            p5_x    = float(p_b[i]     + frac * (p_b[i + 1]     - p_b[i]))
            p_avg_i = 0.5 * (float(p_a[i]) + float(p_b[i]) +
                              float(p_a[i + 1]) + float(p_b[i + 1])) / 2.0
            if p_avg_i > best_pavg:
                best_pavg = p_avg_i
                best_nc   = nc_x
                best_p    = 0.5 * (p4_x + p5_x)

    return best_nc, best_p


def _local_sensitivity_dBperRIU(nc_arr, p_arr, n_cross, half_window=8):
    """
    Sensibilidad local  ∂P/∂n_cladding  [dB/RIU]  en n_cross.

    Usa regresión lineal (polyfit grado 1) sobre los 2*half_window+1 puntos
    más próximos a n_cross.  La ventana estrecha captura la pendiente real
    del flanco de resonancia sin promediar en exceso.

    Retorna
    -------
    slope : float   [dB/RIU]   (positivo = subida, negativo = bajada)
    """
    idx_c = int(np.argmin(np.abs(nc_arr - n_cross)))
    i0    = max(0, idx_c - half_window)
    i1    = min(len(nc_arr), idx_c + half_window + 1)
    if i1 - i0 < 2:
        return 0.0
    coeffs = np.polyfit(nc_arr[i0:i1], p_arr[i0:i1], 1)
    return float(coeffs[0])    # [dBm/RIU] ≡ [dB/RIU] para potencias en dBm


def _draw_sensitivity_indicator(
    ax,
    nc_cross, p_cross,
    slope, color, ring_label,
    dx=0.001,
    fontsize=11,
    zorder=16,
    # Posición del texto en fracción del área de ejes (0-1).
    # axes fraction garantiza que la caja siempre quede DENTRO del plano
    # independientemente de dónde caiga el punto de cruce en datos.
    txt_frac_up=(0.74, 0.80),   # subida  (slope > 0) — zona superior-derecha
    txt_frac_dn=(0.20, 0.80),   # bajada  (slope < 0) — zona inferior-izquierda
):
    """
    Dibuja el indicador gráfico de sensibilidad en (nc_cross, p_cross).

    El indicador consta de:
      1. Segmento tangente  centrado en el cruce (longitud 2·dx en n_clad).
      2. Triángulo escalera  (base horizontal + lado vertical), acotando
         visualmente Δn vs ΔP del flanco.
      3. Marcador diamante  en el punto exacto del cruce.
      4. Anotación con flecha cuya PUNTA apunta al cruce (coordenadas datos)
         y cuyo TEXTO está anclado en fracción de ejes, garantizando que
         siempre quede dentro del plano y que las dos etiquetas (subida y
         bajada) estén bien separadas en esquinas opuestas.

    Parámetros
    ----------
    ax           : matplotlib.axes.Axes
    nc_cross     : float   n_cladding en el cruce
    p_cross      : float   P_det [dBm] en el cruce
    slope        : float   ∂P/∂n_clad [dB/RIU]
    color        : str     color hexadecimal del anillo
    ring_label   : str     nombre del anillo ("RING_4" / "RING_5")
    dx           : float   semiancho del indicador [n_cladding]
    fontsize     : int     tamaño de fuente para la etiqueta
    zorder       : int     z-order de renderizado
    txt_frac_up  : (x,y)   posición en axes fraction para slope > 0
    txt_frac_dn  : (x,y)   posición en axes fraction para slope < 0
    """
    # ── Coordenadas del segmento tangente ────────────────────────────────────
    x_l = nc_cross - dx
    x_r = nc_cross + dx
    y_l = p_cross + slope * (-dx)    # y en el extremo izquierdo
    y_r = p_cross + slope * (+dx)    # y en el extremo derecho

    # ── 1. Segmento tangente ──────────────────────────────────────────────────
    ax.plot([x_l, x_r], [y_l, y_r],
            color=color, lw=3.0, solid_capstyle="round", zorder=zorder)

    # ── 2. Triángulo escalera: base horizontal + lado vertical ────────────────
    ax.plot([x_l, x_r], [y_l, y_l],
            color=color, lw=1.0, ls="--", alpha=0.55, zorder=zorder - 1)
    ax.plot([x_r, x_r], [y_l, y_r],
            color=color, lw=1.0, ls="--", alpha=0.55, zorder=zorder - 1)

    # ── 3. Marcador diamante en el cruce ──────────────────────────────────────
    ax.scatter([nc_cross], [p_cross],
               s=80, marker="D", color=color,
               edgecolors="white", linewidth=0.9, zorder=zorder + 1)

    # ── 4. Etiqueta con sensibilidad ──────────────────────────────────────────
    flanco   = "subida  (+)" if slope >= 0 else "bajada  (−)"
    sign_str = "+" if slope >= 0 else "−"
    lbl = (f"{ring_label}\n"
           f"$S =$ {sign_str}{abs(slope):.0f} dB/RIU\n"
           )

    # Seleccionar posición y curvatura de flecha según el flanco
    if slope >= 0:     # SUBIDA — texto en zona superior-derecha del plano
        txt_frac = txt_frac_up
        va, ha   = "center", "left"
        rad      = +0.35
    else:              # BAJADA — texto en zona inferior-izquierda del plano
        txt_frac = txt_frac_dn
        va, ha   = "center", "right"
        rad      = +0.35

    # La punta de la flecha está en coordenadas de DATOS (el cruce exacto).
    # El texto está en fracción de EJES — siempre dentro del área del plano.
    ax.annotate(
        lbl,
        xy=(nc_cross, p_cross),          # punta de flecha → cruce (data coords)
        xycoords="data",
        xytext=txt_frac,                  # caja de texto   → axes fraction
        textcoords="axes fraction",
        fontsize=fontsize,
        color=color,
        fontweight="bold",
        ha=ha,
        va=va,
        arrowprops=dict(
            arrowstyle="->",
            color=color,
            lw=1.2,
            connectionstyle=f"arc3,rad={rad}",
        ),
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor=color,
            alpha=0.93,
            linewidth=1.0,
        ),
        zorder=zorder + 2,
    )


def plot_all_rings_all_fwhm_single_figure(
    figsize=(11, 6.5),
    save: bool = True,
) -> plt.Figure:
    """
    Una sola figura con 6 curvas P_det [dBm] vs n_cladding (2 anillos × 3 FWHM).

    Código visual doble:
      • Color      → identifica el anillo espectrómetro (RING_4 / RING_5)
      • Estilo     → identifica la variante de acoplamiento / FWHM

    Sensibilidad dB/RIU (solo κ_500, sweep_results original):
      1. Busca el cruce entre P_RING4 y P_RING5 (punto de cuadratura).
      2. Calcula ∂P/∂n_clad localmente para cada curva en ese punto.
      3. Dibuja para cada curva:
            – segmento tangente centrado en el cruce
            – triángulo escalera mostrando ΔP/Δn visualmente
            – etiqueta con el valor S y el tipo de flanco (subida/bajada)
      Las curvas κ_300 y κ_100 se trazan sin ninguna modificación.

    Leyenda en dos bloques:
      Bloque superior  → color por anillo   (2 proxies)
      Bloque inferior  → estilo por FWHM    (3 proxies)
    """
    fig, ax = plt.subplots(figsize=figsize)

    # ── Trazar las 6 curvas (3 FWHM × 2 anillos) ─────────────────────────────
    for ks in KAPPA_STYLES:
        for dk in COMPARE_DROP_INDICES:
            nc_v, p_v = _get_nc_power(ks["results"], dk)
            me = max(1, len(nc_v) // ks["markevery_frac"])
            ax.plot(
                nc_v, p_v,
                color     = RING_COLORS[dk],
                ls        = ks["ls"],
                lw        = ks["lw"],
                marker    = ks["marker"],
                ms        = ks["ms"],
                markevery = me,
                alpha     = ks["alpha"],
                zorder    = ks["zorder"],
            )

    # ─────────────────────────────────────────────────────────────────────────
    # ★ SENSIBILIDAD dB/RIU — solo κ_500  ★
    # ─────────────────────────────────────────────────────────────────────────
    dk_4 = COMPARE_DROP_INDICES[0]    # drop_k=3 → RING_4
    dk_5 = COMPARE_DROP_INDICES[1]    # drop_k=4 → RING_5

    nc_4, p_4 = _get_nc_power(sweep_results, dk_4)    # κ_500 RING_4
    nc_5, p_5 = _get_nc_power(sweep_results, dk_5)    # κ_500 RING_5

    nc_cross, p_cross = _find_best_crossing(nc_4, p_4, p_5)

    if nc_cross is not None:
        # ── Pendientes locales en el punto de cruce ───────────────────────────
        S_4 = _local_sensitivity_dBperRIU(nc_4, p_4, nc_cross, half_window=8)
        S_5 = _local_sensitivity_dBperRIU(nc_5, p_5, nc_cross, half_window=8)

        # ── Línea vertical punteada en n_cross (referencia visual) ────────────
        ax.axvline(nc_cross, color="silver", lw=0.9, ls=":", alpha=0.65, zorder=1)

        # ── Marcador estrella en el punto de cruce ────────────────────────────
        ax.scatter(
            [nc_cross], [p_cross],
            s=160, marker="*", color="gold",
            edgecolors="#444444", linewidth=0.7,
            zorder=19,
        )

        # ── Indicadores de sensibilidad para RING_4 y RING_5 ─────────────────
        # txt_frac_up/dn: posición del texto en fracción de ejes (0-1).
        # Las posiciones por defecto colocan las etiquetas en esquinas opuestas
        # del plano (superior-derecha para subida, inferior-izquierda para
        # bajada), garantizando separación y visibilidad dentro del área.
        # Ajusta txt_frac_up / txt_frac_dn si las curvas de tu datos específico
        # requieren reposicionamiento (p.ej. si el cruce cae muy a la izquierda).
        _draw_sensitivity_indicator(
            ax, nc_cross, p_cross,
            slope        = S_4,
            color        = RING_COLORS[dk_4],
            ring_label   = DROP_LABELS[dk_4 - 1],
            dx           = 0.001,
            fontsize     = 13,
            zorder       = 16,
            txt_frac_up  = (0.74, 0.80),   # no usado para S_4 (esperado bajada)
            txt_frac_dn  = (0.20, 0.90),   # zona inferior-izquierda
        )
        _draw_sensitivity_indicator(
            ax, nc_cross, p_cross,
            slope        = S_5,
            color        = RING_COLORS[dk_5],
            ring_label   = DROP_LABELS[dk_5 - 1],
            dx           = 0.001,
            fontsize     = 13,
            zorder       = 16,
            txt_frac_up  = (0.74, 0.74),   # zona superior-derecha
            txt_frac_dn  = (0.18, 0.22),   # no usado para S_5 (esperado subida)
        )

        # ── Resumen numérico en consola ───────────────────────────────────────
        f4 = "subida (+)" if S_4 >= 0 else "bajada (−)"
        f5 = "subida (+)" if S_5 >= 0 else "bajada (−)"
        print(f"\n  ┌─ Sensibilidad en punto de corte κ_500 (FWHM 500 pm) ──────────┐")
        print(f"  │  n_cladding cruce  =  {nc_cross:.5f}  RIU                         │")
        print(f"  │  P_det en el cruce =  {p_cross:.2f}  dBm                         │")
        print(f"  │  S ({DROP_LABELS[dk_4-1]})     =  {S_4:+.1f}  dB/RIU   [{f4}]    │")
        print(f"  │  S ({DROP_LABELS[dk_5-1]})     =  {S_5:+.1f}  dB/RIU   [{f5}]    │")
        print(f"  └───────────────────────────────────────────────────────────────────┘")
        log.info(
            f"[CELL C] Cruce κ_500: n={nc_cross:.5f}  P={p_cross:.2f} dBm  "
            f"S_{DROP_LABELS[dk_4-1]}={S_4:+.1f}  S_{DROP_LABELS[dk_5-1]}={S_5:+.1f} dB/RIU"
        )
    else:
        # Sin cruce detectado — solo aviso; las curvas se trazan igual
        log.warning(
            f"[CELL C] No se encontró punto de corte entre "
            f"{DROP_LABELS[dk_4-1]} y {DROP_LABELS[dk_5-1]} en κ_500.  "
            f"Las curvas no se cruzan en el rango n_clad=[{n_cladding[0]:.3f},{n_cladding[-1]:.3f}]."
        )
        print(f"\n  [AVISO CELL C] Sin cruce entre RING_4 y RING_5 en κ_500. "
              f"Las curvas se trazaron sin indicadores de sensibilidad.")

    # ── Etiquetas de ejes ─────────────────────────────────────────────────────
    ax.set_xlabel(
        "$n_{clad}$",
        fontsize=18,
        labelpad=8,
    )
    ax.set_ylabel(
        "$P_{det}$  (dB)",
        fontsize=18,
        labelpad=8,
    )
    ax.set_title(
        "Respuesta de potencia drop vs $n_{clad}$"
        "  —  6 curvas  (2 anillos  ×  3 variantes FWHM)\n"
        r"$P_{det} = P_{src} \cdot \langle|T_{drop}(\lambda)|^2\rangle_\lambda$"
        "   [V3 — ONA multiport]\n"
        r"$\partial P_{\rm det}/\partial n_{\rm clad}$ [dB/RIU]"
        r" calculada en el punto de corte κ$_{500}$"
        r"  (RING$_4$ $\cap$ RING$_5$)",
        fontsize=11,
        pad=10,
    )

    # ── Formato menor de ejes ─────────────────────────────────────────────────
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(True, alpha=0.30, linewidth=0.5)

    # ── Leyenda bloque 1: ANILLO (color) ──────────────────────────────────────
    handles_rings = [
        mlines.Line2D(
            [], [],
            color  = RING_COLORS[dk],
            lw     = 2.2,
            ls     = "-",
            marker = "o",
            ms     = 10,
            label  = DROP_LABELS[dk - 1],
        )
        for dk in COMPARE_DROP_INDICES
    ]

    # ── Leyenda bloque 2: FWHM / κ (estilo de línea) ─────────────────────────
    handles_kappa = [
        mlines.Line2D(
            [], [],
            color  = "black",
            lw     = ks["lw"],
            ls     = ks["ls"],
            marker = ks["marker"],
            ms     = 10,
            label  = ks["label"],
        )
        for ks in KAPPA_STYLES
    ]

    # Leyenda 1 (anillos) — esquina superior derecha
    leg1 = ax.legend(
        handles        = handles_rings,
        title_fontsize = 9,
        fontsize       = 15,
        loc            = "upper right",
        framealpha     = 0.94,
        edgecolor      = "#AAAAAA",
    )
    ax.add_artist(leg1)

    # Leyenda 2 (FWHM / κ) — esquina inferior derecha
    ax.legend(
        handles        = handles_kappa,
        title_fontsize = 9,
        fontsize       = 15,
        loc            = "lower right",
        framealpha     = 0.94,
        edgecolor      = "#AAAAAA",
    )

    fig.tight_layout()

    # ── Guardado ──────────────────────────────────────────────────────────────
    if save:
        fname_base = "power_vs_ncladding_rings45_sensitivity_k500"
        fig.savefig(
            FIGURES_COMPARE_DIR / f"{fname_base}.png",
            dpi=300, bbox_inches="tight",
        )
        fig.savefig(
            FIGURES_COMPARE_DIR / f"{fname_base}.pdf",
            bbox_inches="tight",
        )
        # Copia rápida en FIGURES_DIR principal
        fig.savefig(
            FIGURES_DIR / f"{fname_base}.png",
            dpi=300, bbox_inches="tight",
        )
        log.info(f"Saved → {fname_base}.png/pdf")
        log.info(f"        {FIGURES_COMPARE_DIR}")

    return fig

def run(state=None):
    """Execute this step. `state` carries the shared namespace
    between steps (the notebook's old kernel globals + bridge).
    Returns the updated `state`."""
    state = {} if state is None else state
    global COMPARE_DROP_INDICES, FIGURES_COMPARE_DIR, KAPPA_STYLES, RING_COLORS, fig_rings45
    globals().update(state)

    KAPPA_STYLES = [
        {
            "label"          : r"$\kappa_{500}$ (FWHM 500 pm)",
            "results"        : sweep_results,          # sweep original (Cell 3)
            "ls"             : "-",
            "lw"             : 2.2,
            "marker"         : "o",
            "ms"             : 7.0,
            "markevery_frac" : 25,
            "alpha"          : 0.95,
            "zorder"         : 4,
        },
        {
            "label"          : r"$\kappa_{300}$  (FWHM 300 pm)",
            "results"        : sweep_results_03,        # kappa_03 (Cell A)
            "ls"             : "--",
            "lw"             : 1.8,
            "marker"         : "s",
            "ms"             : 7,
            "markevery_frac" : 25,
            "alpha"          : 0.90,
            "zorder"         : 3,
        },
        {
            "label"          : r"$\kappa_{100}$  (FWHM 100 pm)",
            "results"        : sweep_results_02,        # kappa_02 (Cell B)
            "ls"             : "-.",
            "lw"             : 1.6,
            "marker"         : "^",
            "ms"             : 7.0,
            "markevery_frac" : 25,
            "alpha"          : 0.85,
            "zorder"         : 2,
        },
    ]
    FIGURES_COMPARE_DIR = DATA_DIR / "figures_comparison"
    FIGURES_COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("  CELL C — Comparación FWHM  │  color=anillo  │  estilo=κ")
    print("  RINGS   :  RING_4 (drop_k=3)   y   RING_5 (drop_k=4)")
    print("  SENSIBILIDAD dB/RIU: κ_500 en punto de corte RING_4 ∩ RING_5")
    print("  κ_300 y κ_100: solo se trazan, sin indicadores de sensibilidad")
    print("=" * 72)
    print(f"  Anillos    : {[DROP_LABELS[dk-1] for dk in COMPARE_DROP_INDICES]}")
    print(f"  Colores    : {[RING_COLORS[dk] for dk in COMPARE_DROP_INDICES]}")
    print(f"  κ variantes: {[ks['label'] for ks in KAPPA_STYLES]}")
    print(f"  Salida     : {FIGURES_COMPARE_DIR}")
    print()
    if _check_results_available():
        fig_rings45 = plot_all_rings_all_fwhm_single_figure(figsize=(11, 6.5), save=True)
        plt.show()
        print()
        print(f"  Figura guardada → {FIGURES_COMPARE_DIR}")
        print(f"  Copia rápida   → {FIGURES_DIR}")
    else:
        print("\n  ⚠  Faltan datos.  Ejecuta las celdas A y/o B primero.")

    state.update({k: globals().get(k) for k in [
        'COMPARE_DROP_INDICES', 'FIGURES_COMPARE_DIR', 'KAPPA_STYLES', 'RING_COLORS', 'fig_rings45',
    ] if k in globals()})
    return state


if __name__ == "__main__":
    run()
