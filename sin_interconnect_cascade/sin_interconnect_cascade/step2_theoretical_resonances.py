"""
step2_theoretical_resonances.py — analytic resonance / FSR map of the 14 rings versus the simulated sweep.

Pure-analytic post-processing (no INTERCONNECT call): computes each
ring's theoretical resonance comb (lambda_res = n_eff L / m) and FSR
and overlays them on the swept neff grid to verify the spectrometer
staggering across the 10 nm free-spectral range.
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
Line2D = None
T_port1_dB = None
T_port2_dB = None
computed = None
exc = None
v = None
wavelengths_m = None

apply_style()

# ─────────────────────────────────────────────────────────
#  Module constants / design knobs for this step
# ─────────────────────────────────────────────────────────
FSR_DESIGN_NM = np.array([
     9.9590,   # Sensor
    10.0116,   # Spec_00
    10.0161,   # Spec_01
    10.0207,   # Spec_02
    10.0253,   # Spec_03
    10.0299,   # Spec_04
    10.0346,   # Spec_05
    10.0392,   # Spec_06
     9.9650,   # Spec_07
     9.9696,   # Spec_08
     9.9742,   # Spec_09
     9.9783,   # Spec_10  ← "9.9783" interpolado entre 9.9742 y 9.9829
     9.9829,   # Spec_11
     9.9875,   # Spec_12
])



def run(state=None):
    """Execute this step. `state` carries the shared namespace
    between steps (the notebook's old kernel globals + bridge).
    Returns the updated `state`."""
    state = {} if state is None else state
    global FSR_DESIGN_NM, L_m, RING_LABELS, SEP, ax_d, ax_fsr, ax_sp, ax_th, bar, bars, color_i, csv_path, delta, delta_floor_nm, delta_ic_vs_design, delta_ic_vs_theo, delta_near_nm, df, dip1_idx, dip2_idx, fig_d, fig_fsr, fig_sp, fig_th, flag_floor, flag_near, frac_err, fsr_design_nm, fsr_theo_floor_nm, fsr_theo_near_nm, hdr, i, lambda_design_m, lambda_design_nm, lambda_ic_port1_nm, lambda_ic_port2_nm, lambda_theo_floor_m, lambda_theo_floor_nm, lambda_theo_near_m, lambda_theo_near_nm, lbl, ld, legend_elems, lw, m_floor, m_near, m_real, neff_calc, neff_sensor_sweep0, ng_calc, ng_sensor_sweep0, row, rows, sign, star, theo_results, units, val, wl_max, wl_min, wl_nm_ref, x
    globals().update(state)

    RING_LABELS = (
        ["Sensor"] +
        [f"Spec_{i:02d}" for i in range(N_RINGS - 1)]
    )
    neff_sensor_sweep0 = float(SWEEP_NEFF[0])
    ng_sensor_sweep0   = float(SWEEP_NG[0])
    neff_calc = RING_NEFF_TE.copy().astype(float)
    ng_calc   = RING_NG_TE.copy().astype(float)
    neff_calc[0] = neff_sensor_sweep0
    ng_calc[0]   = ng_sensor_sweep0
    L_m = RING_RADIUS_M * 2.0 * math.pi   # circumference [m]  shape (14,)
    lambda_design_m  = RING_LAMBDA_RES_M.copy()
    lambda_design_nm = lambda_design_m * 1e9
    m_real  = (neff_calc * L_m) / lambda_design_m          # flotante
    m_near  = np.round(m_real).astype(int)                 # entero más cercano
    m_floor = np.floor(m_real).astype(int)                 # entero inferior
    lambda_theo_near_m  = (neff_calc * L_m) / m_near
    lambda_theo_floor_m = (neff_calc * L_m) / m_floor
    lambda_theo_near_nm  = lambda_theo_near_m  * 1e9
    lambda_theo_floor_nm = lambda_theo_floor_m * 1e9
    fsr_theo_near_nm  = (lambda_theo_near_nm**2)  / (ng_calc * L_m * 1e9)
    fsr_theo_floor_nm = (lambda_theo_floor_nm**2) / (ng_calc * L_m * 1e9)
    fsr_design_nm = FSR_DESIGN_NM.copy()
    lambda_ic_port1_nm = None
    lambda_ic_port2_nm = None
    try:
        # Usamos el primer punto del sweep (índice 0) — mismo estado que neff_calc[0]
        if computed[0] and wavelengths_m is not None:
            wl_nm_ref = wavelengths_m * 1e9
            dip1_idx  = int(np.argmin(T_port1_dB[0, :]))
            dip2_idx  = int(np.argmin(T_port2_dB[0, :]))
            lambda_ic_port1_nm = float(wl_nm_ref[dip1_idx])
            lambda_ic_port2_nm = float(wl_nm_ref[dip2_idx])
            print(f"  INTERCONNECT (sweep pt 0) — ONA port1 dip : {lambda_ic_port1_nm:.4f} nm")
            print(f"  INTERCONNECT (sweep pt 0) — ONA port2 dip : {lambda_ic_port2_nm:.4f} nm")
        else:
            print("  [AVISO] computed[0] = False o wavelengths_m no disponible.")
            print("          La columna 'IC (ONA p1)' aparecerá como N/A.")
    except Exception as exc:
        print(f"  [AVISO] No se pudo extraer dip de INTERCONNECT: {exc}")
    rows = []
    for i in range(N_RINGS):
        # Diferencias λ_theo − λ_design
        delta_near_nm  = lambda_theo_near_nm[i]  - lambda_design_nm[i]
        delta_floor_nm = lambda_theo_floor_nm[i] - lambda_design_nm[i]

        # Error fraccional del m_real respecto al m_near
        frac_err = m_real[i] - m_near[i]   # cuán "descentrado" está m_real

        row = {
            "Ring"              : RING_LABELS[i],
            "λ_design (nm)"     : round(lambda_design_nm[i], 4),
            "neff"              : round(neff_calc[i], 6),
            "ng"                : round(ng_calc[i], 6),
            "L (µm)"            : round(L_m[i] * 1e6, 4),
            "m_real"            : round(m_real[i], 6),
            "m_near"            : int(m_near[i]),
            "λ_theo_mNear (nm)" : round(lambda_theo_near_nm[i], 4),
            "Δλ_mNear (pm)"     : round(delta_near_nm * 1e3, 2),
            "m_floor"           : int(m_floor[i]),
            "λ_theo_mFloor (nm)": round(lambda_theo_floor_nm[i], 4),
            "Δλ_mFloor (pm)"    : round(delta_floor_nm * 1e3, 2),
            "FSR_design (nm)"   : round(fsr_design_nm[i], 4),
            "FSR_theo_mNear (nm)": round(fsr_theo_near_nm[i], 4),
            "FSR_delta (pm)"    : round((fsr_theo_near_nm[i] - fsr_design_nm[i]) * 1e3, 2),
            "m_frac_err"        : round(frac_err, 6),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df.set_index("Ring", inplace=True)
    SEP = "═" * 130
    print()
    print(SEP)
    print("  TABLA DE COMPARACIÓN TEÓRICA vs DISEÑO  │  14-Ring Cascade  │  SiN 400 nm × 1000 nm")
    print(f"  Anillo sensor (Ring 1): neff = {neff_sensor_sweep0:.6f}  ng = {ng_sensor_sweep0:.6f}  "
          f"(primer punto del sweep; diseño: neff={RING_NEFF_TE[0]:.6f}, ng={RING_NG_TE[0]:.6f})")
    print(SEP)
    hdr = (f"  {'Ring':<10}  {'λ_design':>11}  {'neff':>10}  {'L (µm)':>9}  "
           f"{'m_real':>10}  {'m_near':>7}  {'λ_mNear':>10}  {'Δλ_mNear':>11}  "
           f"{'m_floor':>8}  {'λ_mFloor':>11}  {'Δλ_mFloor':>12}  "
           f"{'FSR_des':>9}  {'FSR_teo':>9}  {'ΔFSR':>8}")
    print(hdr)
    print("  " + "─" * 126)
    units = (f"  {'':10}  {'(nm)':>11}  {'':>10}  {'':>9}  "
             f"{'':>10}  {'':>7}  {'(nm)':>10}  {'(pm)':>11}  "
             f"{'':>8}  {'(nm)':>11}  {'(pm)':>12}  "
             f"{'(nm)':>9}  {'(nm)':>9}  {'(pm)':>8}")
    print(units)
    print("  " + "─" * 126)
    for i in range(N_RINGS):
        lbl  = RING_LABELS[i]
        star = " ◄ SENSOR" if i == 0 else ""
        flag_near  = "  ✓" if abs(df.loc[lbl, "Δλ_mNear (pm)"])  < 500 else "  !"
        flag_floor = "  ✓" if abs(df.loc[lbl, "Δλ_mFloor (pm)"]) < 500 else "  !"
        print(
            f"  {lbl:<10}  "
            f"{df.loc[lbl, 'λ_design (nm)']:>11.4f}  "
            f"{df.loc[lbl, 'neff']:>10.6f}  "
            f"{df.loc[lbl, 'L (µm)']:>9.4f}  "
            f"{df.loc[lbl, 'm_real']:>10.4f}  "
            f"{df.loc[lbl, 'm_near']:>7d}  "
            f"{df.loc[lbl, 'λ_theo_mNear (nm)']:>10.4f}{flag_near}  "
            f"{df.loc[lbl, 'Δλ_mNear (pm)']:>+9.1f} pm  "
            f"{df.loc[lbl, 'm_floor']:>8d}  "
            f"{df.loc[lbl, 'λ_theo_mFloor (nm)']:>11.4f}{flag_floor}  "
            f"{df.loc[lbl, 'Δλ_mFloor (pm)']:>+10.1f} pm  "
            f"{df.loc[lbl, 'FSR_design (nm)']:>9.4f}  "
            f"{df.loc[lbl, 'FSR_theo_mNear (nm)']:>9.4f}  "
            f"{df.loc[lbl, 'FSR_delta (pm)']:>+7.1f} pm"
            f"{star}"
        )
    print()
    print(SEP)
    print("  DIAGNÓSTICO — INTERCONNECT vs Teoría  (anillo sensor, sweep pt 0)")
    print(SEP)
    print(f"  λ_design   (diseño)          : {lambda_design_nm[0]:.4f} nm")
    print(f"  λ_theo     (m_near={m_near[0]:d})      : {lambda_theo_near_nm[0]:.4f} nm   "
          f"Δ = {(lambda_theo_near_nm[0]-lambda_design_nm[0])*1e3:+.2f} pm")
    print(f"  λ_theo     (m_floor={m_floor[0]:d})    : {lambda_theo_floor_nm[0]:.4f} nm   "
          f"Δ = {(lambda_theo_floor_nm[0]-lambda_design_nm[0])*1e3:+.2f} pm")
    print(f"  m_real                       : {m_real[0]:.6f}   (fracción: {m_real[0]-m_near[0]:+.6f})")
    if lambda_ic_port1_nm is not None:
        delta_ic_vs_design = lambda_ic_port1_nm - lambda_design_nm[0]
        delta_ic_vs_theo   = lambda_ic_port1_nm - lambda_theo_near_nm[0]
        print(f"  λ_IC       (ONA port1 dip)   : {lambda_ic_port1_nm:.4f} nm")
        print(f"    → Δ(IC − diseño)           : {delta_ic_vs_design*1e3:+.2f} pm")
        print(f"    → Δ(IC − theo m_near)      : {delta_ic_vs_theo*1e3:+.2f} pm")
        if abs(delta_ic_vs_design) > 0.05:
            print()
            print("  ⚠  SHIFT DETECTADO > 50 pm entre diseño e INTERCONNECT.")
            print("     Posibles causas:")
            print("     1. INTERCONNECT resuelve m = round(m_real) pero con neff re-evaluado")
            print("        en la frecuencia central (dispersión), no en λ_design.")
            print("     2. La propiedad 'frequency' se fija en c/λ_design, pero INTERCONNECT")
            print("        itera internamente hasta convergencia con neff(λ) → shift residual.")
            print("     3. Las pérdidas (101 dB/m) ensanchan la resonancia y desplazan el")
            print("        mínimo aparente de transmisión respecto al centro Lorentziano.")
            print("     4. El valor de m_real no es entero exacto → INTERCONNECT elige el")
            f"        m más cercano, produciendo Δλ = {(lambda_theo_near_nm[0]-lambda_design_nm[0])*1e3:+.2f} pm."
            print(f"        m más cercano: Δλ = {(lambda_theo_near_nm[0]-lambda_design_nm[0])*1e3:+.2f} pm.")
    else:
        print("  λ_IC (ONA port1)             : N/A — sweep no ejecutado aún")
    print(SEP)
    print()
    print("  RESUMEN ESPECTRÓMETROS — Δλ (teoría m_near − diseño) [pm]")
    print("  " + "─" * 65)
    for i in range(N_RINGS):
        lbl   = RING_LABELS[i]
        delta = df.loc[lbl, "Δλ_mNear (pm)"]
        bar   = "█" * int(abs(delta) / 50)
        sign  = "+" if delta >= 0 else "-"
        print(f"  {lbl:<10}  {delta:>+8.1f} pm  {bar}")
    print()
    fig_th, ax_th = plt.subplots(figsize=(12, 5))
    x = np.arange(N_RINGS)
    ax_th.scatter(x, lambda_design_nm,       marker="o", s=60,  zorder=6,
                  color="#2166ac", label="λ diseño")
    ax_th.scatter(x, lambda_theo_near_nm,    marker="^", s=60,  zorder=5,
                  color="#d6604d", label="λ_theo  (m = round)")
    ax_th.scatter(x, lambda_theo_floor_nm,   marker="s", s=40,  zorder=4,
                  color="#4dac26", label="λ_theo  (m = floor)", alpha=0.75)
    if lambda_ic_port1_nm is not None:
        ax_th.axhline(lambda_ic_port1_nm, color="#762a83", lw=1.5, ls="--",
                      label=f"IC ONA p1 dip  ({lambda_ic_port1_nm:.3f} nm)")
    ax_th.set_xticks(x)
    ax_th.set_xticklabels(RING_LABELS, rotation=35, ha="right", fontsize=8)
    ax_th.set_ylabel("Longitud de onda (nm)")
    ax_th.set_xlabel("Anillo")
    ax_th.set_title(
        "Resonancias de diseño vs teóricas por anillo\n"
        "λ_theo = (neff × L) / m    [m = round(neff·L/λ_design) ó floor]"
    )
    ax_th.legend(framealpha=0.9, fontsize=9)
    ax_th.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig_th.tight_layout()
    fig_th.savefig(FIGURES_DIR / "theoretical_lambda_vs_design.png", dpi=200)
    fig_th.savefig(FIGURES_DIR / "theoretical_lambda_vs_design.pdf")
    print(f"  Guardada → theoretical_lambda_vs_design.png/pdf")
    fig_d, ax_d = plt.subplots(figsize=(12, 4))
    bars = ax_d.bar(x, df["Δλ_mNear (pm)"].values,
                    color=["#d6604d" if v >= 0 else "#4393c3"
                           for v in df["Δλ_mNear (pm)"].values],
                    edgecolor="k", linewidth=0.5, zorder=4)
    ax_d.axhline(0, color="k", lw=0.8, ls="-")
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(RING_LABELS, rotation=35, ha="right", fontsize=8)
    ax_d.set_ylabel("Δλ  (pm)  [λ_theo(m_near) − λ_diseño]")
    ax_d.set_title(
        "Shift teórico respecto al diseño por anillo  [Δλ = λ_theo − λ_diseño]\n"
        "Causado por m_real no entero → INTERCONNECT elige m_near"
    )
    for bar, val in zip(bars, df["Δλ_mNear (pm)"].values):
        ax_d.text(bar.get_x() + bar.get_width() / 2, val + (5 if val >= 0 else -15),
                  f"{val:+.0f}", ha="center", va="bottom", fontsize=7)
    ax_d.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig_d.tight_layout()
    fig_d.savefig(FIGURES_DIR / "delta_lambda_theoretical_vs_design.png", dpi=200)
    fig_d.savefig(FIGURES_DIR / "delta_lambda_theoretical_vs_design.pdf")
    print(f"  Guardada → delta_lambda_theoretical_vs_design.png/pdf")
    fig_fsr, ax_fsr = plt.subplots(figsize=(12, 4))
    ax_fsr.scatter(x, fsr_design_nm,     marker="o", s=60, zorder=5,
                   color="#2166ac",  label="FSR diseño (tabla)")
    ax_fsr.scatter(x, fsr_theo_near_nm,  marker="^", s=60, zorder=5,
                   color="#d6604d",  label="FSR teórico  (λ_theo/ng/L)")
    ax_fsr.set_xticks(x)
    ax_fsr.set_xticklabels(RING_LABELS, rotation=35, ha="right", fontsize=8)
    ax_fsr.set_ylabel("FSR  (nm)")
    ax_fsr.set_xlabel("Anillo")
    ax_fsr.set_title("FSR de diseño vs FSR teórico calculado")
    ax_fsr.legend(framealpha=0.9, fontsize=9)
    ax_fsr.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    fig_fsr.tight_layout()
    fig_fsr.savefig(FIGURES_DIR / "fsr_design_vs_theoretical.png", dpi=200)
    fig_fsr.savefig(FIGURES_DIR / "fsr_design_vs_theoretical.pdf")
    print(f"  Guardada → fsr_design_vs_theoretical.png/pdf")
    if lambda_ic_port1_nm is not None and wavelengths_m is not None:
        fig_sp, ax_sp = plt.subplots(figsize=(10, 4))
        wl_nm_ref = wavelengths_m * 1e9
        ax_sp.plot(wl_nm_ref, T_port1_dB[0, :],
                   color="#2166ac", lw=1.4, label="IC ONA port1 (sweep pt 0)")

        # Marcar resonancias teóricas de todos los anillos que caen en la ventana
        wl_min, wl_max = wl_nm_ref.min(), wl_nm_ref.max()
        for i in range(N_RINGS):
            lbl = RING_LABELS[i]
            lw  = lambda_theo_near_nm[i]
            if wl_min <= lw <= wl_max:
                color_i = "#d6604d" if i == 0 else "#4dac26"
                ax_sp.axvline(lw, color=color_i, lw=1.0, ls="--", alpha=0.7)
                ax_sp.text(lw, ax_sp.get_ylim()[0] + 0.5,
                           f"{lbl}\n{lw:.2f}", fontsize=5, ha="center",
                           color=color_i, rotation=90, va="bottom")
            ld = lambda_design_nm[i]
            if wl_min <= ld <= wl_max:
                ax_sp.axvline(ld, color="#762a83", lw=0.8, ls=":", alpha=0.5)

        from matplotlib.lines import Line2D
        legend_elems = [
            Line2D([0], [0], color="#2166ac", lw=1.4, label="IC ONA port1 (sweep 0)"),
            Line2D([0], [0], color="#d6604d", lw=1.0, ls="--", label="λ_theo sensor (m_near)"),
            Line2D([0], [0], color="#4dac26", lw=1.0, ls="--", label="λ_theo espectrómetros (m_near)"),
            Line2D([0], [0], color="#762a83", lw=0.8, ls=":",  label="λ diseño"),
        ]
        ax_sp.legend(handles=legend_elems, fontsize=8, framealpha=0.9, loc="lower right")
        ax_sp.set_xlabel("Longitud de onda (nm)")
        ax_sp.set_ylabel("Transmisión (dB)")
        ax_sp.set_title(
            "Espectro INTERCONNECT (sweep pt 0)  +  resonancias teóricas\n"
            "Líneas discontinuas: λ_theo(m_near)  │  Líneas punteadas: λ_diseño"
        )
        ax_sp.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        fig_sp.tight_layout()
        fig_sp.savefig(FIGURES_DIR / "spectrum_ic_vs_theoretical_resonances.png", dpi=200)
        fig_sp.savefig(FIGURES_DIR / "spectrum_ic_vs_theoretical_resonances.pdf")
        print(f"  Guardada → spectrum_ic_vs_theoretical_resonances.png/pdf")
    csv_path = DATA_DIR / f"{VERSION_NAME}_theoretical_comparison.csv"
    df.to_csv(csv_path, float_format="%.6f")
    print(f"\n  CSV exportado → {csv_path}")
    print(f"  Figuras       → {FIGURES_DIR}")
    print()
    plt.show()
    theo_results = dict(
        lambda_design_nm     = lambda_design_nm,
        lambda_theo_near_nm  = lambda_theo_near_nm,
        lambda_theo_floor_nm = lambda_theo_floor_nm,
        fsr_design_nm        = fsr_design_nm,
        fsr_theo_near_nm     = fsr_theo_near_nm,
        m_real               = m_real,
        m_near               = m_near,
        m_floor              = m_floor,
        neff_used            = neff_calc,
        ng_used              = ng_calc,
        L_m                  = L_m,
        df_table             = df,
    )
    print("  Variable 'theo_results' disponible para celdas posteriores.")

    state.update({k: globals().get(k) for k in [
        'FSR_DESIGN_NM', 'L_m', 'RING_LABELS', 'SEP', 'ax_d', 'ax_fsr',
        'ax_sp', 'ax_th', 'bar', 'bars', 'color_i', 'csv_path',
        'delta', 'delta_floor_nm', 'delta_ic_vs_design', 'delta_ic_vs_theo', 'delta_near_nm', 'df',
        'dip1_idx', 'dip2_idx', 'fig_d', 'fig_fsr', 'fig_sp', 'fig_th',
        'flag_floor', 'flag_near', 'frac_err', 'fsr_design_nm', 'fsr_theo_floor_nm', 'fsr_theo_near_nm',
        'hdr', 'i', 'lambda_design_m', 'lambda_design_nm', 'lambda_ic_port1_nm', 'lambda_ic_port2_nm',
        'lambda_theo_floor_m', 'lambda_theo_floor_nm', 'lambda_theo_near_m', 'lambda_theo_near_nm', 'lbl', 'ld',
        'legend_elems', 'lw', 'm_floor', 'm_near', 'm_real', 'neff_calc',
        'neff_sensor_sweep0', 'ng_calc', 'ng_sensor_sweep0', 'row', 'rows', 'sign',
        'star', 'theo_results', 'units', 'val', 'wl_max', 'wl_min',
        'wl_nm_ref', 'x',
    ] if k in globals()})
    return state


if __name__ == "__main__":
    run()
