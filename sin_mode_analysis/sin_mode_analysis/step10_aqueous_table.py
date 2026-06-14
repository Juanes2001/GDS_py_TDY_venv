"""
step10_aqueous_table.py — render the aqueous-index sweep summary table.

Presentation-only: renders the aqueous-index sweep DataFrame as a
publication-quality PNG/PDF table.
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
AIS_BEND_RADIUS_UM = None
AIS_CSV_NAME = None
AIS_DELTA_LAM_NM = None
AIS_LAM0_NM = None
AIS_N_POINTS = None
AIS_WG_HEIGHT_NM = None
AIS_WG_WIDTH_NM = None
Rectangle = None
ais_S_lam_pm_RIU = None
ais_S_neff = None
ais_S_ng = None
ais_n_aq_valid = None
ais_neff = None
ais_ng = None
ais_te_frac = None
f = None

apply_style()

# ─────────────────────────────────────────────────────────
#  Module constants / design knobs for this step
# ─────────────────────────────────────────────────────────
TABLE_ROW_STEP   = "auto"
TABLE_SHOW_ENDPOINTS = True
TABLE_DPI        = 180        # dots per inch  (180 gives crisp A3-style output)
TABLE_SAVE_PDF   = True       # also save a vector PDF alongside the PNG
TABLE_PNG_NAME   = ""         # e.g. "my_table.png" or "" for auto
TABLE_PDF_NAME   = ""
_C = {
    "bg":        "#0F1923",    # figure background
    "meta_bg":   "#0A1520",    # title / metadata strip background
    "head_bg":   "#1C3A50",    # column header background
    "foot_bg":   "#0D1E2B",    # footer background
    "row_odd":   "#172330",    # odd data row
    "row_even":  "#16202B",    # even data row
    "row_hi":    "#1A2E40",    # first / last row highlight
    "border":    "#1E3A52",    # cell border lines
    "accent":    "#00D4FF",    # cyan — header text, frame, accent lines
    "accent2":   "#FF6B35",    # orange — n_aq column + "contaminated" label
    "text":      "#E8F4FD",    # primary data text (n_eff, n_g)
    "dim":       "#7EA8C4",    # dimmed text (TE frac, metadata)
    "mono":      "#7ECFEA",    # sensitivity columns
    "naq":       "#D4935A",    # n_aq column (non-highlighted rows)
    "green":     "#2ECC71",    # statistics bar accent
}

def _hline(y, color=_C["border"], lw=0.5, z=2):
    ax.plot([0, FIG_W], [y, y], color=color, lw=lw, zorder=z)


def _vline(x, y_bot, y_top, color=_C["border"], lw=0.4, z=2):
    ax.plot([x, x], [y_bot, y_top], color=color, lw=lw, zorder=z)


def _fmt_array(arr, var_name, decimals, per_line, indent):
    """
    Format a 1-D array as a multi-line Python np.array([...]) assignment.

    Parameters
    ----------
    arr      : 1-D array-like   values to print
    var_name : str              left-hand-side variable name
    decimals : int              decimal places per float
    per_line : int              values per source line
    indent   : str              whitespace prefix for value lines

    Returns
    -------
    str  —  valid Python source, copy-paste ready
    """
    fmt_str = f"{{:.{decimals}f}}"
    vals    = [fmt_str.format(float(v)) for v in arr]
    lines   = []
    for _s in range(0, len(vals), per_line):
        chunk = vals[_s : _s + per_line]
        lines.append(indent + ", ".join(chunk) + ",")
    return f"{var_name} = np.array([\n" + "\n".join(lines) + "\n])"

def run(state=None):
    """Execute this step. `state` carries the shared namespace
    between steps (the notebook's old kernel globals + bridge).
    Returns the updated `state`."""
    state = {} if state is None else state
    global ARRAY_INDENT, ARRAY_NAQ_DECIMALS, ARRAY_NEFF_DECIMALS, ARRAY_VALUES_PER_LINE, FIG_H, FIG_W, FOOT_H, HEAD_H, META_H, PAD, ROW_H, TABLE_DPI, TABLE_PDF_NAME, TABLE_PNG_NAME, TABLE_ROW_STEP, TABLE_SAVE_PDF, TABLE_SHOW_ENDPOINTS, _C, _COL_HEADERS, _CUT, _S_lam_mean, _S_lam_std, _S_neff_mean, _S_ng_mean, _bg, _col, _col_centers, _col_fracs, _col_widths, _cw, _cx, _daq_span, _dneff_total, _dng_total, _fc, _foot_text, _fs, _fw, _header, _i, _idxs, _is_first, _is_last, _j, _meta_line, _n_rows, _n_valid, _pdf_path, _pdf_stem, _png_path, _png_stem, _ri, _row, _rows, _step, _str_naq, _str_neff, _str_ng, _val, _w, _x, _y_accent, _y_data_bot, _y_data_top, _y_foot_top, _y_head_bot, _y_head_top, _y_meta_bot, _y_meta_top, _yb, _ym, _yt, ax, fig
    globals().update(state)

    _n_valid = len(ais_n_aq_valid)
    if TABLE_ROW_STEP == "auto":
        _step = max(1, int(np.ceil(_n_valid / 20)))
    else:
        _step = int(TABLE_ROW_STEP)
    _idxs = list(range(0, _n_valid, _step))
    if TABLE_SHOW_ENDPOINTS:
        if 0 not in _idxs:
            _idxs.insert(0, 0)
        if (_n_valid - 1) not in _idxs:
            _idxs.append(_n_valid - 1)
    _idxs = sorted(set(_idxs))   # deduplicate and sort
    _rows = []
    for _i in _idxs:
        _rows.append([
            f"{ais_n_aq_valid[_i]:.6f}",           # n_aq
            f"{ais_neff[_i]:.10f}",                 # n_eff  (10 decimal places)
            f"{ais_ng[_i]:.10f}",                   # n_g    (10 decimal places)
            f"{ais_te_frac[_i]:.4f}",               # TE fraction
            f"{ais_S_neff[_i]:.6f}",                # ∂n_eff/∂n_aq
            f"{ais_S_ng[_i]:.6f}",                  # ∂n_g/∂n_aq
            f"{ais_S_lam_pm_RIU[_i]:.2f}",          # ∂λ_res/∂n_aq  [pm/RIU]
        ])
    _n_rows = len(_rows)
    _COL_HEADERS = [
        "n_aq",
        "n_eff",
        "n_g",
        "TE frac",
        "∂n_eff/∂n_aq\n[RIU⁻¹]",
        "∂n_g/∂n_aq\n[RIU⁻¹]",
        "∂λres/∂n_aq\n[pm/RIU]",
    ]
    print(f"Table: {_n_rows} rows (step = {_step})  × {len(_COL_HEADERS)} columns")
    FIG_W   = 20.0            # figure width in inches
    ROW_H   = 0.42            # height per data row
    HEAD_H  = 0.80            # height of the column header row
    META_H  = 1.10            # height of the title / metadata strip
    FOOT_H  = 0.55            # height of the statistics footer
    PAD     = 0.20            # outer vertical padding
    FIG_H   = META_H + HEAD_H + _n_rows * ROW_H + FOOT_H + PAD
    _col_fracs = [0.095, 0.148, 0.148, 0.078, 0.145, 0.145, 0.241]
    assert abs(sum(_col_fracs) - 1.0) < 1e-9, \
        f"Column fractions sum to {sum(_col_fracs):.6f}, must be 1.0"
    _col_widths  = [f * FIG_W for f in _col_fracs]
    _col_centers = []
    _x = 0.0
    for _w in _col_widths:
        _col_centers.append(_x + _w / 2.0)
        _x += _w
    _y_foot_top  = FOOT_H + PAD / 2
    _y_data_bot  = _y_foot_top
    _y_data_top  = _y_data_bot + _n_rows * ROW_H
    _y_head_bot  = _y_data_top
    _y_head_top  = _y_head_bot + HEAD_H
    _y_meta_bot  = _y_head_top
    _y_meta_top  = _y_meta_bot + META_H
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=_C["bg"])
    ax  = fig.add_axes([0, 0, 1, 1], facecolor=_C["bg"])
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")
    ax.add_patch(Rectangle((0, _y_meta_bot), FIG_W, META_H,
                            fc=_C["meta_bg"], ec="none", zorder=1))
    _hline(_y_meta_bot, color=_C["accent"], lw=2.0, z=3)
    ax.text(FIG_W / 2, _y_meta_bot + META_H * 0.68,
            "AQUEOUS INDEX SWEEP — Modal Index Summary Table",
            ha="center", va="center",
            fontsize=14, fontweight="bold", color=_C["text"],
            fontfamily="DejaVu Sans", zorder=4)
    _meta_line = (
        f"SiN {AIS_WG_HEIGHT_NM:.0f} nm × {AIS_WG_WIDTH_NM:.0f} nm  │  "
        f"R = {AIS_BEND_RADIUS_UM:.4f} µm  │  "
        f"bend_orientation = 0  (radius ‖ Y-axis / wide face)  │  "
        f"λ₀ = {AIS_LAM0_NM:.0f} nm  │  "
        f"ng stencil ±{AIS_DELTA_LAM_NM:.0f} nm  │  "
        f"{AIS_N_POINTS} pts total  │  "
        f"n_SiN = {N_SIN_FIXED}  n_SiO₂ = {N_SIO2_FIXED}"
    )
    ax.text(FIG_W / 2, _y_meta_bot + META_H * 0.25,
            _meta_line,
            ha="center", va="center",
            fontsize=7.8, color=_C["dim"],
            fontfamily="DejaVu Sans", zorder=4)
    ax.add_patch(Rectangle((0, _y_head_bot), FIG_W, HEAD_H,
                            fc=_C["head_bg"], ec="none", zorder=1))
    _hline(_y_head_top, color=_C["accent"], lw=1.5, z=3)
    _hline(_y_head_bot, color=_C["accent"], lw=1.5, z=3)
    for _j, (_cx, _cw, _col) in enumerate(
            zip(_col_centers, _col_widths, _COL_HEADERS)):
        ax.text(_cx, _y_head_bot + HEAD_H / 2, _col,
                ha="center", va="center",
                fontsize=8.6, fontweight="bold", color=_C["accent"],
                fontfamily="DejaVu Sans",
                multialignment="center", zorder=4)
        if _j > 0:
            _vline(_cx - _cw / 2, _y_head_bot, _y_head_top,
                   color=_C["border"], lw=0.8, z=3)
    for _ri, _row in enumerate(_rows):

        _yt = _y_data_top - _ri * ROW_H
        _yb = _yt - ROW_H
        _ym = (_yt + _yb) / 2.0

        _is_first = (_ri == 0)
        _is_last  = (_ri == _n_rows - 1)

        # Row background
        if _is_first or _is_last:
            _bg = _C["row_hi"]
        else:
            _bg = _C["row_odd"] if _ri % 2 == 0 else _C["row_even"]

        ax.add_patch(Rectangle((0, _yb), FIG_W, ROW_H,
                                fc=_bg, ec="none", zorder=1))
        _hline(_yb, z=2)

        # Cell values
        for _j, (_cx, _cw, _val) in enumerate(
                zip(_col_centers, _col_widths, _row)):

            # Per-column colour and weight
            if _j == 0:
                _fc = _C["accent2"] if (_is_first or _is_last) else _C["naq"]
                _fw = "bold"
                _fs = 8.2
            elif _j in (1, 2):     # neff, ng — most important columns
                _fc = _C["text"]
                _fw = "normal"
                _fs = 7.9
            elif _j == 3:          # TE fraction — less important
                _fc = _C["dim"]
                _fw = "normal"
                _fs = 7.9
            else:                  # sensitivity columns
                _fc = _C["mono"]
                _fw = "normal"
                _fs = 7.9

            ax.text(_cx, _ym, _val,
                    ha="center", va="center",
                    fontsize=_fs, fontweight=_fw, color=_fc,
                    fontfamily="DejaVu Sans Mono", zorder=3)

            if _j > 0:
                _vline(_cx - _cw / 2, _yb, _yt, z=2)

        # Side annotation for first / last row
        if _is_first:
            ax.text(FIG_W - 0.10, _ym, "◀ pure water",
                    ha="right", va="center",
                    fontsize=7.0, color=_C["accent"],
                    fontstyle="italic", zorder=3)
        if _is_last:
            ax.text(FIG_W - 0.10, _ym, "◀ contaminated",
                    ha="right", va="center",
                    fontsize=7.0, color=_C["accent2"],
                    fontstyle="italic", zorder=3)
    _hline(_y_data_bot, color=_C["border"], lw=0.8, z=2)
    ax.add_patch(Rectangle((0, 0), FIG_W, _y_foot_top,
                            fc=_C["foot_bg"], ec="none", zorder=1))
    _hline(_y_foot_top, color=_C["accent"], lw=1.2, z=3)
    _S_neff_mean = float(np.mean(ais_S_neff))
    _S_ng_mean   = float(np.mean(ais_S_ng))
    _S_lam_mean  = float(np.mean(ais_S_lam_pm_RIU))
    _S_lam_std   = float(np.std(ais_S_lam_pm_RIU))
    _dneff_total = float(ais_neff[-1] - ais_neff[0])
    _dng_total   = float(ais_ng[-1]   - ais_ng[0])
    _daq_span    = float(ais_n_aq_valid[-1] - ais_n_aq_valid[0])
    _foot_text = (
        f"Computed from {len(ais_n_aq_valid)} valid pts  │  "
        f"Δn_aq = {_daq_span:.4f} RIU  │  "
        f"Δn_eff = {_dneff_total:+.8f}  │  "
        f"Δn_g = {_dng_total:+.8f}  │  "
        f"⟨∂n_eff/∂n_aq⟩ = {_S_neff_mean:.6f} RIU⁻¹  │  "
        f"⟨∂n_g/∂n_aq⟩ = {_S_ng_mean:.6f} RIU⁻¹  │  "
        f"⟨∂λ/∂n_aq⟩ = {_S_lam_mean:.1f} ± {_S_lam_std:.1f} pm/RIU"
    )
    ax.text(FIG_W / 2, _y_foot_top / 2,
            _foot_text,
            ha="center", va="center",
            fontsize=7.8, color=_C["dim"],
            fontfamily="DejaVu Sans", zorder=4)
    ax.add_patch(Rectangle((0.02, 0.02), FIG_W - 0.04, FIG_H - 0.04,
                            fc="none", ec=_C["accent"], lw=1.8, zorder=5))
    for _y_accent in (_y_head_top, _y_head_bot):
        ax.plot([0.02, FIG_W - 0.02], [_y_accent, _y_accent],
                color=_C["accent"], lw=1.4, zorder=5)
    _png_stem = TABLE_PNG_NAME if TABLE_PNG_NAME else f"{AIS_CSV_NAME}_summary_table"
    _pdf_stem = TABLE_PDF_NAME if TABLE_PDF_NAME else _png_stem
    _png_path = DATA_DIR / f"{_png_stem}.png"
    _pdf_path = DATA_DIR / f"{_pdf_stem}.pdf"
    fig.savefig(str(_png_path), dpi=TABLE_DPI, bbox_inches="tight",
                facecolor=_C["bg"], edgecolor="none")
    print(f"  PNG saved → {_png_path}")
    if TABLE_SAVE_PDF:
        fig.savefig(str(_pdf_path), bbox_inches="tight",
                    facecolor=_C["bg"], edgecolor="none")
        print(f"  PDF saved → {_pdf_path}")
    plt.show()
    plt.close(fig)
    print()
    print("─" * 62)
    print(f"  Rows displayed  : {_n_rows}  (step = {_step}, "
          f"endpoints = {TABLE_SHOW_ENDPOINTS})")
    print(f"  Full sweep pts  : {len(ais_n_aq_valid)}")
    print(f"  n_aq range      : {float(ais_n_aq_valid[0]):.4f} → "
          f"{float(ais_n_aq_valid[-1]):.4f}  RIU")
    print(f"  Δneff total     : {_dneff_total:+.8f}")
    print(f"  Δng   total     : {_dng_total:+.8f}")
    print(f"  ⟨∂λ/∂n_aq⟩     : {_S_lam_mean:.1f} ± {_S_lam_std:.1f}  pm/RIU")
    print("─" * 62)
    ARRAY_VALUES_PER_LINE = 10      # numbers per wrapped line inside np.array([])
    ARRAY_NEFF_DECIMALS   = 10      # decimal places for neff and ng
    ARRAY_NAQ_DECIMALS    = 6       # decimal places for n_aq
    ARRAY_INDENT          = "    "  # leading whitespace inside np.array([...])
    _str_naq  = _fmt_array(
        ais_n_aq_valid, "n_aq_sweep",
        ARRAY_NAQ_DECIMALS, ARRAY_VALUES_PER_LINE, ARRAY_INDENT,
    )
    _str_neff = _fmt_array(
        ais_neff, "neff_sweep",
        ARRAY_NEFF_DECIMALS, ARRAY_VALUES_PER_LINE, ARRAY_INDENT,
    )
    _str_ng   = _fmt_array(
        ais_ng, "ng_sweep",
        ARRAY_NEFF_DECIMALS, ARRAY_VALUES_PER_LINE, ARRAY_INDENT,
    )
    _CUT = "✂" + "─" * 76 + "✂"
    _header = (
        f"# ── Aqueous index sweep results — {len(ais_n_aq_valid)} points ───────────────────────\n"
        f"# SiN {AIS_WG_HEIGHT_NM:.0f} nm × {AIS_WG_WIDTH_NM:.0f} nm  │  "
        f"R = {AIS_BEND_RADIUS_UM:.4f} µm  │  "
        f"λ₀ = {AIS_LAM0_NM:.0f} nm  │  "
        f"ng stencil ±{AIS_DELTA_LAM_NM:.0f} nm\n"
        f"# n_SiN = {N_SIN_FIXED}   n_SiO₂ = {N_SIO2_FIXED}   "
        f"n_aq: {float(ais_n_aq_valid[0]):.4f} → "
        f"{float(ais_n_aq_valid[-1]):.4f} RIU\n"
        f"import numpy as np"
    )
    print()
    print(_CUT)
    print(_header)
    print()
    print(_str_naq)
    print()
    print(_str_neff)
    print()
    print(_str_ng)
    print(_CUT)
    print()
    print(f"  ↑  Copy everything between the ✂ lines  ↑")
    print(f"  3 arrays: n_aq_sweep / neff_sweep / ng_sweep  —  "
          f"{len(ais_n_aq_valid)} values each")

    state.update({k: globals().get(k) for k in [
        'ARRAY_INDENT', 'ARRAY_NAQ_DECIMALS', 'ARRAY_NEFF_DECIMALS', 'ARRAY_VALUES_PER_LINE', 'FIG_H', 'FIG_W',
        'FOOT_H', 'HEAD_H', 'META_H', 'PAD', 'ROW_H', 'TABLE_DPI',
        'TABLE_PDF_NAME', 'TABLE_PNG_NAME', 'TABLE_ROW_STEP', 'TABLE_SAVE_PDF', 'TABLE_SHOW_ENDPOINTS', '_C',
        '_COL_HEADERS', '_CUT', '_S_lam_mean', '_S_lam_std', '_S_neff_mean', '_S_ng_mean',
        '_bg', '_col', '_col_centers', '_col_fracs', '_col_widths', '_cw',
        '_cx', '_daq_span', '_dneff_total', '_dng_total', '_fc', '_foot_text',
        '_fs', '_fw', '_header', '_i', '_idxs', '_is_first',
        '_is_last', '_j', '_meta_line', '_n_rows', '_n_valid', '_pdf_path',
        '_pdf_stem', '_png_path', '_png_stem', '_ri', '_row', '_rows',
        '_step', '_str_naq', '_str_neff', '_str_ng', '_val', '_w',
        '_x', '_y_accent', '_y_data_bot', '_y_data_top', '_y_foot_top', '_y_head_bot',
        '_y_head_top', '_y_meta_bot', '_y_meta_top', '_yb', '_ym', '_yt',
        'ax', 'fig',
    ] if k in globals()})
    return state


if __name__ == "__main__":
    run()
