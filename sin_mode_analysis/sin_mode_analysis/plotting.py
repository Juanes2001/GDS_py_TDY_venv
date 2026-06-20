"""
plotting.py — Centralised matplotlib styling and figure export.

The original notebook repeated the same `rcParams` block, the same
colour-blind-safe palette, and the same "save PNG + PDF at publication DPI"
logic in nearly every plotting cell.  They are consolidated here so that:

    * every figure in the package shares one consistent look, and
    * `save_fig(fig, "stem")` exports a PNG *and* a vector PDF in one call.

Import this module (`from .plotting import *`) at the top of any step that
draws figures.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
# ─────────────────────────────────────────────────────────────────────────────
#  Non-interactive backend: figures are SAVED to disk, never displayed.
#  This is deliberate — an interactive backend's plt.show() blocks the script
#  until the window is closed, which stalls a long FDE pipeline. With "Agg" no
#  window is ever opened and plt.show() does nothing, so the run never blocks.
#  Every step still writes its figures (PNG/PDF) to FIGURES_DIR / DATA_DIR;
#  open that folder to view them.
# ─────────────────────────────────────────────────────────────────────────────
mpl.use("Agg")
import matplotlib.pyplot as plt
# Make any stray plt.show() a silent no-op (suppresses the Agg "cannot show"
# warning and guarantees no blocking, anywhere in the package).
plt.show = lambda *args, **kwargs: None
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from .config import FIGURES_DIR, log

# ─────────────────────────────────────────────────────────────────────────────
# Global publication style (serif, light grid, readable fonts)
# ─────────────────────────────────────────────────────────────────────────────
PUB_RCPARAMS = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "dejavuserif",
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.axisbelow": True,
}

# Okabe-Ito colour-blind-safe palette (used throughout the package)
CB_BLUE = "#0072B2"
CB_VERMILION = "#D55E00"
CB_GREEN = "#009E73"
CB_PINK = "#CC79A7"
CB_AMBER = "#E69F00"
CB_SKY = "#56B4E9"


def apply_style() -> None:
    """Apply the package-wide publication rcParams."""
    mpl.rcParams.update(PUB_RCPARAMS)


def save_fig(fig, stem: str, out_dir: Path = FIGURES_DIR,
             dpi: int = 300, also_pdf: bool = True) -> Path:
    """
    Save a figure as PNG (always) and PDF (optional) into `out_dir`.

    Inputs
    ------
    fig : matplotlib.figure.Figure
    stem : str
        Filename without extension (e.g. "modal_analysis").
    out_dir : Path
        Destination directory (created if missing).
    dpi : int
        Raster resolution for the PNG.
    also_pdf : bool
        If True, also write a vector PDF.

    Outputs
    -------
    Path
        Path to the written PNG.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    if also_pdf:
        fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    log.info(f"Saved figure → {png}")
    return png


def make_colorbar(fig, ax, values, cmap_name: str = "viridis", label: str = ""):
    """
    Attach a colour bar mapping a numeric array to a colormap.

    Returns
    -------
    (ScalarMappable, Colorbar)
    """
    norm = Normalize(vmin=float(min(values)), vmax=float(max(values)))
    sm = ScalarMappable(norm=norm, cmap=cmap_name)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    if label:
        cbar.set_label(label)
    return sm, cbar
