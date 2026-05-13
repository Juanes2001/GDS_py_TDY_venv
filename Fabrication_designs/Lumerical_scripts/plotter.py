"""
plotter.py — Publication-quality plots for the 7-ring cascade
==============================================================
Provides:
  plot_all_ports()          : all 8 ONA spectra on one figure (main diagnostic plot)
  plot_single_port()        : one port's spectrum
  plot_sweep_colormap()     : 2-D colourmap — swept param vs wavelength
  plot_sweep_overlay()      : all sweep curves overlaid per port
  animate_sweep()           : animated GIF / MP4 of spectral evolution
  save_figure()             : export helper
"""

import logging
from pathlib import Path
from typing  import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot    as plt
import matplotlib.animation as animation
import matplotlib.ticker    as ticker
import matplotlib.colors    as mcolors
import numpy as np

import config

log = logging.getLogger(__name__)

try:
    plt.style.use(config.PLOT_STYLE)
except Exception:
    pass   # fall back to matplotlib defaults


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _to_db(arr: np.ndarray) -> np.ndarray:
    return 10 * np.log10(np.abs(arr) ** 2 + 1e-30)

def _x_axis(frequency: np.ndarray, mode: str):
    if mode == "wavelength":
        return (config.C_LIGHT / frequency) * 1e9, "Wavelength (nm)"
    return frequency * 1e-12, "Frequency (THz)"

def save_figure(fig: plt.Figure, filename: str, directory: Path = None) -> Path:
    directory = Path(directory or config.FIGURES_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    ext  = config.PLOT_FORMAT.lstrip(".")
    path = directory / f"{filename}.{ext}"
    fig.savefig(path, dpi=config.PLOT_DPI, bbox_inches="tight")
    log.info(f"Figure saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# 1.  All-ports overlay — primary diagnostic plot for the ring cascade
# ══════════════════════════════════════════════════════════════════════════════

def plot_all_ports(
    bundle,                          # ResultBundle
    x_axis    : str   = "wavelength",
    y_axis    : str   = "dB",
    title     : str   = "7-ring cascade — all ONA ports",
    figsize   : Tuple = None,
    alpha     : float = 0.85,
    save_as   : str   = None,
) -> plt.Figure:
    """
    Plot all 8 ONA port spectra on a single axes.

    Each port is plotted with a distinct colour from the configured colormap.
    Through ports are drawn with a solid line; drop ports with a dashed line.

    Parameters
    ----------
    bundle  : ResultBundle from data_extractor.build_result_bundle()
    x_axis  : "wavelength" (nm) or "frequency" (THz)
    y_axis  : "dB" or "linear"
    title   : figure title
    save_as : filename stem for saving; None → skip

    Returns
    -------
    plt.Figure
    """
    if not bundle.spectra:
        log.warning("plot_all_ports: no spectra in bundle.")
        return None

    x, xlab = _x_axis(bundle.frequency, x_axis)

    cmap   = matplotlib.colormaps.get_cmap(config.PLOT_COLORMAP)
    labels = list(bundle.spectra.keys())
    n      = len(labels)
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    fig, ax = plt.subplots(figsize=figsize or config.PLOT_FIGSIZE)

    for i, (label, arr) in enumerate(bundle.spectra.items()):
        if arr is None:
            continue
        y        = _to_db(arr) if y_axis == "dB" else np.abs(arr) ** 2
        linestyle = "-" if "through" in label.lower() else "--"
        ax.plot(x, y, lw=1.1, alpha=alpha, color=colors[i],
                linestyle=linestyle, label=label)

    ylab = "Transmission (dB)" if y_axis == "dB" else "Transmission (|T|²)"
    ax.set_xlabel(xlab, fontsize=11)
    ax.set_ylabel(ylab, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, ncol=2, loc="lower right",
              framealpha=0.7, edgecolor="none")
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5)
    ax.minorticks_on()

    fig.tight_layout()
    if save_as:
        save_figure(fig, save_as)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Single-port spectrum
# ══════════════════════════════════════════════════════════════════════════════

def plot_single_port(
    bundle,
    port_label : str,
    x_axis     : str   = "wavelength",
    y_axis     : str   = "dB",
    color      : str   = "steelblue",
    title      : str   = None,
    save_as    : str   = None,
    figsize    : Tuple = None,
) -> plt.Figure:
    """
    Plot a single ONA port spectrum.

    Parameters
    ----------
    bundle     : ResultBundle
    port_label : key in bundle.spectra (e.g. "RING_3 — drop")
    """
    if port_label not in bundle.spectra:
        raise KeyError(f"'{port_label}' not found in bundle.spectra. "
                       f"Available: {list(bundle.spectra.keys())}")

    arr      = bundle.spectra[port_label]
    x, xlab  = _x_axis(bundle.frequency, x_axis)
    y        = _to_db(arr) if y_axis == "dB" else np.abs(arr) ** 2
    ylab     = "Transmission (dB)" if y_axis == "dB" else "Transmission (|T|²)"
    title    = title or f"Spectrum — {port_label}"

    fig, ax = plt.subplots(figsize=figsize or config.PLOT_FIGSIZE)
    ax.plot(x, y, lw=1.2, color=color, label=port_label)
    ax.set_xlabel(xlab, fontsize=11)
    ax.set_ylabel(ylab, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    ax.minorticks_on()
    fig.tight_layout()

    if save_as:
        save_figure(fig, save_as)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Sweep colourmap — spectral evolution vs a swept parameter
# ══════════════════════════════════════════════════════════════════════════════

def plot_sweep_colormap(
    frequency           : np.ndarray,
    sweep_values        : List[float],
    transmission_matrix : np.ndarray,       # shape (M, N)
    port_label          : str   = "",
    sweep_label         : str   = "Sweep parameter",
    sweep_units         : str   = "",
    x_axis              : str   = "wavelength",
    y_axis              : str   = "dB",
    title               : str   = "Spectral evolution",
    colormap            : str   = None,
    vmin                : float = None,
    vmax                : float = None,
    figsize             : Tuple = None,
    save_as             : str   = None,
) -> plt.Figure:
    """
    2-D colourmap: X = wavelength/frequency, Y = sweep parameter, Z = |T| in dB.

    Parameters
    ----------
    transmission_matrix : complex array (M swept values × N frequency points)
    sweep_values        : list length M
    port_label          : appended to title for multi-panel figures
    """
    x, xlab = _x_axis(frequency, x_axis)
    Z   = _to_db(transmission_matrix) if y_axis == "dB" else np.abs(transmission_matrix) ** 2
    zlab = "Transmission (dB)" if y_axis == "dB" else "|T|²"
    ylabel = f"{sweep_label} ({sweep_units})" if sweep_units else sweep_label
    ttl = f"{title}  [{port_label}]" if port_label else title

    fig, ax = plt.subplots(figsize=figsize or (9, 5))
    cm = ax.pcolormesh(x, sweep_values, Z,
                       cmap=colormap or config.PLOT_COLORMAP,
                       vmin=vmin, vmax=vmax, shading="auto")
    cb = fig.colorbar(cm, ax=ax, pad=0.02)
    cb.set_label(zlab, fontsize=10)
    ax.set_xlabel(xlab, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(ttl, fontsize=12, fontweight="bold")
    fig.tight_layout()

    if save_as:
        save_figure(fig, save_as)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Multi-port sweep colourmap grid
#     One subplot per ONA port — shows how each channel evolves with a parameter
# ══════════════════════════════════════════════════════════════════════════════

def plot_sweep_grid(
    frequency           : np.ndarray,
    sweep_values        : List[float],
    matrix_dict         : Dict[str, np.ndarray],   # {port_label → (M,N) complex}
    sweep_label         : str   = "Sweep parameter",
    sweep_units         : str   = "",
    x_axis              : str   = "wavelength",
    y_axis              : str   = "dB",
    suptitle            : str   = "Sweep — all ports",
    colormap            : str   = None,
    figsize             : Tuple = None,
    save_as             : str   = None,
) -> plt.Figure:
    """
    Grid of colourmap subplots, one per ONA port.

    Parameters
    ----------
    matrix_dict : {port_label → complex matrix (M, N)}
    """
    n       = len(matrix_dict)
    ncols   = 2
    nrows   = (n + 1) // ncols
    fw, fh  = figsize or (14, nrows * 3.5)

    fig, axes = plt.subplots(nrows, ncols, figsize=(fw, fh), squeeze=False)
    axes_flat = axes.ravel()

    x, xlab  = _x_axis(frequency, x_axis)
    ylabel   = f"{sweep_label} ({sweep_units})" if sweep_units else sweep_label
    zlab     = "Transmission (dB)" if y_axis == "dB" else "|T|²"
    cmap     = colormap or config.PLOT_COLORMAP

    # Global colour scale for fair comparison across ports
    all_Z  = np.concatenate([
        (_to_db(M) if y_axis == "dB" else np.abs(M) ** 2).ravel()
        for M in matrix_dict.values()
    ])
    vmin, vmax = np.nanpercentile(all_Z, 1), np.nanpercentile(all_Z, 99)

    for ax, (label, M) in zip(axes_flat, matrix_dict.items()):
        Z = _to_db(M) if y_axis == "dB" else np.abs(M) ** 2
        cm = ax.pcolormesh(x, sweep_values, Z,
                           cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.set_xlabel(xlab, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)

    # Hide unused subplots
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.colorbar(cm, ax=axes_flat[:n], label=zlab, shrink=0.6, pad=0.02)
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save_as:
        save_figure(fig, save_as)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Sweep overlay — all parameter values on one axes per port
# ══════════════════════════════════════════════════════════════════════════════

def plot_sweep_overlay(
    frequency           : np.ndarray,
    sweep_values        : List[float],
    transmission_matrix : np.ndarray,
    port_label          : str   = "",
    sweep_label         : str   = "Param",
    sweep_units         : str   = "",
    x_axis              : str   = "wavelength",
    y_axis              : str   = "dB",
    title               : str   = "Sweep overlay",
    alpha               : float = 0.7,
    figsize             : Tuple = None,
    save_as             : str   = None,
) -> plt.Figure:
    """Overlay all sweep parameter values on one axes for one port."""
    x, xlab  = _x_axis(frequency, x_axis)
    cmap     = matplotlib.colormaps.get_cmap(config.PLOT_COLORMAP)
    norm     = mcolors.Normalize(vmin=min(sweep_values), vmax=max(sweep_values))
    ttl      = f"{title}  [{port_label}]" if port_label else title
    ylab     = "Transmission (dB)" if y_axis == "dB" else "Transmission (|T|²)"

    fig, ax = plt.subplots(figsize=figsize or config.PLOT_FIGSIZE)

    for sv, row in zip(sweep_values, transmission_matrix):
        y = _to_db(row) if y_axis == "dB" else np.abs(row) ** 2
        ax.plot(x, y, color=cmap(norm(sv)), lw=0.9, alpha=alpha)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    clabel = f"{sweep_label} ({sweep_units})" if sweep_units else sweep_label
    fig.colorbar(sm, ax=ax, label=clabel, pad=0.02)

    ax.set_xlabel(xlab, fontsize=11)
    ax.set_ylabel(ylab, fontsize=11)
    ax.set_title(ttl, fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    ax.minorticks_on()
    fig.tight_layout()

    if save_as:
        save_figure(fig, save_as)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Animation — sweep movie (one frame per swept value, all ports)
# ══════════════════════════════════════════════════════════════════════════════

def animate_sweep(
    frequency           : np.ndarray,
    sweep_values        : List[float],
    matrix_dict         : Dict[str, np.ndarray],   # {port_label → (M,N) complex}
    sweep_label         : str   = "Param",
    sweep_units         : str   = "",
    x_axis              : str   = "wavelength",
    y_axis              : str   = "dB",
    title_template      : str   = "{label} = {value:.4g} {units}",
    fps                 : int   = 8,
    output_format       : str   = "gif",    # "gif" | "mp4"
    output_filename     : str   = "sweep_animation",
    output_dir          : Path  = None,
    figsize             : Tuple = None,
    y_limits            : Tuple = None,
) -> Path:
    """
    Animated GIF / MP4 showing how all port spectra evolve with a swept parameter.

    Each frame corresponds to one sweep value; all 8 port spectra are drawn
    together on one axes so you can see the whole cascade response change at once.

    Parameters
    ----------
    matrix_dict   : {port_label → complex matrix (M, N)}
    sweep_values  : list of M parameter values
    fps           : frames per second
    output_format : "gif" (Pillow, no extra install) or "mp4" (requires ffmpeg)

    Returns
    -------
    Path to the output animation file
    """
    output_dir = Path(output_dir or config.FIGURES_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    x, xlab = _x_axis(frequency, x_axis)
    ylab    = "Transmission (dB)" if y_axis == "dB" else "Transmission (|T|²)"

    labels = list(matrix_dict.keys())
    n      = len(labels)
    cmap   = matplotlib.colormaps.get_cmap(config.PLOT_COLORMAP)
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    # Pre-compute all dB matrices for speed
    Z_dict = {
        lbl: (_to_db(M) if y_axis == "dB" else np.abs(M) ** 2)
        for lbl, M in matrix_dict.items()
    }

    # Fixed Y limits across all frames
    if y_limits is None:
        all_vals = np.concatenate([Z.ravel() for Z in Z_dict.values()])
        ymin, ymax = np.nanpercentile(all_vals, 0.5), np.nanpercentile(all_vals, 99.5)
        margin = (ymax - ymin) * 0.04
        y_limits = (ymin - margin, ymax + margin)

    fig, ax = plt.subplots(figsize=figsize or config.PLOT_FIGSIZE)
    lines   = []
    for i, label in enumerate(labels):
        ls = "-" if "through" in label.lower() else "--"
        ln, = ax.plot([], [], lw=1.1, color=colors[i],
                      linestyle=ls, label=label, alpha=0.85)
        lines.append(ln)

    title_obj = ax.set_title("", fontsize=11, fontweight="bold")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(*y_limits)
    ax.set_xlabel(xlab, fontsize=11)
    ax.set_ylabel(ylab, fontsize=11)
    ax.legend(fontsize=7, ncol=2, loc="lower right",
              framealpha=0.6, edgecolor="none")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    fig.tight_layout()

    def _init():
        for ln in lines:
            ln.set_data([], [])
        return lines

    def _update(frame_idx):
        for ln, label in zip(lines, labels):
            ln.set_data(x, Z_dict[label][frame_idx])
        sv  = sweep_values[frame_idx]
        ttl = title_template.format(label=sweep_label, value=sv, units=sweep_units)
        title_obj.set_text(ttl)
        return lines + [title_obj]

    ani = animation.FuncAnimation(
        fig, _update, frames=len(sweep_values),
        init_func=_init, blit=True, repeat=False,
    )

    if output_format == "mp4":
        writer   = animation.FFMpegWriter(fps=fps, bitrate=1800)
        out_path = output_dir / f"{output_filename}.mp4"
    else:
        writer   = animation.PillowWriter(fps=fps)
        out_path = output_dir / f"{output_filename}.gif"

    ani.save(str(out_path), writer=writer)
    plt.close(fig)
    log.info(f"Animation saved: {out_path}")
    return out_path
