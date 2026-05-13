"""
main_script.py — Complete workflow for the 7-ring add-drop cascade
==================================================================
This is the SINGLE file you run.  It orchestrates all other modules:

  config.py          → all constants, paths, ring lengths, port names
  session_manager.py → opens / closes INTERCONNECT sessions safely
  circuit_builder.py → adds elements and wires them exactly as in the GUI
  data_extractor.py  → pulls all 8 ONA port spectra into a ResultBundle
  sweep_runner.py    → runs a sweep over coupling coefficients (or any param)
  plotter.py         → generates all figures and the animated GIF

HOW TO RUN:
  python main_script.py --single        # one baseline run + plots
  python main_script.py --sweep         # coupling sweep over all 7 rings
  python main_script.py --plot          # regenerate figures from saved results
  python main_script.py --all           # single → sweep → plot in sequence

FILL-IN CHECKLIST (before first run):
  [ ] config.py : LUMERICAL_API_PATH
  [ ] config.py : LUMERICAL_VERSION
  [ ] config.py : DEFAULT_START_FREQ, DEFAULT_STOP_FREQ, DEFAULT_NUM_POINTS
  [ ] config.py : RING_ELEMENT_LIBRARY_NAME  (verify in Element Library)
  [ ] config.py : DEFAULT_RING_COUPLING
  [ ] config.py : RING_PORT_INPUT/THROUGH/ADD/DROP  (verify by hovering in GUI)
  [ ] config.py : ONA_OUTPUT_PORT
  [ ] circuit_builder.py : coupling property names on the ring element
  [ ] circuit_builder.py : WIRE_DROP_TO_ONA_DIRECTLY flag
  [ ] data_extractor.py  : result path string in _result_path() if it differs
"""

import argparse
import logging
import numpy as np
from pathlib import Path

# ── Bootstrap logging first ───────────────────────────────────────────────
import logger_setup   # configures root logger from config.py settings
import config

from session_manager import SessionManager
from circuit_builder import build_ring_chain, reset_circuit
from data_extractor  import build_result_bundle, save_results, load_hdf5
from sweep_runner    import SweepRunner, build_param_grid, build_linear_sweep
from plotter         import (
    plot_all_ports,
    plot_single_port,
    plot_sweep_colormap,
    plot_sweep_grid,
    plot_sweep_overlay,
    animate_sweep,
    save_figure,
)

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Single simulation
#   Builds the exact circuit from the GUI, runs it once with default values,
#   saves results, and generates the baseline all-ports plot.
# ══════════════════════════════════════════════════════════════════════════════

def run_single() -> None:
    """
    Run one simulation with the default coupling from config.py and the
    seven ring lengths read from the GUI screenshot.

    Output files:
      results/results.h5             ← HDF5 with all 8 port spectra
      figures/baseline_all_ports.pdf ← overlay of all 8 transmission curves
      figures/baseline_<port>.pdf    ← one figure per ONA port
    """
    log.info("━" * 60)
    log.info("SINGLE SIMULATION — 7-ring add-drop cascade")
    log.info("━" * 60)

    params = {
        "coupling" : config.DEFAULT_RING_COUPLING,
        "loss_db"  : config.DEFAULT_RING_LOSS_DB,
        # ring lengths are always taken from config.RING_LENGTHS_M
    }

    with SessionManager(hide=config.HIDE_GUI) as ic:

        # ── 1. Build circuit ───────────────────────────────────────────────
        log.info("Building circuit …")
        circuit = build_ring_chain(ic)
        #
        # circuit = {"ona": "ONA_1", "rings": ["RING_1", ..., "RING_7"]}
        # build_ring_chain() adds ONA_1, RING_1…RING_7 and wires them
        # according to the topology seen in the GUI screenshot.

        # ── 2. Run simulation ──────────────────────────────────────────────
        log.info("Running simulation …")
        ic.run()
        log.info("Simulation complete.")

        # ── 3. Extract all 8 ONA ports ────────────────────────────────────
        bundle = build_result_bundle(
            ic,
            run_id   = "baseline",
            params   = params,
            ona_name = circuit["ona"],
        )

    # ── 4. Save ───────────────────────────────────────────────────────────
    save_results(bundle)
    log.info(f"Results saved to {config.RESULTS_DIR}")

    # ── 5. Plot all ports together ────────────────────────────────────────
    fig = plot_all_ports(
        bundle,
        title   = "7-ring cascade — baseline  (all ONA ports)",
        save_as = "baseline_all_ports",
    )
    log.info("All-ports figure saved.")

    # ── 6. Plot each port individually ───────────────────────────────────
    for label in bundle.spectra:
        safe_name = label.replace(" ", "_").replace("—", "").strip("_")
        plot_single_port(
            bundle,
            port_label = label,
            title      = f"Baseline — {label}",
            save_as    = f"baseline_{safe_name}",
        )

    log.info("Single simulation done.  Check figures/ for output.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Parameter sweep
#   Sweeps the coupling coefficient for all rings simultaneously.
#   Each run rebuilds the full circuit with the new coupling value.
#   Results are saved incrementally (safe to interrupt and resume).
# ══════════════════════════════════════════════════════════════════════════════

def _simulation_fn(ic, params: dict):
    """
    Simulation function called once per sweep point.

    Receives an open ic session and a params dict.
    Builds the circuit, runs it, extracts results, returns a ResultBundle.

    params keys used here:
      "coupling"  → coupling coefficient applied to all 7 rings
      "loss_db"   → ring round-trip loss (optional, defaults to config value)
    """
    from circuit_builder import build_ring_chain, reset_circuit
    from data_extractor  import build_result_bundle

    # Always start from a clean slate inside a sweep
    reset_circuit(ic)

    # Build circuit with swept coupling
    circuit = build_ring_chain(
        ic,
        coupling = params.get("coupling", config.DEFAULT_RING_COUPLING),
    )

    ic.run()

    return build_result_bundle(
        ic,
        run_id   = "temp",   # SweepRunner replaces this with the hash-based ID
        params   = params,
        ona_name = circuit["ona"],
    )


def run_sweep() -> None:
    """
    Sweep the coupling coefficient from 0.05 to 0.50 in 10 steps.

    ← FILL THIS IN: adjust the sweep range and step count below.

    Each combination is saved separately.  The sweep can be interrupted
    and resumed safely — already-finished runs are skipped.

    Output files:
      results/results.h5          ← all runs in one HDF5
      results/registry.json       ← which runs are done
      results/csv/*.csv           ← per-run CSV (if SAVE_FORMAT includes csv)
    """
    log.info("━" * 60)
    log.info("COUPLING SWEEP — 7-ring add-drop cascade")
    log.info("━" * 60)

    # ── Define sweep ──────────────────────────────────────────────────────
    # 1-D coupling sweep (most informative for this topology)
    coupling_values = np.linspace(0.05, 0.50, 10).tolist()  # ← FILL THIS IN

    param_grid = build_linear_sweep(
        "coupling",
        coupling_values,
        loss_db = config.DEFAULT_RING_LOSS_DB,
    )

    # Alternatively, 2-D grid (coupling × loss):
    # param_grid = build_param_grid(
    #     coupling = coupling_values,
    #     loss_db  = [0.0, 1.0, 3.0],
    # )

    log.info(f"Sweep: {len(param_grid)} runs  |  "
             f"coupling = {coupling_values[0]:.3f} … {coupling_values[-1]:.3f}")

    runner = SweepRunner(
        simulation_fn = _simulation_fn,
        results_dir   = config.RESULTS_DIR,
        run_id_prefix = "kappa",
        resume        = True,
    )
    results = runner.run(param_grid)

    n_ok = sum(1 for r in results if r is not None)
    log.info(f"Sweep finished: {n_ok}/{len(param_grid)} successful.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Post-processing and figures
#   Loads all sweep results from the HDF5 file and generates:
#     - Colourmap for each ONA port (spectral evolution vs coupling)
#     - Grid colourmap (all 8 ports side-by-side)
#     - Overlay plot per port
#     - Animated GIF of the full cascade response vs coupling
# ══════════════════════════════════════════════════════════════════════════════

def run_plots() -> None:
    """
    Load sweep results from results/results.h5 and generate all figures.
    """
    log.info("━" * 60)
    log.info("POST-PROCESSING & FIGURES")
    log.info("━" * 60)

    h5_file = config.RESULTS_DIR / "results.h5"
    if not h5_file.exists():
        log.error(f"No results file at {h5_file}.  Run --sweep first.")
        return

    # ── Load all runs ─────────────────────────────────────────────────────
    import h5py
    with h5py.File(h5_file, "r") as f:
        run_ids = [k for k in f.keys() if k != "baseline"]

    if not run_ids:
        log.warning("No sweep runs found (only baseline exists). Run --sweep first.")
        return

    log.info(f"Loading {len(run_ids)} sweep runs …")
    bundles = [load_hdf5(h5_file, rid) for rid in run_ids]

    # ── Sort by coupling ─────────────────────────────────────────────────
    sweep_param = "coupling"
    bundles.sort(key=lambda b: b.params.get(sweep_param, 0))

    sweep_values = [b.params[sweep_param] for b in bundles]
    frequency    = bundles[0].frequency
    port_labels  = list(bundles[0].spectra.keys())

    log.info(f"Coupling range: {sweep_values[0]:.3f} … {sweep_values[-1]:.3f}")

    # ── Build per-port transmission matrices (M × N) ──────────────────────
    matrix_dict = {
        label: np.vstack([b.spectra[label] for b in bundles])
        for label in port_labels
        if all(b.spectra.get(label) is not None for b in bundles)
    }

    # ── 1. Grid colourmap (all 8 ports in one figure) ─────────────────────
    log.info("Generating grid colourmap …")
    plot_sweep_grid(
        frequency    = frequency,
        sweep_values = sweep_values,
        matrix_dict  = matrix_dict,
        sweep_label  = "Coupling coefficient",
        sweep_units  = "",
        suptitle     = "7-ring cascade — spectral evolution vs coupling",
        save_as      = "sweep_grid_coupling",
    )

    # ── 2. Individual colourmap per port ──────────────────────────────────
    log.info("Generating per-port colourmaps …")
    for label, M in matrix_dict.items():
        safe = label.replace(" ", "_").replace("—", "").strip("_")
        plot_sweep_colormap(
            frequency           = frequency,
            sweep_values        = sweep_values,
            transmission_matrix = M,
            port_label          = label,
            sweep_label         = "Coupling coefficient κ",
            title               = "Spectral evolution",
            save_as             = f"sweep_colormap_{safe}",
        )

    # ── 3. Overlay per port ───────────────────────────────────────────────
    log.info("Generating per-port overlays …")
    for label, M in matrix_dict.items():
        safe = label.replace(" ", "_").replace("—", "").strip("_")
        plot_sweep_overlay(
            frequency           = frequency,
            sweep_values        = sweep_values,
            transmission_matrix = M,
            port_label          = label,
            sweep_label         = "Coupling κ",
            title               = "Sweep overlay",
            save_as             = f"sweep_overlay_{safe}",
        )

    # ── 4. Animated GIF ───────────────────────────────────────────────────
    log.info("Generating animation …")
    anim_path = animate_sweep(
        frequency       = frequency,
        sweep_values    = sweep_values,
        matrix_dict     = matrix_dict,
        sweep_label     = "Coupling κ",
        title_template  = "7-ring cascade   κ = {value:.3f}",
        fps             = 6,
        output_format   = "gif",    # change to "mp4" if ffmpeg is installed
        output_filename = "cascade_coupling_sweep",
    )
    log.info(f"Animation: {anim_path}")

    log.info(f"All figures saved to {config.FIGURES_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="7-ring add-drop cascade — INTERCONNECT automation"
    )
    parser.add_argument("--single", action="store_true",
                        help="Run one baseline simulation and generate initial figures")
    parser.add_argument("--sweep",  action="store_true",
                        help="Run the coupling coefficient parameter sweep")
    parser.add_argument("--plot",   action="store_true",
                        help="Generate all figures from saved sweep results")
    parser.add_argument("--all",    action="store_true",
                        help="Run single → sweep → plot in sequence")
    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
    else:
        if args.all or args.single:
            run_single()
        if args.all or args.sweep:
            run_sweep()
        if args.all or args.plot:
            run_plots()
