#!/usr/bin/env python3
"""
main.py — orchestrator for the SiN FDE mode-analysis package.

Runs the analysis steps in order, threading a single shared ``state`` dict
through every ``step.run(state)`` call.  ``state`` plays the role the notebook's
kernel namespace used to play: each step reads the products of earlier steps from
it and writes its own products back.  The final step exports the
MODE->INTERCONNECT bridge file consumed by the ``sin_interconnect_cascade``
package.

Requirements
------------
* Ansys Lumerical (with the Python ``lumapi``) must be installed; set
  ``LUMERICAL_VERSION`` in ``sin_mode_analysis/config.py`` to match your install
  (default ``"v202"``).  The FDE sweeps cannot run without it.
* All results are cached in ``data_STRp_SiN_mode_analysis_LUM/*.h5``; re-running
  resumes from the cache instead of recomputing.

Run from this directory (so the package imports resolve):
    python main.py
    python main.py --only step9_aqueous_sweep
    python main.py --no-lumerical        # analytic/plot-only steps
"""
from __future__ import annotations

import argparse
import importlib

# Steps that require a live Lumerical FDE session.
_LUMERICAL_STEPS = {
    "step1_width_sweep", "step1b_width_sweep_sio2", "step3_ring_radius",
    "step5_spectrometer", "step7_coupler_gap", "step9_aqueous_sweep",
}

ORDER = [
    "step1_width_sweep",
    "step1b_width_sweep_sio2",
    "step2_modal_plots",
    "step3_ring_radius",
    "step4_phase_matching",
    "step5_spectrometer",
    "step6_critical_coupling",
    "step7_coupler_gap",
    "step8_design_summary",
    "step9_aqueous_sweep",
    "step10_aqueous_table",
    "step11_through_varfdtd",
]


def run_pipeline(only=None, skip_lumerical=False):
    from sin_mode_analysis import config, export_platform
    state: dict = {}
    steps = [only] if only else ORDER
    for name in steps:
        if skip_lumerical and name in _LUMERICAL_STEPS:
            config.log.info(f"— skipping {name} (Lumerical disabled)")
            continue
        config.log.info(f"━━━ running {name} ━━━")
        mod = importlib.import_module(f"sin_mode_analysis.{name}")
        state = mod.run(state) or state

    # Export the bridge for the INTERCONNECT package.
    export_platform.export_bridge(state)
    config.log.info("Pipeline complete.")
    return state


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", help="run a single step module by name")
    ap.add_argument("--no-lumerical", action="store_true",
                    help="skip steps that need a live Lumerical FDE session")
    args = ap.parse_args()
    run_pipeline(only=args.only, skip_lumerical=args.no_lumerical)
