#!/usr/bin/env python3
"""
main.py — orchestrator for the SiN 14-ring INTERCONNECT cascade package.

Runs the circuit steps in order, threading one shared ``state`` dict through
every ``step.run(state)`` call (the role the notebook kernel namespace played).
``step1`` runs the core sensor-ring n_eff/n_g sweep and writes ``sweep_results``
into ``state``; the downstream steps read it to build the n_cladding-axis
figures, the FWHM-variant sweeps and the dB/RIU sensitivity summary.

MODE->INTERCONNECT bridge
-------------------------
If a ``platform_bridge.json`` exported by the ``sin_mode_analysis`` package is
present (pass ``--bridge PATH`` or drop it in the data dir), its freshly simulated
radii / n_eff / n_g / kappa^2 / loss and sweep arrays override the study defaults
in ``config`` *before* any sweep runs.  Without it, the package runs standalone on
the config values.

Requirements
------------
* Ansys Lumerical (with the Python ``lumapi``) must be installed; set
  ``LUMERICAL_VERSION`` in ``sin_interconnect_cascade/config.py`` to match your
  install (default ``"v202"``).  The cascade sweeps cannot run without it.
* Results are cached in ``data_ICNT_cascade_ring_sweep/*.h5``; re-running resumes
  from the cache.

Run from this directory:
    python main.py
    python main.py --bridge ../sin_mode_analysis/data_STRp_SiN_mode_analysis_LUM/platform_bridge.json
    python main.py --only step9_sensitivity_summary
    python main.py --no-lumerical        # analytic/plot-only steps
"""
from __future__ import annotations

import argparse
import importlib

# Steps that require a live Lumerical INTERCONNECT session.
_LUMERICAL_STEPS = {
    "step1_cascade_sweep", "step5_two_ring_radius_sweep",
    "step7_sweep_fwhm300", "step8_sweep_fwhm100",
}

ORDER = [
    "step1_cascade_sweep",
    "step2_theoretical_resonances",
    "step3_ncladding_axis",
    "step4_ncladding_plots",
    "step5_two_ring_radius_sweep",
    "step6_kappa_variants",
    "step7_sweep_fwhm300",
    "step8_sweep_fwhm100",
    "step9_sensitivity_summary",
    "step10_resonance_tracking_varfdtd",
]


def run_pipeline(only=None, skip_lumerical=False, bridge_path=None):
    from sin_interconnect_cascade import config, platform_bridge
    # Apply the FDE bridge first so every step sees the simulated parameters.
    platform_bridge.apply_bridge(platform_bridge.load_bridge(bridge_path))

    state: dict = {}
    steps = [only] if only else ORDER
    for name in steps:
        if skip_lumerical and name in _LUMERICAL_STEPS:
            config.log.info(f"— skipping {name} (Lumerical disabled)")
            continue
        config.log.info(f"━━━ running {name} ━━━")
        mod = importlib.import_module(f"sin_interconnect_cascade.{name}")
        state = mod.run(state) or state

    config.log.info("Pipeline complete.")
    return state


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", help="run a single step module by name")
    ap.add_argument("--bridge", help="path to platform_bridge.json from the FDE package")
    ap.add_argument("--no-lumerical", action="store_true",
                    help="skip steps that need a live Lumerical session")
    args = ap.parse_args()
    run_pipeline(only=args.only, skip_lumerical=args.no_lumerical,
                 bridge_path=args.bridge)
