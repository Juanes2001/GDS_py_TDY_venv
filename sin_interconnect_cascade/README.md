# sin_interconnect_cascade — INTERCONNECT Circuit Model of the 14-Ring SiN Cascade

Modular refactor of the original `Lumerical_Interconnect.ipynb` notebook into a
clean, PyCharm/GitHub-ready Python package. It is the **circuit-level counterpart**
of `sin_mode_analysis`: it takes the per-ring parameters from the FDE study and
runs the Lumerical **INTERCONNECT** cascade sweeps that yield the drop spectra,
through-port responses and the refractive-index power sensitivities.

## Circuit (V3 topology)

A unidirectional cascade of **14 add–drop ring resonators** measured by a single
**15-input Optical Network Analyzer (ONA)**:

```
ONA out         → RING_1 input
RING_1  through → ONA input 1            (aqueous SENSOR ring, swept)
RING_1  drop    → RING_2 input           (sensor drop feeds the cascade)
RING_n  through → RING_{n+1} input       n = 2..13
RING_n  drop    → ONA input n            n = 2..14  (13 SiO₂ SPECTROMETER drops)
RING_14 through → ONA input 15           (cascade end)
```

Each ring is a Lumerical *“Double Bus Ring Resonator”* in *unidirectional* mode;
its physical parameters are `length = 2πR`, `frequency = c/λ_res`, effective/group
index, loss, dispersion, and the two coupling coefficients (input `1 1`, drop `1 2`).
Physics reference: Bogaerts et al., *Silicon microring resonators*, Laser &
Photonics Reviews **6**, 47–73 (2012).

## Layout

```
sin_interconnect_cascade/
├── main.py                          # orchestrator: applies the FDE bridge, then threads `state` through the steps
├── requirements.txt
├── README.md
└── sin_interconnect_cascade/        # the importable package
    ├── config.py                    # ring arrays, ONA map, sweep ranges, paths, logger
    ├── lumerical_session.py         # robust lumapi import + open_interconnect()
    ├── storage.py                   # HDF5 cache / resume helpers
    ├── plotting.py                  # publication style + save_fig (PNG + PDF)
    ├── platform_bridge.py           # loads the MODE→INTERCONNECT bridge JSON (optional override)
    ├── step1_cascade_sweep.py       # CORE ENGINE: circuit builder + ONA sweep + result plots
    ├── step2_theoretical_resonances.py
    ├── step3_ncladding_axis.py
    ├── step4_ncladding_plots.py
    ├── step5_two_ring_radius_sweep.py
    ├── step6_kappa_variants.py      # FWHM 300 pm / 100 pm coupling arrays
    ├── step7_sweep_fwhm300.py
    ├── step8_sweep_fwhm100.py
    ├── step9_sensitivity_summary.py # dB/RIU sensitivity, FWHM × ring comparison
    └── step10_resonance_tracking_varfdtd.py
```

## Design of the refactor

* **`step1_cascade_sweep` is the engine.** It holds the circuit primitives
  (`_eval`, `_build_circuit`, `_extract_results`, `run_interconnect_sweep`,
  `get_results`, …). The other steps import what they need from it rather than
  re-declaring it.
* **Configuration separated from logic** (per the project's restructuring notes):
  every editable number — ring arrays, κ², ONA settings, sweep ranges, paths —
  lives in `config.py`. The κ-variant arrays (FWHM 500/300/100 pm = core /
  `_03` / `_01`) are kept verbatim under their original names so the cached HDF5
  dataset names and the FWHM study reproduce exactly.
* **Explicit `state` bridge.** Each step is `run(state)`; `step1` writes
  `sweep_results` into `state` and the downstream steps read it.
* **Cache / resume retained**, and INTERCONNECT always receives SI units (metres,
  Hz); nm/µm/dBm are display-only.

## MODE → INTERCONNECT bridge

`config.py` ships with the original study's ring parameters, so the package runs
**standalone**. If you also run `sin_mode_analysis`, point this package at the JSON
it exports to drive the circuit with freshly-simulated values:

```bash
python main.py --bridge ../sin_mode_analysis/data_STRp_SiN_mode_analysis_LUM/platform_bridge.json
```

`platform_bridge.apply_bridge()` overrides the matching `config` arrays *before*
any sweep runs.

## Running it

```bash
cd sin_interconnect_cascade
pip install -r requirements.txt
python main.py                       # full pipeline
python main.py --only step9_sensitivity_summary
python main.py --no-lumerical        # analytic + plotting steps only
```

## ⚠️ Verification status (read this)

This package was refactored and checked for **structural and import correctness**:
all 14 modules byte-compile and import cleanly (with a mocked `lumapi` and a
headless matplotlib backend).

It was **not executed end-to-end against Lumerical.** `lumapi` is proprietary and
unavailable in the refactoring environment, so the INTERCONNECT sweeps (`step1`,
`step5`, `step7`, `step8`) could not be run here. Run on a machine with Ansys
Lumerical (set `LUMERICAL_VERSION` in `config.py`) to reproduce the full study.
The ring parameter arrays in `config.py` are the values from the original study.
