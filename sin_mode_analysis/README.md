# sin_mode_analysis — FDE Mode Analysis of the SiN Biosensor Platform

Modular refactor of the original `Lumerical_Mode_analysis.ipynb` notebook into a
clean, PyCharm/GitHub-ready Python package. It performs the finite-difference
eigenmode (FDE) study of a **silicon-nitride (SiN) strip-waveguide evanescent
refractive-index biosensor** and produces every per-ring parameter the companion
circuit package (`sin_interconnect_cascade`) needs.

## Platform

| Quantity | Value |
|---|---|
| Core | Si₃N₄ strip, n ≈ 1.99, height 400 nm, width **auto-selected single-mode** (TE cutoff − `SINGLE_MODE_MARGIN_NM`, default 300 nm) |
| Lower cladding | SiO₂ (n ≈ 1.4469) |
| Upper cladding | aqueous analyte (n ≈ 1.33) for the **sensor** ring; SiO₂ for the 13 **spectrometer** rings |
| Polarization | TE₀ |
| Wavelength | 1550 nm (13-point comb across a ~10 nm FSR) |
| Topology | 1 aqueous sensor ring + 13 SiO₂-clad spectrometer rings |

## Layout

```
sin_mode_analysis/
├── main.py                       # orchestrator: threads a shared `state` dict through every step
├── requirements.txt
├── README.md
└── sin_mode_analysis/            # the importable package
    ├── config.py                 # SINGLE control panel: all user inputs + width-selection rule
    ├── lumerical_session.py      # robust lumapi import + open_mode()
    ├── storage.py                # HDF5 cache / resume helpers
    ├── plotting.py               # publication style + save_fig (PNG + PDF)
    ├── physics.py                # pure-analytic ring/coupler formulas (NO Lumerical)
    ├── export_platform.py        # writes the MODE→INTERCONNECT bridge JSON
    ├── step1_width_sweep.py      # 2D FDE width sweep (aqueous clad)
    ├── step1b_width_sweep_sio2.py
    ├── step2_modal_plots.py
    ├── step3_ring_radius.py
    ├── step4_phase_matching.py
    ├── step5_spectrometer.py
    ├── step6_critical_coupling.py
    ├── step7_coupler_gap.py
    ├── step8_design_summary.py
    ├── step9_aqueous_sweep.py    # sensor aqueous-index sweep → bridge data (ais_neff/ais_ng)
    ├── step10_aqueous_table.py
    └── step11_through_varfdtd.py
```

## Configuration (everything you set lives in `config.py`)

`config.py` is the single user-facing control panel. The top **USER INPUTS**
block holds every quantity that can be fixed without a prior simulation:

| Input | Meaning |
|---|---|
| `LAMBDA0_NM` | central / design wavelength (sensor ring) |
| `TARGET_FSR_NM` | target free spectral range, every ring |
| `N_SPEC_RINGS` | number of SiO₂ spectrometer rings |
| `SENSOR_FWHM_NM`, `SPEC_FWHM_NM` | target FWHM of the sensor / spectrometer rings |
| `RING_RESONANCES_NM` | derived as `LAMBDA0_NM + n·(TARGET_FSR_NM/N_SPEC_RINGS)` |
| `SINGLE_MODE_MARGIN_NM` | backoff below the TE multimode cutoff (default **300 nm**) |
| `WG_WIDTH_OVERRIDE_NM` | `None` = auto-select width; a number forces a width |
| `WG_WIDTH_FALLBACK_NM` | width used only if a step runs without the modal step |

**Single-mode width is the one geometric quantity that cannot be fixed up
front** (it depends on where the guide goes multimode). Instead of hard-coding
it, `config.py` defines the *rule*: `step2_modal_plots` measures the TE
multimode-cutoff width from the width sweep and sets

```
working width = (TE cutoff width) − SINGLE_MODE_MARGIN_NM
```

publishing it through `state` as `selected_width_nm`. The four geometry-building
steps (`step3`, `step5`, `step7`, `step9`) consume that width; the display/
analytic steps (`step4`, `step6`, `step8`, `step10`, `step11`) receive every
parameter through `state` and never hold their own copy.

## Design of the refactor

* **One responsibility per module.** The brittle, repeated machinery the notebook
  carried at the top of every cell (lumapi path/DLL setup, HDF5 cache logic,
  matplotlib `rcParams`, the colour-blind palette) is centralised once in
  `lumerical_session`, `storage` and `plotting`.
* **Scientific bodies preserved verbatim.** The FDE geometry construction,
  `findmodes()` calls, TE/TM classification, and every physics formula are the
  original notebook code, only relocated — not rewritten.
* **Explicit `state` bridge.** Each step is `run(state)`. `state` carries the
  products that used to live in the notebook's kernel namespace (e.g. the swept
  `neff`, the aqueous-sweep `ais_neff`/`ais_ng`). This makes the data flow between
  steps explicit and testable.
* **Cache / resume retained.** Each sweep writes every point to HDF5 and
  `flush()`es immediately, so an interrupted run resumes from the cache.

## Running it

```bash
cd sin_mode_analysis
pip install -r requirements.txt
python main.py                       # full pipeline
python main.py --only step9_aqueous_sweep
python main.py --no-lumerical        # analytic + plotting steps only
```

`main.py` finishes by writing `data_STRp_SiN_mode_analysis_LUM/platform_bridge.json`,
which `sin_interconnect_cascade` can load to drive the circuit sweeps with these
freshly-simulated parameters.

## ⚠️ Verification status (read this)

This package was refactored and checked for **structural and import correctness**:
every module byte-compiles and imports cleanly, and the pure-Python `physics.py`
runs and is internally consistent (e.g. FSR ≈ 10 nm, λ_res ≈ 1550 nm at the
design radius).

It was **not executed end-to-end against Lumerical.** `lumapi` is proprietary and
not available in the refactoring environment, so the FDE sweeps (`step1`,
`step1b`, `step3`, `step5`, `step7`, `step9`) could not be run here. Set
`LUMERICAL_VERSION` in `config.py` to match your install and run on a machine with
Ansys Lumerical to reproduce the full study. The numerical parameter values shipped
in the companion package's `config.py` are the values from the original study.
