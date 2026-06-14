"""
sin_mode_analysis — modular Lumerical FDE mode-analysis package for the SiN
strip-waveguide evanescent biosensor platform (LAB 5).

The package reorganises the original ``Lumerical_Mode_analysis`` notebook into a
clean set of single-responsibility modules while preserving, verbatim, the
scientific bodies, the HDF5 cache/resume logic and the cross-step data flow.

Shared infrastructure:
    config              platform constants, paths, logger
    lumerical_session   robust lumapi import + open_mode()
    storage             HDF5 cache / resume helpers
    plotting            publication style + save_fig (PNG+PDF)
    physics             pure-analytic ring/coupler formulas (no Lumerical)
    export_platform     write the MODE->INTERCONNECT bridge JSON

Step modules (run in order via main.py):
    step1_width_sweep             2D FDE width sweep (aqueous clad)
    step1b_width_sweep_sio2       same sweep, symmetric SiO2 stack
    step2_modal_plots             TE/TM post-processing + figures
    step3_ring_radius             sensor-ring radius for target FSR
    step4_phase_matching          analytic phase-matching correction
    step5_spectrometer            13 SiO2 spectrometer rings
    step6_critical_coupling       analytic critical-coupling design
    step7_coupler_gap             directional-coupler gap sweep
    step8_design_summary          design tables + through spectra
    step9_aqueous_sweep           sensor aqueous-index sweep (bridge data)
    step10_aqueous_table          aqueous-sweep summary table
    step11_through_varfdtd        analytic through model vs varFDTD
"""
__all__ = ["config", "lumerical_session", "storage", "plotting", "physics",
           "export_platform"]
