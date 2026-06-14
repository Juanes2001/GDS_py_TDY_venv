"""
sin_interconnect_cascade — modular Lumerical INTERCONNECT circuit package for the
SiN 14-ring evanescent biosensor cascade (1 aqueous sensor + 13 SiO2 spectrometer
drops) at 1550 nm.

The package is the circuit-level counterpart of `sin_mode_analysis`: it consumes
the per-ring radii / n_eff / n_g / kappa^2 / loss produced by the FDE study (see
`platform_bridge`) and runs the ONA-multiport cascade sweeps that yield the drop
spectra, through-port responses and the dB/RIU power sensitivities.

Step modules (run in order via main.py):
    step1_cascade_sweep            core engine + sensor n_eff/n_g sweep
    step2_theoretical_resonances   analytic resonance/FSR map
    step3_ncladding_axis           re-express sweep on aqueous-index axis
    step4_ncladding_plots          drop-power / through figures (n_cladding)
    step5_two_ring_radius_sweep    focused 2-ring radius sweep
    step6_kappa_variants           FWHM 300 pm / 100 pm coupling arrays
    step7_sweep_fwhm300            cascade sweep, FWHM ~300 pm couplers
    step8_sweep_fwhm100            cascade sweep, FWHM ~100 pm couplers
    step9_sensitivity_summary      FWHM x ring dB/RIU sensitivity summary
    step10_resonance_tracking_varfdtd  resonance overlay vs varFDTD
"""
__all__ = ["config", "lumerical_session", "storage", "plotting", "platform_bridge"]
