"""
physics.py — Pure-Python analytic photonics formulas used across the package.

These are the closed-form relations behind the FDE sweeps.  They contain NO
Lumerical calls, so they can be unit-tested and reused freely.  Each function
documents its inputs/outputs/units and gives a worked example.

Primary reference for the ring-resonator relations:
    W. Bogaerts et al., "Silicon microring resonators,"
    Laser & Photonics Reviews 6, 47-73 (2012).
"""
from __future__ import annotations

import numpy as np

LN10 = np.log(10.0)


def ring_circumference_um(radius_um: float) -> float:
    """
    Ring round-trip length L = 2*pi*R.

    Inputs : radius_um [um]
    Outputs: circumference [um]
    Example: ring_circumference_um(19.0) -> 119.38
    """
    return 2.0 * np.pi * float(radius_um)


def fsr_nm(lambda0_nm: float, n_group: float, length_um: float) -> float:
    """
    Free spectral range  FSR = lambda^2 / (n_g * L).

    Inputs : lambda0_nm [nm], n_group [-], length_um [um]
    Outputs: FSR [nm]
    Example: fsr_nm(1550.0, 2.0, 119.38) -> ~10.06 nm
    """
    lam_m = float(lambda0_nm) * 1e-9
    L_m = float(length_um) * 1e-6
    return (lam_m ** 2 / (float(n_group) * L_m)) * 1e9


def group_index_central_diff(neff_0: float, neff_minus: float, neff_plus: float,
                             lambda0_nm: float, dlambda_nm: float) -> float:
    """
    Group index by central difference of n_eff(lambda):
        n_g = n_eff(l0) - l0 * [n_eff(l0+dl) - n_eff(l0-dl)] / (2*dl)

    Inputs : neff_0, neff_minus, neff_plus [-], lambda0_nm [nm], dlambda_nm [nm]
    Outputs: n_group [-]
    Example: group_index_central_diff(1.70,1.701,1.699,1550,5) -> ~2.01
    """
    dndl = (neff_plus - neff_minus) / (2.0 * float(dlambda_nm))
    return float(neff_0) - float(lambda0_nm) * dndl


def resonance_order(neff: float, length_um: float, lambda0_nm: float) -> int:
    """
    Nearest integer resonance order m minimising |m*lambda - neff*L|:
        m = round(neff * L / lambda)

    Inputs : neff [-], length_um [um], lambda0_nm [nm]
    Outputs: m [int]
    Example: resonance_order(1.70, 119.38, 1550.0) -> 130
    """
    L_nm = float(length_um) * 1e3
    return int(round(float(neff) * L_nm / float(lambda0_nm)))


def resonance_wavelength_nm(neff: float, length_um: float, m: int) -> float:
    """
    Ring resonance wavelength  lambda_res = neff * L / m.

    Inputs : neff [-], length_um [um], m [int]
    Outputs: lambda_res [nm]
    """
    L_nm = float(length_um) * 1e3
    return float(neff) * L_nm / int(m)


def db_per_cm_to_alpha_field(db_per_cm: float) -> float:
    """
    Convert a power loss in dB/cm to a field amplitude loss coefficient [1/m].

        alpha_power[1/m] = (db/cm) * 100 * ln(10) / 10
        alpha_field[1/m] = alpha_power / 2

    Inputs : db_per_cm [dB/cm]
    Outputs: alpha_field [1/m]
    Example: db_per_cm_to_alpha_field(1.0) -> ~11.51
    """
    alpha_power_per_m = float(db_per_cm) * 100.0 * LN10 / 10.0
    return alpha_power_per_m / 2.0


def alpha_field_to_db_per_m(alpha_field_per_m: float) -> float:
    """
    Convert a field amplitude loss coefficient [1/m] to a power loss [dB/m].

        alpha_power[dB/m] = alpha_field[1/m] * 20 / ln(10)

    Inputs : alpha_field_per_m [1/m]
    Outputs: power loss [dB/m]
    """
    return float(alpha_field_per_m) * 20.0 / LN10


def alpha_bend_from_imag_neff(imag_neff: float, lambda_m: float) -> float:
    """
    Bend (radiation) field-loss coefficient from the imaginary part of n_eff:
        alpha_bend[1/m] = 4*pi*Im(n_eff) / lambda

    Inputs : imag_neff [-], lambda_m [m]
    Outputs: alpha_bend [1/m]
    """
    return 4.0 * np.pi * float(imag_neff) / float(lambda_m)


def roundtrip_amplitude(alpha_field_per_m: float, length_um: float) -> float:
    """
    Single-pass / round-trip field amplitude transmission a = exp(-alpha*L).

    Inputs : alpha_field_per_m [1/m], length_um [um]
    Outputs: a [-] in (0, 1]
    Example: roundtrip_amplitude(11.51, 119.38) -> ~0.9986
    """
    L_m = float(length_um) * 1e-6
    return float(np.exp(-float(alpha_field_per_m) * L_m))


def kappa_from_r(r: float) -> float:
    """
    Coupler cross-coupling amplitude from self-coupling: k = sqrt(1 - r^2).

    Inputs : r [-] in [0, 1]
    Outputs: k [-]
    """
    return float(np.sqrt(max(0.0, 1.0 - float(r) ** 2)))


def delta_n_for_coupler(k: float, lambda_m: float, lc_m: float) -> float:
    """
    Supermode index split Delta_n required for a directional coupler with
    coupling length Lc to reach amplitude cross-coupling k:
        k = sin(pi * Delta_n * Lc / lambda)  ->  Delta_n = (lambda/(pi*Lc))*asin(k)

    Inputs : k [-], lambda_m [m], lc_m [m]
    Outputs: Delta_n [-]
    """
    return (float(lambda_m) / (np.pi * float(lc_m))) * np.arcsin(np.clip(float(k), 0.0, 1.0))


def add_drop_through(phi, a: float, r1: float, r2: float):
    """
    Add-drop ring THROUGH-port power transmission (Bogaerts Eq. 5):
        T = (r2^2 a^2 - 2 r1 r2 a cos(phi) + r1^2)
            / (1 - 2 r1 r2 a cos(phi) + (r1 r2 a)^2)

    Inputs : phi [rad] (scalar or array), a [-], r1 [-], r2 [-]
    Outputs: T_through [-] (same shape as phi)
    """
    phi = np.asarray(phi, dtype=float)
    num = r2 ** 2 * a ** 2 - 2.0 * r1 * r2 * a * np.cos(phi) + r1 ** 2
    den = 1.0 - 2.0 * r1 * r2 * a * np.cos(phi) + (r1 * r2 * a) ** 2
    return num / den


def add_drop_drop(phi, a: float, r1: float, r2: float):
    """
    Add-drop ring DROP-port power transmission (Bogaerts Eq. 6):
        T = ((1-r1^2)(1-r2^2) a)
            / (1 - 2 r1 r2 a cos(phi) + (r1 r2 a)^2)

    Inputs : phi [rad] (scalar or array), a [-], r1 [-], r2 [-]
    Outputs: T_drop [-] (same shape as phi)
    """
    phi = np.asarray(phi, dtype=float)
    num = (1.0 - r1 ** 2) * (1.0 - r2 ** 2) * a
    den = 1.0 - 2.0 * r1 * r2 * a * np.cos(phi) + (r1 * r2 * a) ** 2
    return num / den


def mzi_delta_length_um(fsr_nm_target: float, n_group: float,
                        lambda0_nm: float) -> float:
    """
    Path-length imbalance dL of an unbalanced MZI for a target fringe FSR:
        FSR = lambda^2 / (n_g * dL)  ->  dL = lambda^2 / (n_g * FSR)

    Inputs : fsr_nm_target [nm], n_group [-], lambda0_nm [nm]
    Outputs: dL [um]
    Example: mzi_delta_length_um(10.0, 2.0, 1550.0) -> ~120.1 um
    """
    lam_m = float(lambda0_nm) * 1e-9
    fsr_m = float(fsr_nm_target) * 1e-9
    dL_m = lam_m ** 2 / (float(n_group) * fsr_m)
    return dL_m * 1e6


def quality_factor_from_fwhm(lambda0_nm: float, fwhm_nm: float) -> float:
    """
    Loaded quality factor from resonance FWHM: Q = lambda0 / FWHM.

    Inputs : lambda0_nm [nm], fwhm_nm [nm]
    Outputs: Q [-]
    """
    return float(lambda0_nm) / float(fwhm_nm)


def finesse(fsr_nm_val: float, fwhm_nm: float) -> float:
    """
    Resonator finesse F = FSR / FWHM.

    Inputs : fsr_nm_val [nm], fwhm_nm [nm]
    Outputs: F [-]
    """
    return float(fsr_nm_val) / float(fwhm_nm)
