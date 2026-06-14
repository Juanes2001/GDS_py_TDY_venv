"""
step6_kappa_variants.py — coupling-coefficient variant arrays (FWHM 300 pm and 100 pm).

Defines the alternative coupling-strength arrays used by the FWHM
comparison: RING_KAPPA_*_03 (FWHM ~300 pm) and RING_KAPPA_*_01
(FWHM ~100 pm), kept verbatim so the variant sweeps reproduce the
original FWHM-vs-finesse study. (Core arrays = FWHM ~500 pm.)
"""
from __future__ import annotations

import math
import time
from datetime import datetime

import numpy as np
import h5py

from .config import *  # shared platform constants & paths

# ─────────────────────────────────────────────────────────
#  Module constants / design knobs for this step
# ─────────────────────────────────────────────────────────
RING_KAPPA_INPUT_SQ_03 = np.array([
    0.090264, 0.089812, 0.089773, 0.089734,
    0.089695, 0.089655, 0.089616, 0.089577,
    0.090212, 0.090172, 0.090132, 0.090098,
    0.090058, 0.090018,
])
RING_KAPPA_DROP_SQ_03 = np.array([
    0.087734, 0.087257, 0.087217, 0.087176,
    0.087135, 0.087094, 0.087053, 0.087012,
    0.087627, 0.087586, 0.087544, 0.087508,
    0.087466, 0.087425,
])

RING_KAPPA_INPUT_SQ_01 = np.array([
    0.031052, 0.030891, 0.030877, 0.030863,
    0.030849, 0.030835, 0.030821, 0.030807,
    0.031033, 0.031019, 0.031005, 0.030993,
    0.030979, 0.030964,
])
RING_KAPPA_DROP_SQ_01 = np.array([
    0.028358, 0.028177, 0.028155, 0.028140,
    0.028124, 0.028109, 0.028093, 0.028077,
    0.028280, 0.028264, 0.028249, 0.028235,
    0.028219, 0.028203,
])



def run(state=None):
    """Execute this step. `state` carries the shared namespace
    between steps (the notebook's old kernel globals + bridge).
    Returns the updated `state`."""
    state = {} if state is None else state
    global RING_KAPPA_DROP_SQ_01, RING_KAPPA_DROP_SQ_03, RING_KAPPA_INPUT_SQ_01, RING_KAPPA_INPUT_SQ_03
    globals().update(state)


    state.update({k: globals().get(k) for k in [
        'RING_KAPPA_DROP_SQ_01', 'RING_KAPPA_DROP_SQ_03', 'RING_KAPPA_INPUT_SQ_01', 'RING_KAPPA_INPUT_SQ_03',
    ] if k in globals()})
    return state


if __name__ == "__main__":
    run()
