"""Extract classification features from range-Doppler map + IQ cube."""

import numpy as np
from dataclasses import dataclass
from radar_sim import RadarParams


@dataclass
class Features:
    velocity_mps: float
    peak_power_db: float      # proxy for RCS
    blade_cv: float           # coefficient of variation of slow-time amplitude at target range bin
                              # high → blade modulation (drone), low → rigid body (car/person)
    doppler_symmetry: float   # 1=symmetric (hovering), 0=asymmetric


def extract(rdm_db: np.ndarray, iq_rc: np.ndarray, params: RadarParams) -> Features:
    p = params

    # --- peak in RDM ---
    peak_idx = np.argmax(rdm_db)
    row, col = np.unravel_index(peak_idx, rdm_db.shape)

    # velocity from Doppler bin
    doppler_shifted = row - p.num_chirps // 2
    velocity = abs(doppler_shifted * p.velocity_res)
    peak_power_db = float(rdm_db[row, col])

    # --- micro-Doppler: slow-time amplitude variance at target range bin ---
    # iq_rc shape: (num_chirps, num_range_bins)
    # amplitude of each chirp at the target range bin
    slow_time_amp = np.abs(iq_rc[:, col])
    mean_amp = slow_time_amp.mean()
    if mean_amp > 1e-9:
        blade_cv = float(slow_time_amp.std() / mean_amp)
    else:
        blade_cv = 0.0

    # --- Doppler symmetry at target range bin ---
    half = p.num_chirps // 2
    doppler_col = rdm_db[:, col]
    pos_energy = 10 ** (doppler_col[half:].mean() / 10)
    neg_energy = 10 ** (doppler_col[:half].mean() / 10)
    symmetry = 1.0 - abs(pos_energy - neg_energy) / (pos_energy + neg_energy + 1e-9)

    return Features(
        velocity_mps=velocity,
        peak_power_db=peak_power_db,
        blade_cv=blade_cv,
        doppler_symmetry=symmetry,
    )
