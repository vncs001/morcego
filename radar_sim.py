"""FMCW radar signal simulator with micro-Doppler generation."""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class RadarParams:
    fc: float = 77e9
    bw: float = 4e9
    chirp_duration: float = 50e-6
    num_chirps: int = 128
    num_samples: int = 256
    fs: float = 10e6

    @property
    def lambda_(self) -> float:
        return 3e8 / self.fc

    @property
    def range_res(self) -> float:
        return 3e8 / (2 * self.bw)

    @property
    def max_range(self) -> float:
        return self.num_samples * self.range_res

    @property
    def velocity_res(self) -> float:
        return self.lambda_ / (2 * self.num_chirps * self.chirp_duration)

    @property
    def max_velocity(self) -> float:
        return self.lambda_ / (4 * self.chirp_duration)


@dataclass
class Target:
    label: str
    range_m: float
    velocity_mps: float
    rcs_dbsm: float
    blade_freq_hz: float = 0.0
    blade_amplitude: float = 0.0
    azimuth_deg: float = 0.0   # ângulo horizontal em graus (0 = frente, +90 = direita)


TARGETS = {
    "drone": Target(
        label="drone",
        range_m=80.0, velocity_mps=5.0, rcs_dbsm=-10.0,
        blade_freq_hz=150.0, blade_amplitude=0.65,
    ),
    "bird": Target(
        label="bird",
        range_m=50.0, velocity_mps=8.0, rcs_dbsm=-18.0,
        blade_freq_hz=6.0, blade_amplitude=0.12,
    ),
    "person": Target(
        label="person",
        range_m=30.0, velocity_mps=1.2, rcs_dbsm=0.0,
        blade_freq_hz=0.0, blade_amplitude=0.0,
    ),
    "car": Target(
        label="car",
        range_m=120.0, velocity_mps=14.0, rcs_dbsm=20.0,
        blade_freq_hz=0.0, blade_amplitude=0.0,
    ),
    "motorcycle": Target(
        label="motorcycle",
        range_m=95.0, velocity_mps=12.0, rcs_dbsm=8.0,
        blade_freq_hz=0.0, blade_amplitude=0.0,
    ),
}


class FMCWSimulator:
    def __init__(self, params: RadarParams = None):
        self.p = params or RadarParams()

    def generate_frame(self, targets: list, snr_db: float = 20.0):
        """
        Returns (rdm_db, iq_cube):
          rdm_db   — range-Doppler map in dB, shape (num_chirps, num_samples)
          iq_cube  — range-compressed IQ, shape (num_chirps, num_samples), complex
        """
        p = self.p
        noise_power = 10 ** (-snr_db / 10)
        t_chirp = np.linspace(0, p.chirp_duration, p.num_samples)
        iq_raw = np.zeros((p.num_chirps, p.num_samples), dtype=complex)

        for tgt in targets:
            amplitude = 10 ** (tgt.rcs_dbsm / 20)
            for ci in range(p.num_chirps):
                t_slow = ci * p.chirp_duration
                tau = 2 * tgt.range_m / 3e8
                range_phase = np.exp(1j * 2 * np.pi * p.bw / p.chirp_duration * tau * t_chirp)
                doppler_phase = np.exp(1j * 2 * np.pi * 2 * tgt.velocity_mps * t_slow / p.lambda_)
                if tgt.blade_freq_hz > 0:
                    md = 1.0 + tgt.blade_amplitude * np.sin(2 * np.pi * tgt.blade_freq_hz * t_slow)
                else:
                    md = 1.0
                iq_raw[ci] += amplitude * md * range_phase * doppler_phase

        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(*iq_raw.shape) + 1j * np.random.randn(*iq_raw.shape)
        )
        iq_raw += noise

        # range compress: FFT along fast time → (chirps, range bins)
        iq_rc = np.fft.fft(iq_raw * np.hanning(p.num_samples), axis=1)

        # Doppler FFT → range-Doppler map
        window2d = np.outer(np.hanning(p.num_chirps), np.ones(p.num_samples))
        rdm = np.fft.fftshift(np.fft.fft(iq_rc * window2d, axis=0), axes=0)
        rdm_db = 20 * np.log10(np.abs(rdm) + 1e-12)

        return rdm_db, iq_rc
