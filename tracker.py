"""Simple 1-D Kalman filter tracker for radar detections."""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class Track:
    track_id: int
    label: str
    range_est: float    # estimated range (m)
    vel_est: float      # estimated velocity (m/s)
    p_range: float = 100.0   # range variance
    p_vel: float = 10.0      # velocity variance
    age: int = 0
    misses: int = 0


class KalmanTracker:
    _next_id = 1

    def __init__(self, dt: float = 0.05, max_misses: int = 5, gate_m: float = 20.0):
        self.dt = dt
        self.max_misses = max_misses
        self.gate = gate_m
        self.tracks: list[Track] = []

    def _predict(self, t: Track):
        t.range_est += t.vel_est * self.dt
        t.p_range += t.p_vel * self.dt ** 2 + 0.5  # process noise

    def _update(self, t: Track, z_range: float, z_vel: float):
        # range update
        R_r = 4.0
        K_r = t.p_range / (t.p_range + R_r)
        t.range_est += K_r * (z_range - t.range_est)
        t.p_range *= (1 - K_r)
        # velocity update
        R_v = 1.0
        K_v = t.p_vel / (t.p_vel + R_v)
        t.vel_est += K_v * (z_vel - t.vel_est)
        t.p_vel *= (1 - K_v)

    def update(self, detections: list[tuple[float, float, str]]) -> list[Track]:
        """
        detections: list of (range_m, velocity_mps, label)
        Returns active tracks.
        """
        for t in self.tracks:
            self._predict(t)

        matched_det: set[int] = set()
        for t in self.tracks:
            best_dist, best_i = float("inf"), -1
            for i, (r, v, lbl) in enumerate(detections):
                dist = abs(t.range_est - r)
                if dist < best_dist:
                    best_dist, best_i = dist, i

            if best_i >= 0 and best_dist < self.gate and best_i not in matched_det:
                r, v, lbl = detections[best_i]
                self._update(t, r, v)
                t.label = lbl
                t.age += 1
                t.misses = 0
                matched_det.add(best_i)
            else:
                t.misses += 1

        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]

        for i, (r, v, lbl) in enumerate(detections):
            if i not in matched_det:
                t = Track(
                    track_id=KalmanTracker._next_id,
                    label=lbl,
                    range_est=r,
                    vel_est=v,
                )
                KalmanTracker._next_id += 1
                self.tracks.append(t)

        return self.tracks
