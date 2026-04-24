"""Real-time range-Doppler map visualization."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from radar_sim import RadarParams
from tracker import Track


class Visualizer:
    def __init__(self, params: RadarParams):
        self.p = params
        plt.ion()
        self.fig = plt.figure(figsize=(13, 6))
        gs = gridspec.GridSpec(1, 2, figure=self.fig)
        self.ax_rdm = self.fig.add_subplot(gs[0, 0])
        self.ax_tracks = self.fig.add_subplot(gs[0, 1])
        self.fig.tight_layout(pad=3)

    def update(self, rdm_db: np.ndarray, tracks: list[Track], frame: int, label: str = ""):
        p = self.p

        self.ax_rdm.clear()
        extent = [0, rdm_db.shape[1] * p.range_res, -p.max_velocity, p.max_velocity]
        im = self.ax_rdm.imshow(
            rdm_db, aspect="auto", origin="lower",
            extent=extent, cmap="inferno", vmin=-40, vmax=rdm_db.max(),
        )
        self.ax_rdm.set_xlabel("Range (m)")
        self.ax_rdm.set_ylabel("Velocity (m/s)")
        title = f"Range-Doppler Map — frame {frame}"
        if label:
            title += f"  [{label}]"
        self.ax_rdm.set_title(title)

        self.ax_tracks.clear()
        colors = {"drone": "red", "bird": "lime", "person": "yellow",
                  "car": "cyan", "motorcycle": "orange", "unknown": "white"}
        for t in tracks:
            c = colors.get(t.label, "white")
            self.ax_tracks.scatter(t.range_est, t.vel_est, color=c, s=100, zorder=5,
                                   edgecolors="white", linewidths=0.5)
            self.ax_tracks.annotate(
                f"{t.label}\n#{t.track_id}",
                (t.range_est, t.vel_est),
                fontsize=8, color=c,
                xytext=(6, 6), textcoords="offset points",
            )
        self.ax_tracks.set_facecolor("#111111")
        self.ax_tracks.set_xlim(0, p.max_range)
        self.ax_tracks.set_ylim(0, p.max_velocity * 1.1)
        self.ax_tracks.set_xlabel("Range (m)")
        self.ax_tracks.set_ylabel("Velocity (m/s)")
        self.ax_tracks.set_title("Active Tracks")
        self.ax_tracks.grid(True, alpha=0.2)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.08)

    def close(self):
        plt.ioff()
        plt.show()
