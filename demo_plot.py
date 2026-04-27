"""
Gera uma figura estática comparando as assinaturas de radar de cada alvo.
Salva em demo_output.png (não precisa de janela aberta).

Uso:
  python demo_plot.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")   # sem janela — salva direto em PNG
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from radar_sim import FMCWSimulator, RadarParams, TARGETS
from feature_extractor import extract
from classifier import classify

LABELS = ["drone", "bird", "person", "car", "motorcycle"]
COLORS = {
    "drone": "#ff4444",
    "bird": "#44ff88",
    "person": "#ffdd44",
    "car": "#44ddff",
    "motorcycle": "#ff9944",
}

params = RadarParams()
sim = FMCWSimulator(params)

np.random.seed(42)

rdms = {}
feats_all = {}
cls_all = {}
cvs = {lbl: [] for lbl in LABELS}

# run 30 frames per target to get CV statistics
for lbl in LABELS:
    tgt = TARGETS[lbl]
    for _ in range(30):
        rdm_db, iq_rc = sim.generate_frame([tgt], snr_db=20.0)
        f = extract(rdm_db, iq_rc, params)
        cvs[lbl].append(f.blade_cv)
    # keep last frame for RDM display
    rdms[lbl] = rdm_db
    feats_all[lbl] = f
    cls_all[lbl] = classify(f)

# ── Layout ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 10), facecolor="#0d0d0d")
fig.suptitle(
    "Drone Detection Radar — Assinaturas de Micro-Doppler por Alvo",
    color="white", fontsize=14, fontweight="bold", y=0.98,
)

outer = gridspec.GridSpec(2, 1, figure=fig, hspace=0.45)

# Row 1: RDMs
rdm_gs = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=outer[0], wspace=0.35)

for col, lbl in enumerate(LABELS):
    ax = fig.add_subplot(rdm_gs[col])
    rdm = rdms[lbl]
    vmax = rdm.max()
    ax.imshow(
        rdm, aspect="auto", origin="lower",
        extent=[0, rdm.shape[1] * params.range_res, -params.max_velocity, params.max_velocity],
        cmap="inferno", vmin=vmax - 50, vmax=vmax,
    )
    cls = cls_all[lbl]
    ok = cls.label == lbl
    border_color = COLORS[lbl]
    for spine in ax.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(2)
    ax.set_title(
        f"{lbl.upper()}\n→ {cls.label} ({cls.confidence:.0%})",
        color=border_color, fontsize=9, fontweight="bold",
    )
    ax.set_xlabel("Range (m)", color="gray", fontsize=7)
    if col == 0:
        ax.set_ylabel("Velocity (m/s)", color="gray", fontsize=7)
    ax.tick_params(colors="gray", labelsize=6)
    ax.set_facecolor("#111111")

# Row 2: CV bar chart
ax_cv = fig.add_subplot(outer[1])
ax_cv.set_facecolor("#111111")

means = [np.mean(cvs[lbl]) for lbl in LABELS]
stds  = [np.std(cvs[lbl])  for lbl in LABELS]
bars = ax_cv.bar(
    LABELS, means, yerr=stds,
    color=[COLORS[l] for l in LABELS],
    error_kw=dict(ecolor="white", capsize=5, lw=1.5),
    width=0.5, zorder=3,
)

# threshold line
ax_cv.axhline(0.25, color="white", linestyle="--", lw=1.2, label="Drone threshold (CV=0.25)")
ax_cv.axhline(0.04, color="gray",  linestyle=":",  lw=1.0, label="Bird threshold (CV=0.04)")

for bar, mean in zip(bars, means):
    ax_cv.text(
        bar.get_x() + bar.get_width() / 2, mean + 0.01,
        f"{mean:.3f}", ha="center", va="bottom", color="white", fontsize=9,
    )

ax_cv.set_ylabel("Blade CV  (σ/μ amplitude slow-time)", color="white", fontsize=10)
ax_cv.set_title(
    "Coeficiente de Variação da Amplitude no Slow-Time — discriminador de micro-Doppler",
    color="white", fontsize=10,
)
ax_cv.tick_params(colors="white")
ax_cv.yaxis.label.set_color("white")
for spine in ax_cv.spines.values():
    spine.set_edgecolor("#444444")
ax_cv.grid(axis="y", alpha=0.2, zorder=0)
ax_cv.legend(facecolor="#222222", labelcolor="white", fontsize=9)

out_path = os.path.join(os.path.dirname(__file__), "demo_output.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Salvo em: {out_path}")
