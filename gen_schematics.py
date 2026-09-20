"""Generate schematic (non-data) figures for the paper.

These diagrams illustrate structural concepts already stated in the text
(fixed vs growing storage; lookup vs superposition retrieval; the failure
taxonomy). They contain NO measured data and are labeled as schematics.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

NAVY = "#1f3864"
LIGHT = "#ebeef4"
GRAY = "#5a5a5a"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})


def box(ax, xy, w, h, text, fontsize=10, fc=LIGHT, ec=NAVY):
    b = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02",
                       fc=fc, ec=ec, lw=1.5)
    ax.add_patch(b)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=NAVY)


def arrow(ax, p1, p2):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", lw=1.5,
                                 color=GRAY, mutation_scale=14))


def finalize(ax, fname, w=10, h=4.2):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, h + 0.6)
    ax.axis("off")
    fig = ax.figure
    fig.set_size_inches(w, h + 0.6)
    fig.tight_layout()
    fig.savefig(f"public_evidence/figures/{fname}", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote", fname)


# Figure 1: growing per-token slots vs one fixed bank, small N and large N.
ax = plt.subplots()[1]
ax.text(2.0, 4.4, "Softmax: slots grow with N", ha="center", fontsize=12,
        color=NAVY, weight="bold")
ax.text(7.8, 4.4, "Linear: one fixed bank", ha="center", fontsize=12,
        color=NAVY, weight="bold")
for i in range(3):
    box(ax, (0.5 + i * 1.05, 3.0), 0.95, 0.8, f"(k{i}, v{i})", fontsize=9)
ax.text(3.9, 3.4, "+ …", fontsize=12, color=GRAY)
box(ax, (6.9, 3.0), 1.8, 0.8, "S (fixed k×dv)", fontsize=9)
ax.text(2.0, 2.2, "small N: 3 slots vs 1 bank", ha="center", fontsize=10,
        color=GRAY)
for i in range(6):
    box(ax, (0.2 + i * 0.62, 0.7), 0.55, 0.7, "", fc="#f4f6fa")
ax.text(4.15, 1.05, "+ … (keeps growing: O(N))", fontsize=10, color=GRAY)
box(ax, (6.9, 0.7), 1.8, 0.7, "S (same bank)", fontsize=9)
ax.text(7.8, 0.15, "large N: still 1 bank", ha="center", fontsize=10,
        color=GRAY)
finalize(ax, "schematic_conflict.png")

# Figure 2: lookup vs similarity retrieval.
ax = plt.subplots()[1]
ax.text(2.0, 4.4, "Softmax retrieval: lookup", ha="center", fontsize=12,
        color=NAVY, weight="bold")
ax.text(7.8, 4.4, "Linear retrieval: similarity", ha="center", fontsize=12,
        color=NAVY, weight="bold")
box(ax, (1.4, 3.0), 1.2, 0.8, "query q")
for i in range(3):
    box(ax, (0.5 + i * 1.05, 1.5), 0.95, 0.8, f"(k{i}, v{i})", fontsize=9)
arrow(ax, (2.0, 3.0), (2.05, 2.3))
ax.text(0.2, 0.7, "output = v_j for matching slot j", fontsize=10, color=GRAY)
box(ax, (7.2, 3.0), 1.2, 0.8, "query q")
box(ax, (6.6, 1.5), 2.4, 0.8, "S = Σ φ(kᵢ)vᵢᵀ (one bank)")
arrow(ax, (7.8, 3.0), (7.8, 2.3))
ax.text(5.3, 0.7, "output = Σᵢ overlap(q, kᵢ)·vᵢ", fontsize=10, color=GRAY)
finalize(ax, "schematic_storage.png")

# Figure 4: taxonomy flowchart.
ax = plt.subplots()[1]
box(ax, (0.3, 2.2), 2.9, 1.2, "(a) Associative\ninterference", fontsize=10)
box(ax, (3.6, 2.2), 2.9, 1.2, "(b) Temporal\nattenuation", fontsize=10)
box(ax, (6.9, 2.2), 2.9, 1.2, "(c) Numerical\nprecision loss", fontsize=10)
box(ax, (3.1, 0.4), 3.9, 1.0, "joint symptom:\ninformation dissipation",
     fontsize=10, fc="#fdf3e7", ec="#8a6d3b")
arrow(ax, (1.75, 2.2), (4.2, 1.4))
arrow(ax, (5.05, 2.2), (5.05, 1.4))
arrow(ax, (8.35, 2.2), (5.9, 1.4))
ax.text(5.05, 3.8, "distinct mechanisms → shared symptom",
        ha="center", fontsize=12, color=NAVY, weight="bold")
finalize(ax, "schematic_taxonomy.png", h=4.6)
