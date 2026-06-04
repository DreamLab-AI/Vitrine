#!/usr/bin/env python3
"""Generate all figures for the cultural heritage preservation report.

Outputs PDF vector figures to ../figures/ relative to this script.
"""

import os
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

NAVY   = "#1B2A4A"
TEAL   = "#2A7B88"
CORAL  = "#E07A5F"
GOLD   = "#D4A843"
SLATE  = "#5C6B7A"
LIGHT  = "#F0F2F5"
WHITE  = "#FFFFFF"

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         10,
    "axes.facecolor":    WHITE,
    "figure.facecolor":  WHITE,
    "axes.edgecolor":    SLATE,
    "axes.labelcolor":   NAVY,
    "xtick.color":       SLATE,
    "ytick.color":       SLATE,
    "axes.grid":         True,
    "grid.color":        "#DDE1E6",
    "grid.linewidth":    0.5,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.15,
})


def savefig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved {path}")


# ===================================================================
# 1. Pipeline Overview
# ===================================================================
def fig_pipeline_overview():
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis("off")

    stages = [
        ("Video",       0.2,  "#3B82F6"),  # blue  - input
        ("Frames",      1.7,  "#3B82F6"),
        ("COLMAP",      3.2,  TEAL),       # green - processing
        ("3DGS\nTraining", 4.7, TEAL),
        ("Segment-\nation", 6.2, TEAL),
        ("Mesh\nExtract", 7.7, TEAL),
        ("Blender",     9.2,  CORAL),      # orange - output
        ("USD /\nViewer", 10.7, CORAL),
    ]

    box_w, box_h = 1.2, 1.6
    y_center = 1.5

    for label, x, color in stages:
        rect = FancyBboxPatch(
            (x, y_center - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="white", linewidth=1.5, alpha=0.88,
        )
        ax.add_patch(rect)
        ax.text(x + box_w / 2, y_center, label,
                ha="center", va="center", fontsize=8.5,
                fontweight="bold", color="white", linespacing=1.3)

    for i in range(len(stages) - 1):
        x_start = stages[i][1] + box_w + 0.02
        x_end = stages[i + 1][1] - 0.02
        ax.annotate("", xy=(x_end, y_center), xytext=(x_start, y_center),
                     arrowprops=dict(arrowstyle="-|>", color=SLATE,
                                     lw=1.5, mutation_scale=14))

    # Legend
    legend_patches = [
        mpatches.Patch(color="#3B82F6", label="Input"),
        mpatches.Patch(color=TEAL, label="Processing"),
        mpatches.Patch(color=CORAL, label="Output"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", frameon=True,
              fontsize=8, fancybox=True, framealpha=0.9)

    fig.suptitle("3D Gaussian Splatting Pipeline for Heritage Preservation",
                 fontsize=12, fontweight="bold", color=NAVY, y=0.98)
    savefig(fig, "pipeline_overview.pdf")


# ===================================================================
# 2. Mesh Comparison (bar chart)
# ===================================================================
def fig_mesh_comparison():
    methods  = ["TSDF", "MILo", "SuGaR", "CoMe"]
    vertices = [122, 736, 200, 500]      # thousands
    time_min = [2, 17, 30, 20]
    quality  = [3, 7, 8, 9]

    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=False)

    colors = [CORAL, TEAL, GOLD, NAVY]

    # Vertices
    axes[0].bar(methods, vertices, color=colors, edgecolor="white", linewidth=0.8)
    axes[0].set_ylabel("Vertices (thousands)")
    axes[0].set_title("Mesh Vertex Count", fontweight="bold", color=NAVY)

    # Time
    axes[1].bar(methods, time_min, color=colors, edgecolor="white", linewidth=0.8)
    axes[1].set_ylabel("Time (minutes)")
    axes[1].set_title("Extraction Time", fontweight="bold", color=NAVY)

    # Quality
    axes[2].bar(methods, quality, color=colors, edgecolor="white", linewidth=0.8)
    axes[2].set_ylabel("Quality (1\u201310)")
    axes[2].set_title("Subjective Quality", fontweight="bold", color=NAVY)
    axes[2].set_ylim(0, 10)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Mesh Extraction Method Comparison",
                 fontsize=13, fontweight="bold", color=NAVY)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    savefig(fig, "mesh_comparison.pdf")


# ===================================================================
# 3. Training Comparison
# ===================================================================
def fig_training_comparison():
    strategies = ["MRNF 30K", "MCMC 15K", "MILo 18K"]
    gaussians  = [556, 1000, 46]   # thousands
    loss       = [0.011, 0.015, 0.0005]
    coverage   = [17, 1, None]     # percent; None = N/A
    time_min   = [5, 5, 17]

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    colors = [TEAL, CORAL, GOLD]
    x = np.arange(len(strategies))

    # Gaussians
    axes[0].bar(x, gaussians, color=colors, edgecolor="white")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(strategies, fontsize=8)
    axes[0].set_ylabel("Gaussians (K)")
    axes[0].set_title("Gaussian Count", fontweight="bold", color=NAVY)

    # Loss
    axes[1].bar(x, loss, color=colors, edgecolor="white")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(strategies, fontsize=8)
    axes[1].set_ylabel("Training Loss")
    axes[1].set_title("Final Loss", fontweight="bold", color=NAVY)

    # Time
    axes[2].bar(x, time_min, color=colors, edgecolor="white")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(strategies, fontsize=8)
    axes[2].set_ylabel("Minutes")
    axes[2].set_title("Training Time", fontweight="bold", color=NAVY)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Training Strategy Comparison",
                 fontsize=13, fontweight="bold", color=NAVY)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    savefig(fig, "training_comparison.pdf")


# ===================================================================
# 4. Architecture Diagram (two-container Docker)
# ===================================================================
def fig_architecture_diagram():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def draw_container(x, y, w, h, title, color, items):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                               facecolor=color, edgecolor=NAVY,
                               linewidth=2, alpha=0.15)
        ax.add_patch(rect)
        border = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                 facecolor="none", edgecolor=color,
                                 linewidth=2.5)
        ax.add_patch(border)
        ax.text(x + w / 2, y + h - 0.35, title, ha="center", va="top",
                fontsize=11, fontweight="bold", color=NAVY)
        for i, item in enumerate(items):
            ax.text(x + 0.3, y + h - 0.85 - i * 0.4, item,
                    fontsize=8, color=SLATE, family="monospace")

    # gaussian-toolkit container
    draw_container(0.5, 0.5, 4, 6, "gaussian-toolkit (GPU 0)", TEAL, [
        "COLMAP 3.11",
        "gsplat / nerfstudio",
        "SuGaR mesh extraction",
        "Open3D + Blender 4.3",
        "Jupyter :8888",
        "Viser viewer :8080",
        "Flask API :5000",
        "SSH :2222",
        "CUDA 12.4 runtime",
    ])

    # milo container
    draw_container(5.5, 1.5, 4, 4, "milo (GPU 1)", CORAL, [
        "MILo training",
        "MILo mesh extraction",
        "Custom CUDA kernels",
        "PyTorch 2.x",
        "Port :6006 (TBoard)",
    ])

    # Shared volumes
    vol_y = 0.1
    rect = FancyBboxPatch((2.5, vol_y), 5, 0.7, boxstyle="round,pad=0.08",
                           facecolor=GOLD, edgecolor=NAVY,
                           linewidth=1.5, alpha=0.25)
    ax.add_patch(rect)
    ax.text(5, vol_y + 0.35, "Shared Volumes:  /data  /models  /outputs",
            ha="center", va="center", fontsize=9, fontweight="bold", color=NAVY)

    # Arrows between containers
    ax.annotate("", xy=(5.5, 3.5), xytext=(4.5, 3.5),
                arrowprops=dict(arrowstyle="<->", color=NAVY, lw=2))
    ax.text(5, 3.75, "docker\nnetwork", ha="center", va="bottom",
            fontsize=7, color=SLATE)

    fig.suptitle("Two-Container GPU Architecture",
                 fontsize=13, fontweight="bold", color=NAVY, y=0.97)
    savefig(fig, "architecture_diagram.pdf")


# ===================================================================
# 5. Quality Bottleneck (funnel / waterfall)
# ===================================================================
def fig_quality_bottleneck():
    fig, ax = plt.subplots(figsize=(9, 5))

    stages   = ["Source\nVideo", "COLMAP\nSfM", "3DGS\nTraining", "Mesh\n(TSDF)", "Mesh\n(SuGaR)", "Mesh\n(MILo)"]
    quality  = [100, 95, 17, 5, 14, 12]
    colors_q = [NAVY, TEAL, CORAL, "#B0B0B0", GOLD, TEAL]

    bars = ax.bar(stages, quality, color=colors_q, edgecolor="white", linewidth=1.2, width=0.6)

    for bar, val in zip(bars, quality):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{val}%", ha="center", va="bottom", fontsize=10,
                fontweight="bold", color=NAVY)

    # Highlight bottleneck
    ax.annotate("BOTTLENECK", xy=(2, 17), xytext=(3.5, 60),
                fontsize=11, fontweight="bold", color=CORAL,
                arrowprops=dict(arrowstyle="-|>", color=CORAL, lw=2),
                ha="center")

    ax.set_ylabel("Estimated Coverage / Quality (%)", fontweight="bold")
    ax.set_ylim(0, 110)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("Quality Degradation Through the Pipeline",
                 fontsize=13, fontweight="bold", color=NAVY)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    savefig(fig, "quality_bottleneck.pdf")


# ===================================================================
# 6. Capture vs YouTube
# ===================================================================
def fig_capture_vs_youtube():
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    metrics = ["Resolution", "Frame\nRate", "Overlap\nControl", "Lighting\nConsistency", "COLMAP\nSuccess"]

    youtube_vals  = [4, 3, 2, 3, 3]
    capture_vals  = [9, 8, 9, 7, 9]

    y = np.arange(len(metrics))
    bar_h = 0.5

    axes[0].barh(y, youtube_vals, height=bar_h, color=CORAL, edgecolor="white")
    axes[0].set_xlim(10, 0)  # reversed
    axes[0].set_yticks([])
    axes[0].set_title("YouTube Source", fontweight="bold", color=CORAL, fontsize=11)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["left"].set_visible(False)
    for i, v in enumerate(youtube_vals):
        axes[0].text(v + 0.3, i, str(v), va="center", ha="left",
                     fontweight="bold", color=CORAL)

    axes[1].barh(y, capture_vals, height=bar_h, color=TEAL, edgecolor="white")
    axes[1].set_xlim(0, 10)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(metrics, fontsize=9)
    axes[1].yaxis.set_ticks_position("none")
    axes[1].set_title("Proper Capture", fontweight="bold", color=TEAL, fontsize=11)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    for i, v in enumerate(capture_vals):
        axes[1].text(v + 0.3, i, str(v), va="center", ha="left",
                     fontweight="bold", color=TEAL)

    # Central metric labels
    for i, m in enumerate(metrics):
        fig.text(0.5, 0.18 + i * 0.145, m, ha="center", va="center",
                 fontsize=9, fontweight="bold", color=NAVY,
                 transform=fig.transFigure)

    fig.suptitle("Input Source Quality Comparison (1\u201310 scale)",
                 fontsize=13, fontweight="bold", color=NAVY, y=0.98)
    fig.subplots_adjust(wspace=0.35)
    savefig(fig, "capture_vs_youtube.pdf")


# ===================================================================
# 7. Project Timeline (Gantt chart)
# ===================================================================
def fig_timeline():
    fig, ax = plt.subplots(figsize=(10, 3.5))

    phases = [
        ("Phase 1: Ideation workshops",   0, 4),
        ("Phase 2: Pipeline development",  2, 6),
        ("Phase 3: Capture and testing",   5, 8),
        ("Phase 4: Reporting",             8, 12),
    ]
    colors_t = [NAVY, TEAL, CORAL, GOLD]

    for i, (label, start, end) in enumerate(phases):
        ax.barh(i, end - start, left=start, height=0.55,
                color=colors_t[i], edgecolor="white", linewidth=1.2, alpha=0.88)
        ax.text(start + (end - start) / 2, i, label,
                ha="center", va="center", fontsize=8.5,
                fontweight="bold", color="white")

    ax.set_xlim(-0.5, 13)
    ax.set_yticks([])
    ax.set_xlabel("Weeks", fontweight="bold")
    ax.set_xticks(range(0, 13, 2))
    ax.set_xticklabels([f"W{w}" for w in range(0, 13, 2)])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.invert_yaxis()

    # Month markers
    for wk, label in [(0, "Month 1"), (4, "Month 2"), (8, "Month 3")]:
        ax.axvline(wk, color=SLATE, ls="--", lw=0.8, alpha=0.5)
        ax.text(wk + 0.1, -0.7, label, fontsize=7, color=SLATE)

    fig.suptitle("Project Timeline",
                 fontsize=13, fontweight="bold", color=NAVY, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    savefig(fig, "timeline.pdf")


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    print("Generating figures...")
    fig_pipeline_overview()
    fig_mesh_comparison()
    fig_training_comparison()
    fig_architecture_diagram()
    fig_quality_bottleneck()
    fig_capture_vs_youtube()
    fig_timeline()
    print("All figures generated.")
