#!/usr/bin/env python3
"""Generate performance comparison charts for FusedAttention vs UnfusedAttention on B200."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

sns.set_theme(style="whitegrid", font_scale=1.1)
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# Data
# ============================================================
configs = ["Fused+Opt", "Fused Base", "Unfused+Opt", "Unfused Base"]
colors = ["#2ecc71", "#82e0aa", "#e74c3c", "#f1948a"]

# GPT-OSS 20B (Step 5)
data_20b = {
    "E2E":      [216.6, 294.9, 600.0, 622.3],
    "Training":  [64.8, 113.2, 302.2, 309.3],
    "LogProb":   [44.6,  74.4, 193.6, 209.6],
    "Generation":[91.9,  89.2,  91.0,  87.5],
}

# GPT-OSS 120B (Step 5)
data_120b = {
    "E2E":      [193.7, 273.2, 278.8, 312.6],
    "Training":  [59.7, 108.4, 106.7, 129.3],
    "LogProb":   [38.1,  66.0,  73.5,  86.0],
    "Generation":[64.3,  64.9,  64.3,  64.1],
}


def plot_bar_chart(data, title, filename, phases=("E2E", "Training", "LogProb")):
    """Bar chart comparing configs across phases."""
    fig, axes = plt.subplots(1, len(phases), figsize=(5 * len(phases), 5))
    if len(phases) == 1:
        axes = [axes]

    for ax, phase in zip(axes, phases):
        vals = data[phase]
        bars = ax.barh(configs[::-1], vals[::-1], color=colors[::-1], edgecolor="white", height=0.6)

        # Add value labels
        for bar, val in zip(bars, vals[::-1]):
            ax.text(bar.get_width() + max(vals) * 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}s", va="center", fontweight="bold", fontsize=11)

        # Add speedup annotation
        best, worst = min(vals), max(vals)
        speedup = worst / best
        ax.set_title(f"{phase}\n({speedup:.2f}x speedup)", fontsize=13, fontweight="bold")
        ax.set_xlim(0, max(vals) * 1.25)
        ax.set_xlabel("Time (seconds)")

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(SAVE_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_stacked_bar(data, title, filename):
    """Stacked bar chart showing phase breakdown."""
    fig, ax = plt.subplots(figsize=(10, 5))

    phases = ["Training", "LogProb", "Generation", "Other"]
    phase_colors = ["#e74c3c", "#3498db", "#2ecc71", "#95a5a6"]

    # Calculate "Other"
    other = []
    for i in range(len(configs)):
        o = data["E2E"][i] - data["Training"][i] - data["LogProb"][i] - data["Generation"][i]
        other.append(max(0, o))

    all_data = [data["Training"], data["LogProb"], data["Generation"], other]

    y_pos = np.arange(len(configs))
    left = np.zeros(len(configs))

    for phase_data, color, label in zip(all_data, phase_colors, phases):
        bars = ax.barh(y_pos, phase_data, left=left, color=color, label=label,
                       edgecolor="white", height=0.6)
        # Add labels for significant phases
        for j, (val, l) in enumerate(zip(phase_data, left)):
            if val > data["E2E"][j] * 0.1:  # Only label if > 10% of total
                ax.text(l + val / 2, j, f"{val:.0f}s", ha="center", va="center",
                        fontsize=9, fontweight="bold", color="white")
        left += phase_data

    # Add total labels
    for i, total in enumerate(data["E2E"]):
        ax.text(total + max(data["E2E"]) * 0.02, i, f"{total:.1f}s",
                va="center", fontweight="bold", fontsize=11)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(configs)
    ax.set_xlabel("Time (seconds)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlim(0, max(data["E2E"]) * 1.2)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    path = os.path.join(SAVE_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_seqpack_impact(data, title, filename):
    """Compare Fused Base vs Fused+Opt to show SeqPack+MoePF impact."""
    fig, ax = plt.subplots(figsize=(8, 5))

    phases = ["E2E", "Training", "LogProb"]
    x = np.arange(len(phases))
    width = 0.35

    fused_base = [data[p][1] for p in phases]
    fused_opt = [data[p][0] for p in phases]

    bars1 = ax.bar(x - width / 2, fused_base, width, label="Fused Base (no SeqPack/MoePF)",
                   color="#82e0aa", edgecolor="white")
    bars2 = ax.bar(x + width / 2, fused_opt, width, label="Fused+Opt (SeqPack+MoePF ON)",
                   color="#2ecc71", edgecolor="white")

    # Add value labels and speedup
    for i, (base, opt) in enumerate(zip(fused_base, fused_opt)):
        ax.text(i - width / 2, base + max(fused_base) * 0.02, f"{base:.1f}s",
                ha="center", fontsize=10, fontweight="bold")
        ax.text(i + width / 2, opt + max(fused_base) * 0.02, f"{opt:.1f}s",
                ha="center", fontsize=10, fontweight="bold")
        speedup = base / opt
        ax.text(i, max(base, opt) + max(fused_base) * 0.08,
                f"{speedup:.2f}x", ha="center", fontsize=12, fontweight="bold", color="#c0392b")

    ax.set_ylabel("Time (seconds)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(phases)
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(fused_base) * 1.3)
    fig.tight_layout()
    path = os.path.join(SAVE_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_speedup_summary():
    """Summary chart of all speedup ratios."""
    fig, ax = plt.subplots(figsize=(10, 5))

    categories = [
        "Fused vs Unfused\n(20B Training)",
        "Fused vs Unfused\n(20B LogProb)",
        "Fused vs Unfused\n(20B E2E)",
        "Fused vs Unfused\n(120B Training)",
        "Fused vs Unfused\n(120B LogProb)",
        "Fused vs Unfused\n(120B E2E)",
    ]
    speedups = [4.67, 4.34, 2.77, 1.79, 1.93, 1.44]
    bar_colors = ["#e74c3c", "#3498db", "#2ecc71", "#e74c3c", "#3498db", "#2ecc71"]

    bars = ax.barh(categories[::-1], speedups[::-1], color=bar_colors[::-1],
                   edgecolor="white", height=0.6)

    for bar, val in zip(bars, speedups[::-1]):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}x", va="center", fontweight="bold", fontsize=12)

    ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Speedup (x)", fontsize=12)
    ax.set_title("FusedAttention Speedup vs UnfusedAttention\n(same SeqPack+MoePF config)",
                 fontsize=14, fontweight="bold")
    ax.set_xlim(0, max(speedups) * 1.2)
    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "perf_speedup_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# Generate all charts
plot_bar_chart(data_20b, "GPT-OSS 20B — Step Time Comparison (B200)", "perf_20b_bars.png")
plot_bar_chart(data_120b, "GPT-OSS 120B — Step Time Comparison (B200)", "perf_120b_bars.png")
plot_stacked_bar(data_20b, "GPT-OSS 20B — Phase Breakdown (B200, Step 5)", "perf_20b_stacked.png")
plot_stacked_bar(data_120b, "GPT-OSS 120B — Phase Breakdown (B200, Step 5)", "perf_120b_stacked.png")
plot_seqpack_impact(data_20b, "SeqPack+MoePF Impact on 20B (with FusedAttention)", "perf_20b_seqpack.png")
plot_seqpack_impact(data_120b, "SeqPack+MoePF Impact on 120B (with FusedAttention)", "perf_120b_seqpack.png")
plot_speedup_summary()

print("\nAll charts generated!")
