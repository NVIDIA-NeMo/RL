#!/usr/bin/env python3
"""Rollout-time plots for the Ultra recipes.

Per-step timing/rollout/run_rollouts (s) from each rlvr1 run's ray-driver.log,
across MTP speculative-token counts. Regenerate: python3 images/plot_rollout_times.py
Outputs (PDF = vector, for the paper; PNG = preview) are written next to this script.
"""
import os
import statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "legend.fontsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.linewidth": 0.8,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# (MTP speculative tokens, color, marker, linestyle, per-step run_rollouts seconds)
RLVR1_MTP_RUNS = [
    (0, "#D55E00", "s", "--", [1801.3, 2008.3, 2066.0, 1857.6]),  # job 3035351
    (3, "#E69F00", "^", ":",  [1533.7, 1652.1, 1588.5, 1271.2]),  # job 3053875
    (5, "#009E73", "o", "-",  [1218.7, 1361.2, 1463.5, 1266.9]),  # job 3044848
    (7, "#0072B2", "D", "-.", [1307.0, 1344.4, 1490.5, 1301.3]),  # job 3053847
]


def _save(fig, name):
    for ext in ("pdf", "png"):
        path = os.path.join(OUT_DIR, f"{name}.{ext}")
        fig.savefig(path)
        print("saved:", path)
    plt.close(fig)


def plot_rlvr1_mtp_sweep():
    """Per-step rollout time across MTP speculative-token counts (rlvr1)."""
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    ymin, ymax = float("inf"), 0.0
    for n, color, marker, ls, ys in RLVR1_MTP_RUNS:
        xs = list(range(1, len(ys) + 1))
        ax.plot(xs, ys, marker=marker, linestyle=ls, color=color, label=f"MTP={n}")
        ymin, ymax = min(ymin, min(ys)), max(ymax, max(ys))

    pad = 0.10 * (ymax - ymin)
    ax.set_xlabel("Step")
    ax.set_ylabel("Rollout time (s)")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.grid(axis="y", alpha=0.3, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=10,
              columnspacing=1.0, handlelength=1.6, handletextpad=0.4,
              frameon=True, fancybox=False, edgecolor="0.7", framealpha=1.0)
    fig.tight_layout()
    _save(fig, "rlvr1_mtp_rollout")


def plot_rlvr1_mtp_mean():
    """Mean rollout time (±std over steps) vs MTP speculative-token count (rlvr1)."""
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    # Faint connector to show the trend; colored markers match the per-step figure.
    xs = [n for n, *_ in RLVR1_MTP_RUNS]
    means = [statistics.mean(ys) for *_, ys in RLVR1_MTP_RUNS]
    ax.plot(xs, means, color="0.5", linestyle="-", linewidth=1.5, zorder=1)
    for n, color, marker, _ls, ys in RLVR1_MTP_RUNS:
        ax.plot(n, statistics.mean(ys), marker=marker, markersize=9,
                color=color, zorder=2)

    ax.set_xlabel("Number of MTP tokens")
    ax.set_ylabel("Mean rollout time (s)")
    ax.set_xticks(xs)
    ax.grid(axis="y", alpha=0.3, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, "rlvr1_mtp_mean")


def plot_rlvr1_mtp_bar():
    """Bar chart of mean rollout time per MTP; the gap to the MTP=0 baseline is
    drawn and labeled with the speedup (rlvr1)."""
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    labels = [str(n) for n, *_ in RLVR1_MTP_RUNS]
    means = [statistics.mean(ys) for *_, ys in RLVR1_MTP_RUNS]
    colors = [c for _, c, *_ in RLVR1_MTP_RUNS]
    base = means[0]  # MTP=0 mean
    xs = list(range(len(means)))
    ax.bar(xs, means, color=colors, width=0.62, zorder=2)

    # Dashed line out from the top of the MTP=0 bar, extending across the others.
    ax.plot([0, xs[-1] + 0.35], [base, base], linestyle="--", color="0.35",
            linewidth=1.2, zorder=3)

    # Highlight only MTP=5: draw the gap up to the baseline, labeled with speedup.
    for x, run, m in zip(xs, RLVR1_MTP_RUNS, means):
        if run[0] != 5:
            continue
        ax.annotate("", xy=(x, base), xytext=(x, m),
                    arrowprops=dict(arrowstyle="<->", color="0.35", lw=1.4), zorder=4)
        ax.text(x, (m + base) / 2, f"{base / m:.2f}× Faster", ha="center", va="center",
                fontsize=11, zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))

    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Number of MTP tokens")
    ax.set_ylabel("Mean rollout time (s)")
    ax.set_ylim(1000, base * 1.12)
    ax.grid(axis="y", alpha=0.3, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, "rlvr1_mtp_bar")


def plot_rlvr1_mtp_speedup():
    """Bar chart of speedup vs MTP=0 per MTP speculative-token count (rlvr1)."""
    base = statistics.mean(RLVR1_MTP_RUNS[0][4])
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    labels = [str(n) for n, *_ in RLVR1_MTP_RUNS]
    speedups = [base / statistics.mean(ys) for *_, ys in RLVR1_MTP_RUNS]
    colors = [c for _, c, *_ in RLVR1_MTP_RUNS]
    bars = ax.bar(labels, speedups, color=colors, width=0.62)
    for b, s in zip(bars, speedups):
        ax.annotate(f"{s:.2f}×", (b.get_x() + b.get_width() / 2, s),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=10)
    ax.axhline(1.0, color="0.6", linewidth=1.0, linestyle="--", zorder=0)
    ax.set_xlabel("Number of MTP tokens")
    ax.set_ylabel("Speedup vs MTP=0 (×)")
    ax.set_ylim(0, max(speedups) * 1.15)
    ax.grid(axis="y", alpha=0.3, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, "rlvr1_mtp_speedup")


if __name__ == "__main__":
    plot_rlvr1_mtp_sweep()
    plot_rlvr1_mtp_mean()
    plot_rlvr1_mtp_bar()
    plot_rlvr1_mtp_speedup()
