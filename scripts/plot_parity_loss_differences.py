#!/usr/bin/env python3
"""Plot parity-battery pairwise losses with an explicit |delta| panel per adapter.

Reads results/<prefix>_<mode>_pairwise.csv (from analyze_pairwise_loss.py) and
writes results/<prefix>_<mode>_loss_with_delta.png for mode in {noclip, clip1}.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--prefix", default="parity100",
                help="results/<prefix>_<mode>_pairwise.csv input stem")
ap.add_argument("--steps", type=int, default=100)
args = ap.parse_args()

for mode, label in (
    ("noclip", "NO GRADIENT CLIPPING"),
    ("clip1", "INDEPENDENT PER-ADAPTER GRADIENT CLIPPING, max_norm=1.0"),
):
    src = ROOT / f"results/{args.prefix}_{mode}_pairwise.csv"
    rows = list(csv.DictReader(src.open(encoding="utf-8")))
    fig, axes = plt.subplots(
        4,
        2,
        figsize=(15, 16),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [2.5, 1]},
    )
    for i, adapter in enumerate("abcd"):
        ar = sorted(
            (r for r in rows if r["adapter"] == adapter),
            key=lambda r: int(r["step"]),
        )
        if len(ar) != args.steps:
            raise RuntimeError(
                f"adapter {adapter}: expected {args.steps} steps, found {len(ar)}"
            )
        x = [int(r["step"]) for r in ar]
        single = [float(r["single_loss"]) for r in ar]
        multi = [float(r["multi_loss"]) for r in ar]
        diff = [float(r["abs_diff"]) for r in ar]
        imax = max(range(len(diff)), key=diff.__getitem__)
        violations = sum(d >= 0.1 for d in diff)

        ax, ad = axes[i]
        ax.plot(x, single, lw=1.8, label=f"True single {adapter.upper()}")
        ax.plot(x, multi, "--", lw=1.8, label=f"Multi slot {adapter.upper()}")
        ax.set_title(f"Adapter {adapter.upper()} losses (same y-axis)", weight="bold")
        ax.set_xlabel("Optimizer step")
        ax.set_ylabel("Token-weighted per-adapter loss")
        ax.grid(alpha=0.25)
        ax.legend()

        color = "crimson" if violations else "#166534"
        ad.plot(x, diff, color=color, lw=1.5)
        ad.axhline(0.1, color="black", ls=":", lw=1, label="0.1 yardstick")
        ad.scatter([x[imax]], [diff[imax]], color="black", s=24, zorder=3)
        # Keep the max callout inside the axes and away from the two-line title.
        y_max = max(0.12, diff[imax] * 1.28)
        y_text = min(diff[imax] + y_max * 0.08, y_max * 0.87)
        x_text = min(x[imax] + 5, 78)
        ad.annotate(
            f"max={diff[imax]:.4f} @ step {x[imax]}",
            xy=(x[imax], diff[imax]),
            xytext=(x_text, y_text),
            textcoords="data",
            arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "black"},
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "0.6", "alpha": 0.92},
        )
        ad.set_title(
            f"|Multi − Single|: {'FAIL' if violations else 'PASS'}\n"
            f"mean={sum(diff)/len(diff):.4f} · final={diff[-1]:.4f} · "
            f"≥0.1: {violations}/{args.steps}",
            weight="bold",
            fontsize=10,
            pad=9,
        )
        ad.set_xlabel("Optimizer step")
        ad.set_ylabel("Absolute loss difference")
        ad.set_ylim(0, y_max)
        ad.grid(alpha=0.25)
        ad.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        f"Multi-LoRA parity battery: {args.steps}-step per-adapter loss comparison — {label}\n"
        "left: raw curves, right: absolute difference\n"
        "8 data-parallel ranks; rows aligned by step, rank, input hash, and tokens",
        fontsize=15,
        weight="bold",
    )
    out = ROOT / f"results/{args.prefix}_{mode}_loss_with_delta.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(out)
