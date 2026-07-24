#!/usr/bin/env python3
"""Native100 loss-curve charts, same format as the campaign charts:
per adapter x mode: top = single vs multi overlay, bottom = |delta| log panel.
Plus per-mode match-check figures: native |delta| trace vs campaign |delta| trace.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--native-results", type=Path, default=REPO_ROOT / "results")
ap.add_argument(
    "--reference-results",
    type=Path,
    default=None,
    help="Optional prior-campaign results directory for match-vs-reference charts",
)
ap.add_argument("--output", type=Path, default=REPO_ROOT / "charts_native100")
args = ap.parse_args()

NAT = args.native_results
CAMP = args.reference_results
OUT = args.output
OUT.mkdir(exist_ok=True)

CAMP_CSV = None if CAMP is None else {
    "noclip": CAMP / "code7x_100_noclip_pairwise_loss.csv",
    "clip1": CAMP / "code7x_100_clip1_pairwise_loss.csv",
}
NAT_CSV = {"noclip": NAT / "native100_noclip_pairwise.csv",
           "clip1": NAT / "native100_clip1_pairwise.csv"}


def load(path):
    per = defaultdict(dict)
    with open(path) as f:
        for r in csv.DictReader(f):
            per[r["adapter"]][int(r["step"])] = (
                float(r["single_loss"]), float(r["multi_loss"]), float(r["abs_diff"]))
    return per


for mode in ("noclip", "clip1"):
    nat = load(NAT_CSV[mode])
    camp = load(CAMP_CSV[mode]) if CAMP_CSV is not None else None
    for ad in "abcd":
        steps = sorted(nat[ad])
        s = [nat[ad][t][0] for t in steps]
        m = [nat[ad][t][1] for t in steps]
        d = [nat[ad][t][2] for t in steps]
        mean_d = sum(d) / len(d)

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(11, 7), sharex=True,
            gridspec_kw={"height_ratios": [2, 1], "hspace": 0.12})
        ax1.plot(steps, s, color="#1f77b4", lw=1.6, label=f"Single-LoRA {ad.upper()} (own job)")
        ax1.plot(steps, m, color="#d62728", lw=1.2, ls="--", label=f"Multi-LoRA slot {ad.upper()} (shared job)")
        ax1.set_ylabel("token-weighted loss")
        ax1.set_title(f"NATIVE NeMo-RL (zero nousnet) — adapter {ad.upper()} — {mode} — 100-step single vs multi")
        ax1.legend(loc="upper right", fontsize=9)
        ax1.grid(alpha=0.25)

        ax2.semilogy(steps, [max(x, 1e-8) for x in d], color="#7d3fc9", lw=1.2)
        ax2.axhline(0.1, color="#d62728", ls=":", lw=1.2, label="0.1 yardstick")
        ax2.axhline(mean_d, color="#444", ls="--", lw=1.0, label=f"mean {mean_d:.4f}")
        ax2.set_xlabel("optimizer step")
        ax2.set_ylabel("|Δ loss|")
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(alpha=0.25, which="both")
        fig.savefig(OUT / f"native100_{mode}_{ad}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

    if camp is None:
        continue

    # Optional match-check: native |delta| vs prior reference |delta| per adapter
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    fig.suptitle(f"Match check vs reference — {mode}: |Δ(single,multi)| per step — NATIVE (purple) vs REFERENCE (gray)", fontsize=12)
    for i, ad in enumerate("abcd"):
        ax = axes[i // 2][i % 2]
        steps_n = sorted(nat[ad]); dn = [max(nat[ad][t][2], 1e-8) for t in steps_n]
        steps_c = sorted(camp[ad]); dc = [max(camp[ad][t][2], 1e-8) for t in steps_c]
        mn = sum(dn) / len(dn); mc = sum(dc) / len(dc)
        ax.semilogy(steps_n, dn, color="#7d3fc9", lw=1.2, label=f"native (mean {mn:.4f})")
        ax.semilogy(steps_c, dc, color="#888888", lw=1.2, alpha=0.85, label=f"reference (mean {mc:.4f})")
        ax.axhline(0.1, color="#d62728", ls=":", lw=1.1)
        ax.set_title(f"adapter {ad.upper()}", fontsize=10)
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=8, loc="upper right")
        if i // 2 == 1:
            ax.set_xlabel("optimizer step")
        if i % 2 == 0:
            ax.set_ylabel("|Δ loss|")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / f"native100_{mode}_match_vs_campaign.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # stats
    print(f"== {mode}")
    for ad in "abcd":
        dn = [nat[ad][t][2] for t in sorted(nat[ad])]
        dc = [camp[ad][t][2] for t in sorted(camp[ad])]
        dn_s, dc_s = sorted(dn), sorted(dc)
        print(f"  {ad}: native mean={sum(dn)/len(dn):.4f} med={dn_s[50]:.4f} max={max(dn):.3f} | "
              f"reference mean={sum(dc)/len(dc):.4f} med={dc_s[50]:.4f} max={max(dc):.3f} | "
              f"native step1 s==m: {nat[ad][1][2] == 0.0} reference: {camp[ad][1][2] == 0.0}")
print("charts ->", OUT)
