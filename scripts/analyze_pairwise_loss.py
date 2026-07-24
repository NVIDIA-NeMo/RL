#!/usr/bin/env python3
"""Strict per-adapter loss comparison for production DP runs.

The production mbs=8 Multi-LoRA job emits one trace row per adapter, rank,
and optimizer step. Each true single emits one row per rank and optimizer step.
This tool joins rows by (optimizer_step, DP_rank), then requires exact input
SHA-256 and token-count alignment before comparing losses. It never computes or
plots a sum across adapters.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(root: Path) -> list[dict]:
    rows: list[dict] = []
    paths = sorted((root / "diag_loss_trace").glob("rank_*.jsonl"))
    if not paths:
        raise RuntimeError(f"no trace files under {root / 'diag_loss_trace'}")
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def select(rows: list[dict], who: str) -> list[dict]:
    return [r for r in rows if r.get("who") == who and r.get("loss") is not None]


def index_rows(rows: list[dict], who: str) -> dict[tuple[int, int], dict]:
    out: dict[tuple[int, int], dict] = {}
    for row in select(rows, who):
        key = (int(row["step"]), int(row["rank"]))
        if key in out:
            raise RuntimeError(f"{who}: duplicate trace key {key}")
        out[key] = row
    return out


def weighted_mean(rows: list[dict]) -> float:
    den = sum(int(r["num_tokens"]) for r in rows)
    if den <= 0:
        raise RuntimeError("zero token denominator")
    return sum(float(r["loss"]) * int(r["num_tokens"]) for r in rows) / den


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("multi", type=Path)
    ap.add_argument("single_a", type=Path)
    ap.add_argument("single_b", type=Path)
    ap.add_argument("single_c", type=Path)
    ap.add_argument("single_d", type=Path)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--dp-size", type=int, default=8)
    ap.add_argument(
        "--threshold", type=float, default=0.1,
        help="Blocking threshold for the token-weighted global per-adapter loss at each optimizer step.",
    )
    ap.add_argument(
        "--rank-local-threshold", type=float, default=None,
        help="Optional additional blocking threshold for individual DP-rank microbatch losses. "
             "By default rank-local differences are diagnostic only.",
    )
    ap.add_argument("--output-prefix", type=Path, required=True)
    ap.add_argument("--title", default="Production per-adapter loss comparison")
    args = ap.parse_args()

    expected_keys = {
        (step, rank)
        for step in range(1, args.steps + 1)
        for rank in range(args.dp_size)
    }
    multi_rows = load(args.multi)
    singles = {k: load(getattr(args, f"single_{k}")) for k in "abcd"}

    summary: dict[str, dict] = {}
    curves: dict[str, list[dict]] = {}
    all_ok = True

    for adapter in "abcd":
        who_m = f"multi_adapter_{adapter}"
        who_s = f"single_{adapter}"
        m = index_rows(multi_rows, who_m)
        s = index_rows(singles[adapter], who_s)
        missing_multi = sorted(expected_keys - set(m))
        extra_multi = sorted(set(m) - expected_keys)
        missing_single = sorted(expected_keys - set(s))
        extra_single = sorted(set(s) - expected_keys)
        aligned_keys = sorted(expected_keys & set(m) & set(s))

        hash_bad = []
        token_bad = []
        local_diffs = []
        for key in aligned_keys:
            sr, mr = s[key], m[key]
            if sr["input_sha256"] != mr["input_sha256"]:
                hash_bad.append(key)
            if int(sr["num_tokens"]) != int(mr["num_tokens"]):
                token_bad.append(key)
            local_diffs.append(abs(float(mr["loss"]) - float(sr["loss"])))

        step_rows = []
        for step in range(1, args.steps + 1):
            keys = [(step, rank) for rank in range(args.dp_size)]
            if not all(key in s and key in m for key in keys):
                continue
            s_step = [s[key] for key in keys]
            m_step = [m[key] for key in keys]
            single_loss = weighted_mean(s_step)
            multi_loss = weighted_mean(m_step)
            step_rows.append({
                "step": step,
                "single_loss": single_loss,
                "multi_loss": multi_loss,
                "abs_diff": abs(multi_loss - single_loss),
                "tokens": sum(int(r["num_tokens"]) for r in s_step),
            })

        max_local = max(local_diffs) if local_diffs else float("inf")
        max_step = max((r["abs_diff"] for r in step_rows), default=float("inf"))
        rank_local_ok = (
            args.rank_local_threshold is None
            or max_local < args.rank_local_threshold
        )
        ok = (
            not missing_multi
            and not extra_multi
            and not missing_single
            and not extra_single
            and not hash_bad
            and not token_bad
            and len(aligned_keys) == args.steps * args.dp_size
            and len(step_rows) == args.steps
            and max_step < args.threshold
            and rank_local_ok
        )
        all_ok &= ok
        summary[adapter] = {
            "pass": ok,
            "aligned_rank_rows": len(aligned_keys),
            "expected_rank_rows": args.steps * args.dp_size,
            "steps": len(step_rows),
            "missing_multi": len(missing_multi),
            "extra_multi": len(extra_multi),
            "missing_single": len(missing_single),
            "extra_single": len(extra_single),
            "hash_mismatches": len(hash_bad),
            "token_mismatches": len(token_bad),
            "max_local_abs_diff_diagnostic": max_local,
            "rank_local_threshold": args.rank_local_threshold,
            "max_step_abs_diff": max_step,
        }
        curves[adapter] = step_rows
        print(
            f"adapter {adapter}: {'PASS' if ok else 'FAIL'} "
            f"aligned={len(aligned_keys)}/{args.steps * args.dp_size} "
            f"steps={len(step_rows)}/{args.steps} "
            f"hash_bad={len(hash_bad)} token_bad={len(token_bad)} "
            f"max_local={max_local:.9f} max_step={max_step:.9f}"
        )

    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(
        json.dumps({
            "pass": all_ok,
            "global_step_threshold": args.threshold,
            "rank_local_threshold": args.rank_local_threshold,
            "steps": args.steps,
            "dp_size": args.dp_size,
            "adapters": summary,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["adapter", "step", "single_loss", "multi_loss", "abs_diff", "tokens"],
        )
        writer.writeheader()
        for adapter in "abcd":
            for row in curves[adapter]:
                writer.writerow({"adapter": adapter, **row})

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for ax, adapter in zip(axes.flat, "abcd"):
        rows = curves[adapter]
        x = [r["step"] for r in rows]
        ys = [r["single_loss"] for r in rows]
        ym = [r["multi_loss"] for r in rows]
        ax.plot(x, ys, lw=2, label=f"Single {adapter.upper()}")
        ax.plot(x, ym, "--", lw=2, label=f"Multi slot {adapter.upper()}")
        ax.set_title(
            f"Adapter {adapter.upper()} — max step |Δ|={summary[adapter]['max_step_abs_diff']:.3e}",
            weight="bold",
        )
        ax.set_xlabel("Optimizer step")
        ax.set_ylabel("Token-weighted per-adapter loss")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(
        f"{args.title}\nNo adapter sums; {args.dp_size} DP ranks aligned by (step, rank, SHA-256, tokens)",
        fontsize=15,
        weight="bold",
    )
    fig.savefig(prefix.with_suffix(".png"), dpi=180, bbox_inches="tight")
    print("OVERALL:", "PAIRWISE PASS" if all_ok else "PAIRWISE FAIL")
    print(f"wrote {prefix.with_suffix('.json')}")
    print(f"wrote {prefix.with_suffix('.csv')}")
    print(f"wrote {prefix.with_suffix('.png')}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
