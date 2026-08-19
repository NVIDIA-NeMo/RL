#!/usr/bin/env python3
"""Attribute run-to-run refit variance to a stage and a rank pattern.

``summarize_refit_stages.py`` answers "how fast was this run". This answers "why
do two identical runs differ", which needs a different cut of the same records:
per-stage spread *across* runs, and whether the slowest rank is the same one each
time.

That distinction decides what is worth optimising. If the spread lives in one
stage, optimise that stage. If it lives in a rank that is persistently slow, the
problem is placement, not code. If the slow rank moves between runs, it is
contention, and a code change aimed at the stage will not reproduce.

Takes one JSONL file of ``MX_REFIT_STAGE`` payloads per run.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

# Every stage that contributes to accounted_s, in pipeline order.
STAGES = [
    "prepare_discover_s",
    "prepare_capture_s",
    "prepare_plan_s",
    "prepare_alloc_s",
    "prepare_register_s",
    "prepare_handshake_s",
    "wire_fused_s",
    "install_s",
]


def load(path: Path, warmup: int) -> list[dict]:
    rows = [
        json.loads(l)
        for l in path.read_text().splitlines()
        if l.strip().startswith("{")
    ]
    return [r for r in rows if r["step"] > warmup]


def fleet_critical(rows: list[dict]) -> list[tuple[int, float, int]]:
    """Per step: the slowest rank's accounted_s. A refit ends with its last rank."""
    out = []
    for step in sorted({r["step"] for r in rows}):
        at = [r for r in rows if r["step"] == step]
        worst = max(at, key=lambda r: r["accounted_s"])
        out.append((step, worst["accounted_s"], worst["rank"]))
    return out


def spread(vals: list[float]) -> float:
    lo = min(vals)
    return (max(vals) / lo) if lo > 0 else float("inf")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--warmup-steps", type=int, default=1)
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    runs = [load(p, args.warmup_steps) for p in args.runs]
    names = [p.stem for p in args.runs]

    report: dict = {"runs": names, "warmup_steps_excluded": args.warmup_steps}

    # 1. The headline each run would have quoted.
    print("== fleet-critical accounted_s per run ==")
    headline = []
    crit_ranks = []
    for name, rows in zip(names, runs):
        fc = fleet_critical(rows)
        med = statistics.median([v for _, v, _ in fc])
        headline.append(med)
        ranks = [rk for _, _, rk in fc]
        crit_ranks.append(ranks)
        mode = max(set(ranks), key=ranks.count)
        print(
            f"  {name:28s} median={med:7.3f}s  min={min(v for _, v, _ in fc):7.3f}  "
            f"max={max(v for _, v, _ in fc):7.3f}  critical rank most often={mode} "
            f"({ranks.count(mode)}/{len(ranks)} steps)"
        )
    report["headline_medians"] = headline
    report["headline_spread_x"] = spread(headline)
    print(f"  -> across-run spread: {spread(headline):.2f}x")

    # 2. Which stage carries the spread.
    #
    #    Measure the stages of the rank that was fleet-critical on each step, not
    #    the median over all ranks. Those answer different questions and can
    #    disagree sharply: the body of the per-rank distribution can shift by ~2x
    #    while the tail that actually sets the refit duration barely moves. Only the
    #    critical rank's time is on the critical path, so that is the cut that says
    #    what to optimise.
    print(
        "\n== stage medians of the fleet-critical rank (the one on the critical path) =="
    )
    hdr = (
        "  "
        + "stage".ljust(22)
        + "".join(n[:11].rjust(13) for n in names)
        + "   spread   share"
    )
    print(hdr)

    crit_stage_vals: list[dict[str, list[float]]] = []
    for rows in runs:
        per_stage: dict[str, list[float]] = {s: [] for s in STAGES}
        for step in sorted({r["step"] for r in rows}):
            at = [r for r in rows if r["step"] == step]
            worst = max(at, key=lambda r: r["accounted_s"])
            for s in STAGES:
                per_stage[s].append(worst.get(s, 0.0))
        crit_stage_vals.append(per_stage)

    stage_rows = {}
    totals = [sum(statistics.median(v[s]) for s in STAGES) for v in crit_stage_vals]
    for s in STAGES:
        meds = [statistics.median(v[s]) for v in crit_stage_vals]
        share = (
            (statistics.median(meds) / statistics.median(totals) * 100)
            if any(totals)
            else 0.0
        )
        active = all(m > 0 for m in meds)
        sp = spread(meds) if active else None
        stage_rows[s] = {
            "medians": meds,
            "spread_x": sp,
            "share_pct": share,
            "active": active,
        }
        sp_txt = f"{sp:6.2f}x" if active else "     --"
        print(
            "  "
            + s.ljust(22)
            + "".join(f"{m:13.4f}" for m in meds)
            + f"   {sp_txt}  {share:5.1f}%"
        )
    report["stages_critical_rank"] = stage_rows
    if any(not d["active"] for d in stage_rows.values()):
        print(
            "  ('--' = stage is zero in the warm window; it only runs on the cold step)"
        )

    ranked = sorted(
        (kv for kv in stage_rows.items() if kv[1]["active"]),
        key=lambda kv: (kv[1]["spread_x"] - 1.0) * kv[1]["share_pct"],
        reverse=True,
    )
    print("\n  stages ranked by contribution to the spread (excess spread x share):")
    for s, d in ranked[:4]:
        print(
            f"    {s:22s} spread {d['spread_x']:5.2f}x on {d['share_pct']:5.1f}% of the time"
        )

    # 3. Is the straggler the same rank every run? Placement vs contention.
    print("\n== per-rank max accounted_s ==")
    slowest = []
    for name, rows in zip(names, runs):
        per_rank = {
            rk: max(r["accounted_s"] for r in rows if r["rank"] == rk)
            for rk in sorted({r["rank"] for r in rows})
        }
        worst = max(per_rank, key=per_rank.get)
        best = min(per_rank, key=per_rank.get)
        slowest.append(worst)
        print(
            f"  {name:28s} slowest=rank {worst} ({per_rank[worst]:.3f}s)  "
            f"fastest=rank {best} ({per_rank[best]:.3f}s)  "
            f"ratio={per_rank[worst] / per_rank[best]:.2f}x"
        )
    report["slowest_rank_per_run"] = slowest
    if len(set(slowest)) == 1:
        print(
            f"  -> the SAME rank ({slowest[0]}) is slowest in every run: placement, not contention."
        )
    else:
        print(
            f"  -> the slowest rank MOVES ({slowest}): contention or scheduling, not a fixed rank."
        )

    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
