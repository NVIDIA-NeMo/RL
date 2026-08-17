#!/usr/bin/env python3
"""Summarize MX_REFIT_STAGE records into the numbers the benchmark contract asks for.

Reads one JSON object per line (the payload after the ``MX_REFIT_STAGE`` marker)
and reports, over the measured warm window only:

* the fleet-critical latency per step, i.e. the slowest rank, because a refit is
  not finished until its last rank is;
* min / median / p95 / max of that critical value across steps;
* per-rank maxima, so a single persistently slow rank is visible;
* stage attribution, to satisfy the ">=95% attributed or <=100 ms
  unattributed" gate before any number is quoted.

Throughput is deliberately reported per rank. Ranks overlap in wall clock but
are not measured in one shared window, so summing them would overstate the
fleet; the aggregate is labelled an upper bound.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

# Per-step stages known at the time of writing. Kept only to flag records that
# carry a stage this tool has never seen; the attribution arithmetic below uses
# whatever per-step stages each record actually contains, so a stage added on the
# MX side can never silently masquerade as unattributed time.
KNOWN_STAGE_FIELDS = (
    "descriptor_build_s",
    "wire_fused_s",
    "install_s",
    "reslice_s",
)


def stage_fields(record: dict) -> list[str]:
    """Per-step stage keys in a record.

    ``accounted_s`` is the total being decomposed, and ``prepare_*`` covers
    one-time discovery/handshake work that sits outside the per-step total, so
    both are excluded.
    """
    return [
        key
        for key in record
        if key.endswith("_s")
        and key != "accounted_s"
        and not key.startswith("prepare_")
    ]


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile; avoids interpolating across few samples."""
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _attribution_gate(framework_e2e_s: float | None, mx_critical_s: float) -> dict:
    """Attribution measured against the cost the framework actually pays.

    The benchmark contract's ">=95% attributed or <=100 ms unattributed" gate is
    about explaining a refit, so the denominator has to be the framework-visible
    refit time. Measuring MX's stages against MX's own ``accounted_s`` subtotal
    always yields ~100% and would pass this gate while an arbitrary amount of
    per-refit cost sits outside MX entirely.
    """
    if framework_e2e_s is None:
        return {
            "gate_pass": None,
            "reason": (
                "no --framework-e2e-s supplied; attribution is unknown. MX stages "
                "alone cannot establish it."
            ),
        }
    unattributed = framework_e2e_s - mx_critical_s
    attributed_pct = 100.0 * mx_critical_s / framework_e2e_s
    return {
        "framework_e2e_s": framework_e2e_s,
        "mx_critical_s": mx_critical_s,
        "unattributed_s": unattributed,
        "attributed_pct": attributed_pct,
        "gate_pass": unattributed <= 0.100 or attributed_pct >= 95.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", type=Path)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1,
        help="Leading steps to exclude as cold/warm-up.",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--framework-e2e-s",
        type=float,
        help=(
            "Median warm framework-visible refit seconds "
            "(timing/.../transfer_and_update_weights). Required to evaluate the "
            "attribution gate, because MX's own stages sum to accounted_s by "
            "construction and cannot reveal cost outside MX."
        ),
    )
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.records.read_text().splitlines() if line.strip()]
    if not rows:
        print("no records", file=sys.stderr)
        return 1

    steps = sorted({r["step"] for r in rows})
    measured_steps = steps[args.warmup_steps :]
    measured = [r for r in rows if r["step"] in measured_steps]
    ranks = sorted({r["rank"] for r in rows})

    # A refit ends when its slowest rank ends, so the fleet-critical figure is a
    # per-step max over ranks -- never a mean.
    critical = []
    for step in measured_steps:
        per_step = [r["accounted_s"] for r in measured if r["step"] == step]
        critical.append(max(per_step))

    per_rank_max = {
        rank: max(r["accounted_s"] for r in measured if r["rank"] == rank) for rank in ranks
    }

    unattributed = [
        r["accounted_s"] - sum(r[f] for f in stage_fields(r)) for r in measured
    ]
    worst_unattributed = max(unattributed)
    novel_stages = sorted(
        {f for r in measured for f in stage_fields(r)} - set(KNOWN_STAGE_FIELDS)
    )
    if novel_stages:
        print(
            f"note: records carry stage(s) unknown to this tool: {novel_stages}; "
            "counted as attributed. Add them to KNOWN_STAGE_FIELDS.",
            file=sys.stderr,
        )
    attribution_pct = 100.0 * (
        1.0 - worst_unattributed / max(r["accounted_s"] for r in measured)
    )

    bytes_per_rank = {r["bytes"] for r in measured}
    wire = [r["wire_fused_s"] for r in measured]
    one_rank_bytes = next(iter(bytes_per_rank))
    per_rank_gbps = [8 * one_rank_bytes / w / 1e9 for w in wire]

    fallback_total = sum(r.get("fallback", 0) for r in rows)
    full_pull_total = sum(r.get("full_pull_sources", 0) for r in rows)

    summary = {
        "schema": "refit-summary-v1",
        "ranks": len(ranks),
        "steps_total": len(steps),
        "warmup_steps_excluded": args.warmup_steps,
        "measured_steps": measured_steps,
        "bytes_per_rank": sorted(bytes_per_rank),
        "fleet_critical_accounted_s": {
            "min": min(critical),
            "median": statistics.median(critical),
            "p95": percentile(critical, 0.95),
            "max": max(critical),
        },
        "per_rank_max_accounted_s": {
            "min": min(per_rank_max.values()),
            "max": max(per_rank_max.values()),
            "slowest_rank": max(per_rank_max, key=per_rank_max.get),
        },
        "per_rank_wire_gbps": {
            "min": min(per_rank_gbps),
            "median": statistics.median(per_rank_gbps),
            "max": max(per_rank_gbps),
        },
        "aggregate_wire_gbps_upper_bound": len(ranks) * statistics.median(per_rank_gbps),
        # Internal consistency only. MX's per-step stages are defined to sum to
        # accounted_s, so a high number here is near-tautological: it proves the
        # stage fields were parsed, not that the refit's cost is understood. It is
        # deliberately NOT named "attribution" and carries no gate.
        "internal_stage_consistency": {
            "worst_residual_s": worst_unattributed,
            "consistency_pct_worst_case": attribution_pct,
        },
        "stage_attribution": _attribution_gate(
            args.framework_e2e_s, statistics.median(critical)
        ),
        "correctness_markers": {
            "fallback_total": fallback_total,
            "full_pull_sources_total": full_pull_total,
        },
    }

    print(json.dumps(summary, indent=2))
    if args.json_out:
        args.json_out.write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
