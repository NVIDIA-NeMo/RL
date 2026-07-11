#!/usr/bin/env python3
"""Analyze the seven MX differentiator scenarios into one JSON schema.

The live producers (GRPO logs, elastic_bench.py, fanout_bench.py) write raw
artifacts. This tool turns them into comparable metrics and explicit assertions.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path


def emit(args, scenario, metrics, assertions, inputs=None):
    passed = all(item["passed"] for item in assertions)
    result = {
        "schema_version": 1,
        "scenario": scenario,
        "status": "pass" if passed else "fail",
        "metrics": metrics,
        "assertions": assertions,
        "inputs": inputs or {},
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0 if passed else 2


def assertion(name, passed, observed, expected):
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "expected": expected,
    }


def ep_filter(args):
    local = args.experts // args.rollout_ep
    expected = args.full_expert_bytes * local / args.experts
    actual = args.actual_bytes if args.actual_bytes is not None else expected
    reduction = 1 - actual / args.full_expert_bytes
    return emit(
        args,
        "ep_filter",
        {
            "global_experts": args.experts,
            "local_experts": local,
            "full_expert_bytes": args.full_expert_bytes,
            "actual_bytes": actual,
            "byte_reduction_fraction": reduction,
            "savings_factor": args.full_expert_bytes / actual,
        },
        [
            assertion(
                "local expert count",
                args.experts % args.rollout_ep == 0,
                local,
                f"{args.experts}/{args.rollout_ep}",
            ),
            assertion(
                "wire bytes match local expert ownership",
                math.isclose(actual, expected, rel_tol=args.tolerance),
                actual,
                expected,
            ),
        ],
    )


def tp_slice(args):
    expected = args.full_bytes / args.tp
    actual = args.actual_bytes if args.actual_bytes is not None else expected
    return emit(
        args,
        "tp_local_slice",
        {
            "tp_world_size": args.tp,
            "full_bytes": args.full_bytes,
            "actual_bytes_per_rank": actual,
            "expected_bytes_per_rank": expected,
            "byte_reduction_fraction": 1 - actual / args.full_bytes,
        },
        [
            assertion(
                "TP-local bytes",
                math.isclose(actual, expected, rel_tol=args.tolerance),
                actual,
                expected,
            )
        ],
    )


def partial(args):
    manifest = json.loads(Path(args.manifest).read_text())
    entries = manifest.get("tensors", manifest)
    selectors = args.selector
    selected = []
    total = 0
    for entry in entries:
        name = entry["name"]
        size = int(entry.get("bytes", entry.get("size", 0)))
        total += size
        if any(selector in name for selector in selectors):
            selected.append((name, size))
    chosen = sum(size for _, size in selected)
    return emit(
        args,
        "partial_refit",
        {
            "total_bytes": total,
            "selected_bytes": chosen,
            "selected_tensors": len(selected),
            "total_tensors": len(entries),
            "byte_reduction_fraction": 1 - chosen / total if total else 0,
        },
        [
            assertion("subset is non-empty", chosen > 0, chosen, "> 0"),
            assertion("subset prunes bytes", 0 < chosen < total, chosen, f"< {total}"),
        ],
        {"selectors": selectors, "manifest": args.manifest},
    )


def load_results(path):
    return [json.loads(item.read_text()) for item in sorted(Path(path).glob("result_*.json"))]


def elastic(args):
    rows = load_results(args.results)
    early = [row for row in rows if float(row.get("delay_s", 0)) == 0]
    late = [row for row in rows if float(row.get("delay_s", 0)) > 0]
    early_end = max(row["pull_end_epoch"] for row in early) if early else float("inf")
    late_start = min(row["pull_start_epoch"] for row in late) if late else float("-inf")
    rates = [float(row["gbps"]) for row in rows]
    return emit(
        args,
        "elastic_join",
        {
            "receivers": len(rows),
            "late_receivers": len(late),
            "early_finish_before_late_start_s": late_start - early_end,
            "median_gbps": statistics.median(rates) if rates else 0,
            "min_gbps": min(rates) if rates else 0,
        },
        [
            assertion("has early receiver", bool(early), len(early), ">= 1"),
            assertion("has late joiner", bool(late), len(late), ">= 1"),
            assertion(
                "early receivers finish independently",
                early_end < late_start,
                early_end,
                f"< late start {late_start}",
            ),
        ],
        {"results": args.results},
    )


def straggler(args):
    rows = load_results(args.results)
    healthy = [row for row in rows if float(row.get("delay_s", 0)) == 0]
    slow = [row for row in rows if float(row.get("delay_s", 0)) > 0]
    durations = [float(row["pull_dur_s"]) for row in healthy]
    healthy_p95 = max(durations) if durations else float("inf")
    baseline = statistics.median(durations) if durations else 0
    return emit(
        args,
        "straggler_isolation",
        {
            "healthy_receivers": len(healthy),
            "injected_stragglers": len(slow),
            "healthy_median_seconds": baseline,
            "healthy_max_seconds": healthy_p95,
        },
        [
            assertion("straggler injected", bool(slow), len(slow), ">= 1"),
            assertion(
                "healthy tail bounded",
                healthy_p95 <= baseline * args.max_slowdown if baseline else False,
                healthy_p95,
                f"<= {args.max_slowdown}x median",
            ),
        ],
        {"results": args.results},
    )


def fanout(args):
    def read_trial(path_string):
        path = Path(path_string)
        if path.is_file():
            return json.loads(path.read_text())
        rows = [
            json.loads(item.read_text())
            for item in sorted(path.glob("receiver_*.json"))
        ]
        if not rows:
            raise RuntimeError(f"No receiver JSON files under {path}")
        return {
            "workers": len(rows),
            "makespan_seconds": max(row["end_epoch"] for row in rows)
            - min(row["start_epoch"] for row in rows),
        }

    direct = read_trial(args.direct)
    tree = read_trial(args.tree)
    direct_s = float(direct["makespan_seconds"])
    tree_s = float(tree["makespan_seconds"])
    speedup = direct_s / tree_s
    return emit(
        args,
        "tree_fanout",
        {
            "direct_makespan_seconds": direct_s,
            "tree_makespan_seconds": tree_s,
            "speedup": speedup,
            "workers": int(tree.get("workers", direct.get("workers", 0))),
        },
        [
            assertion(
                "tree beats direct",
                speedup >= args.min_speedup,
                speedup,
                f">= {args.min_speedup}",
            )
        ],
        {"direct": args.direct, "tree": args.tree},
    )


TRANSFER_RE = re.compile(
    r"RDMA transfer complete: (?P<gb>[0-9.]+) GB, .*? "
    r"(?P<seconds>[0-9.]+)s, (?P<gbps>[0-9.]+) Gbps "
    r"\(step=(?P<step>\d+), source_rank=(?P<rank>\d+), "
    r"source_id=(?P<source>[0-9a-f]+)\)"
)


def egress(args):
    rows = []
    for line in Path(args.log).read_text(errors="replace").splitlines():
        match = TRANSFER_RE.search(line)
        if match:
            rows.append(
                {
                    "source_rank": int(match["rank"]),
                    "source_id": match["source"],
                    "step": int(match["step"]),
                    "bytes": float(match["gb"]) * 1e9,
                    "seconds": float(match["seconds"]),
                    "gbps": float(match["gbps"]),
                }
            )
    by_rank = {}
    for row in rows:
        by_rank.setdefault(row["source_rank"], 0)
        by_rank[row["source_rank"]] += row["bytes"]
    values = list(by_rank.values())
    mean = statistics.mean(values) if values else 0
    cv = statistics.pstdev(values) / mean if mean else float("inf")
    return emit(
        args,
        "trainer_egress_balance",
        {
            "transfers": len(rows),
            "sources": len(by_rank),
            "bytes_by_source_rank": by_rank,
            "byte_coefficient_of_variation": cv,
            "median_source_gbps": statistics.median([row["gbps"] for row in rows])
            if rows
            else 0,
        },
        [
            assertion("source logs present", bool(rows), len(rows), ">= 1"),
            assertion(
                "egress balanced",
                cv <= args.max_cv,
                cv,
                f"<= {args.max_cv}",
            ),
        ],
        {"log": args.log},
    )


def parser():
    root = argparse.ArgumentParser()
    root.add_argument("--out")
    sub = root.add_subparsers(dest="scenario", required=True)

    ep = sub.add_parser("ep-filter")
    ep.add_argument("--experts", type=int, default=128)
    ep.add_argument("--rollout-ep", type=int, required=True)
    ep.add_argument("--full-expert-bytes", type=float, required=True)
    ep.add_argument("--actual-bytes", type=float)
    ep.add_argument("--tolerance", type=float, default=0.02)
    ep.set_defaults(run=ep_filter)

    tp = sub.add_parser("tp-slice")
    tp.add_argument("--tp", type=int, required=True)
    tp.add_argument("--full-bytes", type=float, required=True)
    tp.add_argument("--actual-bytes", type=float)
    tp.add_argument("--tolerance", type=float, default=0.02)
    tp.set_defaults(run=tp_slice)

    part = sub.add_parser("partial")
    part.add_argument("--manifest", required=True)
    part.add_argument("--selector", action="append", required=True)
    part.set_defaults(run=partial)

    el = sub.add_parser("elastic")
    el.add_argument("--results", required=True)
    el.set_defaults(run=elastic)

    st = sub.add_parser("straggler")
    st.add_argument("--results", required=True)
    st.add_argument("--max-slowdown", type=float, default=1.25)
    st.set_defaults(run=straggler)

    fan = sub.add_parser("fanout")
    fan.add_argument("--direct", required=True)
    fan.add_argument("--tree", required=True)
    fan.add_argument("--min-speedup", type=float, default=1.05)
    fan.set_defaults(run=fanout)

    eg = sub.add_parser("egress")
    eg.add_argument("--log", required=True)
    eg.add_argument("--max-cv", type=float, default=0.05)
    eg.set_defaults(run=egress)
    return root


def main():
    args = parser().parse_args()
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
