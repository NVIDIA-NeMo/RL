#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Run a resumable TQ checkpoint ablation matrix and generate reports.

Invoke this script through the already-synced NeMo-RL environment:

    uv run --no-sync python tools/run_tq_checkpoint_benchmark_matrix.py \
        --checkpoint-root /lustre/.../tq-ckpt-benchmark \
        --suite core --suite-name core-$(date +%Y%m%d-%H%M%S) \
        --repetitions 3

Outputs are refreshed after every run:

    <checkpoint-root>/<suite-name>/
    ├── state.json
    ├── report.md
    ├── report.json
    ├── runs.csv
    ├── summary.csv
    ├── logs/
    └── runs/

Re-run the same command with the same ``--suite-name`` after a Slurm restart.
Completed cells are skipped and interrupted/failed cells receive a new retry
directory, preserving all prior evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

SuiteName = Literal["smoke", "core", "production", "all"]

GIB = 1024**3

SERIES_TITLES = {
    "smoke": "Smoke test",
    "payload": "Payload profile",
    "storage": "Storage-unit fan-out",
    "topology": "Concurrent producer topology",
    "row_cardinality": "Row cardinality at fixed token volume",
    "raggedness": "Sequence-length raggedness",
    "production": "Production-scale resident rows",
}

RUN_COLUMNS = [
    "case_id",
    "repetition",
    "run_name",
    "status",
    "payload_profile",
    "num_rows",
    "min_seq_len",
    "max_seq_len",
    "num_storage_units",
    "producer_mode",
    "num_producers",
    "valid_tokens",
    "logical_gib",
    "checkpoint_gib",
    "save_s",
    "load_s",
    "effective_save_gib_s",
    "effective_load_gib_s",
    "disk_logical_ratio",
    "base_fill_gib_s",
    "producer_before_rows_s",
    "producer_during_rows_s",
    "producer_after_rows_s",
    "producer_during_p95_ms",
    "overlap_observed",
    "verification_status",
    "result_path",
    "log_path",
    "error",
]

AGGREGATE_METRICS = [
    "valid_tokens",
    "logical_gib",
    "checkpoint_gib",
    "save_s",
    "load_s",
    "effective_save_gib_s",
    "effective_load_gib_s",
    "disk_logical_ratio",
    "base_fill_gib_s",
    "producer_before_rows_s",
    "producer_during_rows_s",
    "producer_after_rows_s",
    "producer_during_p95_ms",
]


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    description: str
    series: tuple[str, ...]
    num_rows: int
    min_seq_len: int
    max_seq_len: int
    payload_profile: str
    num_storage_units: int
    producer_mode: str = "quiescent"
    num_producers: int = 4
    batch_rows: int = 128
    producer_batch_rows: int = 32


def _baseline() -> BenchmarkCase:
    return BenchmarkCase(
        case_id="baseline",
        description="8K rows × 4K tokens, train-ready, 4 storage units",
        series=(
            "payload",
            "storage",
            "topology",
            "row_cardinality",
            "raggedness",
        ),
        num_rows=8192,
        min_seq_len=4096,
        max_seq_len=4096,
        payload_profile="train-ready",
        num_storage_units=4,
    )


def _core_cases() -> list[BenchmarkCase]:
    return [
        _baseline(),
        BenchmarkCase(
            case_id="payload-generation",
            description="Generation-only fields at baseline shape",
            series=("payload",),
            num_rows=8192,
            min_seq_len=4096,
            max_seq_len=4096,
            payload_profile="generation",
            num_storage_units=4,
        ),
        BenchmarkCase(
            case_id="storage-su1",
            description="One storage unit",
            series=("storage",),
            num_rows=8192,
            min_seq_len=4096,
            max_seq_len=4096,
            payload_profile="train-ready",
            num_storage_units=1,
        ),
        BenchmarkCase(
            case_id="storage-su2",
            description="Two storage units",
            series=("storage",),
            num_rows=8192,
            min_seq_len=4096,
            max_seq_len=4096,
            payload_profile="train-ready",
            num_storage_units=2,
        ),
        BenchmarkCase(
            case_id="storage-su8",
            description="Eight storage units",
            series=("storage",),
            num_rows=8192,
            min_seq_len=4096,
            max_seq_len=4096,
            payload_profile="train-ready",
            num_storage_units=8,
        ),
        BenchmarkCase(
            case_id="topology-thread4",
            description="Four threads write while checkpointing",
            series=("topology",),
            num_rows=8192,
            min_seq_len=4096,
            max_seq_len=4096,
            payload_profile="train-ready",
            num_storage_units=4,
            producer_mode="thread",
            num_producers=4,
        ),
        BenchmarkCase(
            case_id="topology-process4",
            description="Four Ray processes write while checkpointing",
            series=("topology",),
            num_rows=8192,
            min_seq_len=4096,
            max_seq_len=4096,
            payload_profile="train-ready",
            num_storage_units=4,
            producer_mode="process",
            num_producers=4,
        ),
        BenchmarkCase(
            case_id="rows-32768x1024",
            description="32K rows × 1K tokens (fixed total tokens)",
            series=("row_cardinality",),
            num_rows=32768,
            min_seq_len=1024,
            max_seq_len=1024,
            payload_profile="train-ready",
            num_storage_units=4,
            batch_rows=256,
        ),
        BenchmarkCase(
            case_id="rows-4096x8192",
            description="4K rows × 8K tokens (fixed total tokens)",
            series=("row_cardinality",),
            num_rows=4096,
            min_seq_len=8192,
            max_seq_len=8192,
            payload_profile="train-ready",
            num_storage_units=4,
        ),
        BenchmarkCase(
            case_id="rows-2048x16384",
            description="2K rows × 16K tokens (fixed total tokens)",
            series=("row_cardinality",),
            num_rows=2048,
            min_seq_len=16384,
            max_seq_len=16384,
            payload_profile="train-ready",
            num_storage_units=4,
            batch_rows=64,
        ),
        BenchmarkCase(
            case_id="ragged-1024-7168",
            description="Moderately ragged lengths with mean near 4K",
            series=("raggedness",),
            num_rows=8192,
            min_seq_len=1024,
            max_seq_len=7168,
            payload_profile="train-ready",
            num_storage_units=4,
        ),
        BenchmarkCase(
            case_id="ragged-128-8064",
            description="Highly ragged lengths with mean near 4K",
            series=("raggedness",),
            num_rows=8192,
            min_seq_len=128,
            max_seq_len=8064,
            payload_profile="train-ready",
            num_storage_units=4,
        ),
    ]


def _production_cases() -> list[BenchmarkCase]:
    cases = []
    for seq_len in (1024, 4096, 8192, 32768, 65536, 131072):
        for profile in ("generation", "train-ready"):
            cases.append(
                BenchmarkCase(
                    case_id=f"prod-131072x{seq_len}-{profile}",
                    description=(f"131K resident rows × {seq_len} tokens, {profile}"),
                    series=("production",),
                    num_rows=131072,
                    min_seq_len=seq_len,
                    max_seq_len=seq_len,
                    payload_profile=profile,
                    num_storage_units=8,
                    batch_rows=256,
                )
            )
    return cases


def build_cases(suite: SuiteName) -> list[BenchmarkCase]:
    if suite == "smoke":
        return [
            BenchmarkCase(
                case_id="smoke",
                description="Small end-to-end correctness smoke",
                series=("smoke",),
                num_rows=128,
                min_seq_len=128,
                max_seq_len=256,
                payload_profile="generation",
                num_storage_units=2,
                batch_rows=32,
            )
        ]
    if suite == "core":
        return _core_cases()
    if suite == "production":
        return _production_cases()
    if suite == "all":
        return [*_core_cases(), *_production_cases()]
    raise ValueError(f"unknown suite: {suite}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _new_state(
    *,
    suite: str,
    suite_name: str,
    repetitions: int,
    cases: list[BenchmarkCase],
) -> dict[str, Any]:
    affinity = None
    if hasattr(os, "sched_getaffinity"):
        affinity = len(os.sched_getaffinity(0))
    return {
        "schema_version": 1,
        "suite": suite,
        "suite_name": suite_name,
        "repetitions": repetitions,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "git_commit": _git_commit(),
        "runner": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "cpu_count": os.cpu_count(),
            "cpu_affinity_count": affinity,
            "TQ_NUM_THREADS": os.environ.get("TQ_NUM_THREADS"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "cases": _case_signature(cases),
        "runs": [],
    }


def _case_signature(cases: list[BenchmarkCase]) -> list[dict[str, Any]]:
    signatures = []
    for case in cases:
        signature = asdict(case)
        # JSON restores tuples as lists. Normalize before persisting/comparing
        # so resuming does not report a false configuration drift.
        signature["series"] = list(case.series)
        signatures.append(signature)
    return signatures


def _validate_existing_state(
    state: dict[str, Any],
    *,
    suite: str,
    repetitions: int,
    cases: list[BenchmarkCase],
) -> None:
    if state["suite"] != suite:
        raise ValueError(
            f"suite directory contains {state['suite']!r}, not requested {suite!r}"
        )
    if int(state["repetitions"]) != repetitions:
        raise ValueError(
            "cannot change --repetitions when resuming a suite: "
            f"existing={state['repetitions']}, requested={repetitions}"
        )
    if state["cases"] != _case_signature(cases):
        raise ValueError(
            "benchmark case definitions changed since this suite began; "
            "start a new --suite-name to avoid mixing configurations"
        )


def _successful_run(
    state: dict[str, Any],
    case_id: str,
    repetition: int,
) -> dict[str, Any] | None:
    for run in reversed(state["runs"]):
        if (
            run["case_id"] == case_id
            and int(run["repetition"]) == repetition
            and run["status"] == "success"
            and Path(run["result_path"]).exists()
        ):
            return run
    return None


def _reconcile_completed_runs(state: dict[str, Any]) -> bool:
    """Reconcile runs left active when a previous runner disappeared."""
    changed = False
    for run in state["runs"]:
        if run["status"] == "success":
            continue
        result_path = Path(run["result_path"])
        valid_result = False
        if result_path.exists():
            try:
                result = _load_json(result_path)
            except (OSError, json.JSONDecodeError):
                pass
            else:
                valid_result = (
                    _optional(result, "load", "verification", "status") == "pass"
                )
        if valid_result:
            run["status"] = "success"
            run["finished_at"] = run.get("finished_at") or _utc_now()
            run["returncode"] = 0
            run["error"] = None
            changed = True
        elif run["status"] == "running":
            run["status"] = "interrupted"
            run["finished_at"] = _utc_now()
            run["error"] = "previous runner exited before producing a valid result"
            changed = True
    return changed


def _next_run_name(
    state: dict[str, Any],
    case_id: str,
    repetition: int,
) -> str:
    base = f"{case_id}-r{repetition:02d}"
    attempts = sum(
        1
        for run in state["runs"]
        if run["case_id"] == case_id and int(run["repetition"]) == repetition
    )
    return base if attempts == 0 else f"{base}-retry{attempts:02d}"


def _command(
    *,
    python: str,
    benchmark_script: Path,
    runs_root: Path,
    run_name: str,
    case: BenchmarkCase,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        python,
        str(benchmark_script),
        "--checkpoint-root",
        str(runs_root),
        "--run-name",
        run_name,
        "--num-rows",
        str(case.num_rows),
        "--min-seq-len",
        str(case.min_seq_len),
        "--max-seq-len",
        str(case.max_seq_len),
        "--payload-profile",
        case.payload_profile,
        "--batch-rows",
        str(case.batch_rows),
        "--num-storage-units",
        str(case.num_storage_units),
        "--producer-mode",
        case.producer_mode,
        "--num-producers",
        str(case.num_producers),
        "--producer-batch-rows",
        str(case.producer_batch_rows),
        "--producer-max-rows",
        str(args.producer_max_rows),
        "--producer-warmup-s",
        str(args.producer_warmup_s),
        "--producer-cooldown-s",
        str(args.producer_cooldown_s),
        "--producer-sleep-ms",
        str(args.producer_sleep_ms),
        "--verify-mode",
        args.verify_mode,
        "--verify-samples",
        str(args.verify_samples),
        "--phase-timeout-s",
        str(args.phase_timeout_s),
        "--torch-num-threads",
        str(args.torch_num_threads),
    ]
    if args.ray_address:
        command.extend(["--ray-address", args.ray_address])
    if args.skip_space_check:
        command.append("--skip-space-check")
    return command


def _run_command(command: list[str], log_path: Path) -> int:
    with log_path.open("w") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        log.flush()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return process.wait()


def _optional(mapping: dict[str, Any], *path: str) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
        if value is None:
            return None
    return value


def flatten_result(
    *,
    case: BenchmarkCase,
    repetition: int,
    run_record: dict[str, Any],
) -> dict[str, Any]:
    row = {column: None for column in RUN_COLUMNS}
    row.update(
        {
            "case_id": case.case_id,
            "repetition": repetition,
            "run_name": run_record["run_name"],
            "status": run_record["status"],
            "payload_profile": case.payload_profile,
            "num_rows": case.num_rows,
            "min_seq_len": case.min_seq_len,
            "max_seq_len": case.max_seq_len,
            "num_storage_units": case.num_storage_units,
            "producer_mode": case.producer_mode,
            "num_producers": case.num_producers,
            "result_path": run_record.get("result_path"),
            "log_path": run_record.get("log_path"),
            "error": run_record.get("error"),
        }
    )
    if run_record["status"] != "success":
        return row

    result = _load_json(Path(run_record["result_path"]))
    logical_bytes = int(_optional(result, "load", "restored", "logical_tensor_bytes"))
    checkpoint_bytes = int(_optional(result, "save", "checkpoint", "disk_bytes"))
    row.update(
        {
            "valid_tokens": _optional(
                result, "load", "restored", "lengths", "total_valid_tokens"
            ),
            "logical_gib": logical_bytes / GIB,
            "checkpoint_gib": checkpoint_bytes / GIB,
            "save_s": _optional(result, "save", "checkpoint", "duration_s"),
            "load_s": _optional(result, "load", "checkpoint", "load_duration_s"),
            "effective_save_gib_s": _optional(
                result, "summary", "effective_save_gib_per_s"
            ),
            "effective_load_gib_s": _optional(
                result, "summary", "effective_load_gib_per_s"
            ),
            "disk_logical_ratio": _optional(
                result, "summary", "checkpoint_to_logical_ratio"
            ),
            "base_fill_gib_s": _optional(
                result, "save", "base_fill", "put_logical_gib_per_s"
            ),
            "producer_before_rows_s": _optional(
                result, "save", "producers", "before", "rows_per_s"
            ),
            "producer_during_rows_s": _optional(
                result, "save", "producers", "during", "rows_per_s"
            ),
            "producer_after_rows_s": _optional(
                result, "save", "producers", "after", "rows_per_s"
            ),
            "producer_during_p95_ms": _optional(
                result,
                "save",
                "producers",
                "during",
                "put_latency_p95_ms",
            ),
            "overlap_observed": _optional(
                result, "save", "producers", "overlap_observed"
            ),
            "verification_status": _optional(result, "load", "verification", "status"),
        }
    )
    return row


def metric_stats(values: list[float | int | None]) -> dict[str, float | None]:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return {"mean": None, "stdev": None, "min": None, "max": None}
    return {
        "mean": statistics.mean(numeric),
        "stdev": statistics.stdev(numeric) if len(numeric) > 1 else 0.0,
        "min": min(numeric),
        "max": max(numeric),
    }


def aggregate_rows(
    rows: list[dict[str, Any]],
    cases: list[BenchmarkCase],
) -> list[dict[str, Any]]:
    aggregates = []
    for case in cases:
        case_rows = [row for row in rows if row["case_id"] == case.case_id]
        successful = [row for row in case_rows if row["status"] == "success"]
        aggregate: dict[str, Any] = {
            "case_id": case.case_id,
            "description": case.description,
            "series": list(case.series),
            "attempted": len(case_rows),
            "successful": len(successful),
            "failed": sum(row["status"] == "failed" for row in case_rows),
            "producer_mode": case.producer_mode,
            "payload_profile": case.payload_profile,
            "num_rows": case.num_rows,
            "min_seq_len": case.min_seq_len,
            "max_seq_len": case.max_seq_len,
            "num_storage_units": case.num_storage_units,
            "num_producers": case.num_producers,
            "overlap_passes": sum(
                row["overlap_observed"] is True for row in successful
            ),
        }
        for metric in AGGREGATE_METRICS:
            aggregate[metric] = metric_stats([row[metric] for row in successful])
        aggregates.append(aggregate)
    return aggregates


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if not math.isfinite(float(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def _fmt_stats(stats: dict[str, float | None], digits: int = 3) -> str:
    if stats["mean"] is None:
        return "—"
    if stats["stdev"] in (None, 0.0):
        return _fmt(stats["mean"], digits)
    return f"{_fmt(stats['mean'], digits)} ± {_fmt(stats['stdev'], digits)}"


def _workload(case: BenchmarkCase) -> str:
    if case.min_seq_len == case.max_seq_len:
        length = f"{case.min_seq_len:,}"
    else:
        length = f"{case.min_seq_len:,}–{case.max_seq_len:,}"
    return f"{case.num_rows:,} × {length}"


def render_markdown(
    *,
    state: dict[str, Any],
    cases: list[BenchmarkCase],
    rows: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
) -> str:
    aggregate_by_id = {item["case_id"]: item for item in aggregates}
    observed_system: dict[str, Any] = {}
    for row in rows:
        if row["status"] != "success" or not row.get("result_path"):
            continue
        try:
            result = _load_json(Path(row["result_path"]))
        except (OSError, json.JSONDecodeError):
            continue
        observed_system = _optional(result, "save", "system") or {}
        break
    successful_runs = sum(row["status"] == "success" for row in rows)
    target_runs = len(cases) * int(state["repetitions"])
    lines = [
        "# TransferQueue checkpoint benchmark report",
        "",
        f"- Suite: `{state['suite_name']}` (`{state['suite']}`)",
        f"- Generated: `{_utc_now()}`",
        f"- Progress: `{successful_runs}/{target_runs}` successful runs",
        f"- Git commit: `{state.get('git_commit') or 'unknown'}`",
        f"- Host: `{state['runner']['hostname']}`",
        f"- Allocated CPU affinity: `{state['runner'].get('cpu_affinity_count')}`",
        f"- `TQ_NUM_THREADS`: `{state['runner'].get('TQ_NUM_THREADS')}`",
        f"- TransferQueue: `{observed_system.get('transfer_queue_version', 'pending')}`",
        f"- Filesystem: `{observed_system.get('filesystem', 'pending')}`",
        f"- Repetitions per case: `{state['repetitions']}`",
        "",
        "Values are mean ± sample standard deviation across successful repetitions.",
        "",
    ]

    for series, title in SERIES_TITLES.items():
        series_cases = [case for case in cases if series in case.series]
        if not series_cases:
            continue
        lines.extend(
            [
                f"## {title}",
                "",
                "| Case | n | Workload | Profile | SU | Producers | Logical GiB | Save s | Save GiB/s | Load s | Load GiB/s | Overlap |",
                "|---|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for case in series_cases:
            item = aggregate_by_id[case.case_id]
            producer = (
                "none"
                if case.producer_mode == "quiescent"
                else f"{case.producer_mode}:{case.num_producers}"
            )
            overlap = (
                "—"
                if case.producer_mode == "quiescent"
                else f"{item['overlap_passes']}/{item['successful']}"
            )
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{case.case_id}`",
                        str(item["successful"]),
                        _workload(case),
                        case.payload_profile,
                        str(case.num_storage_units),
                        producer,
                        _fmt_stats(item["logical_gib"]),
                        _fmt_stats(item["save_s"]),
                        _fmt_stats(item["effective_save_gib_s"]),
                        _fmt_stats(item["load_s"]),
                        _fmt_stats(item["effective_load_gib_s"]),
                        overlap,
                    ]
                )
                + " |"
            )
        lines.append("")

    failures = [row for row in rows if row["status"] != "success"]
    lines.extend(["## Failures and interrupted attempts", ""])
    if failures:
        lines.extend(
            [
                "| Case | Repetition | Run | Status | Error | Log |",
                "|---|---:|---|---|---|---|",
            ]
        )
        for row in failures:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{row['case_id']}`",
                        str(row["repetition"]),
                        f"`{row['run_name']}`",
                        row["status"],
                        str(row.get("error") or "—").replace("|", "\\|"),
                        f"`{row.get('log_path') or '—'}`",
                    ]
                )
                + " |"
            )
    else:
        lines.append("No failed or interrupted attempts recorded.")
    lines.extend(
        [
            "",
            "## Interpretation notes",
            "",
            "- A concurrent case is valid only when every successful repetition reports write/save overlap.",
            "- Compare effective GiB/s for concurrent cases; in-flight writes can change the restored byte count.",
            "- Load follows save on the same node and may benefit from the Linux page cache.",
            "- No clear operations are issued during checkpointing because TQ v0.1.9 does not make that race consistent.",
            "- Every load occurs in a fresh Python/TQ/Ray process and verifies all base keys plus the guaranteed pre-save concurrent-write set.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_csv_rows(
    aggregates: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    columns = [
        "case_id",
        "description",
        "series",
        "attempted",
        "successful",
        "failed",
        "payload_profile",
        "num_rows",
        "min_seq_len",
        "max_seq_len",
        "num_storage_units",
        "producer_mode",
        "num_producers",
        "overlap_passes",
    ]
    for metric in AGGREGATE_METRICS:
        columns.extend(
            [
                f"{metric}_mean",
                f"{metric}_stdev",
                f"{metric}_min",
                f"{metric}_max",
            ]
        )
    flattened = []
    for aggregate in aggregates:
        row = {
            key: (",".join(value) if key == "series" else value)
            for key, value in aggregate.items()
            if key not in AGGREGATE_METRICS
        }
        for metric in AGGREGATE_METRICS:
            for statistic, value in aggregate[metric].items():
                row[f"{metric}_{statistic}"] = value
        flattened.append(row)
    return columns, flattened


def refresh_reports(
    suite_root: Path,
    state: dict[str, Any],
    cases: list[BenchmarkCase],
) -> None:
    case_by_id = {case.case_id: case for case in cases}
    rows = [
        flatten_result(
            case=case_by_id[run["case_id"]],
            repetition=int(run["repetition"]),
            run_record=run,
        )
        for run in state["runs"]
        if run["status"] in {"success", "failed", "interrupted"}
    ]
    aggregates = aggregate_rows(rows, cases)
    report = {
        "state": state,
        "runs": rows,
        "aggregates": aggregates,
    }
    _atomic_json(suite_root / "report.json", report)
    (suite_root / "report.md").write_text(
        render_markdown(
            state=state,
            cases=cases,
            rows=rows,
            aggregates=aggregates,
        )
        + "\n"
    )
    _write_csv(suite_root / "runs.csv", rows, RUN_COLUMNS)
    aggregate_columns, aggregate_csv = _aggregate_csv_rows(aggregates)
    _write_csv(suite_root / "summary.csv", aggregate_csv, aggregate_columns)


def _estimate_logical_gib(case: BenchmarkCase) -> float:
    average_length = (case.min_seq_len + case.max_seq_len) / 2
    bytes_per_token = 14 if case.payload_profile == "generation" else 20
    return case.num_rows * (average_length * bytes_per_token + 12) / GIB


def _print_cases(cases: list[BenchmarkCase], repetitions: int) -> None:
    print(
        f"{len(cases)} cases × {repetitions} repetitions = "
        f"{len(cases) * repetitions} benchmark runs"
    )
    for case in cases:
        print(
            f"  {case.case_id:<30} {_workload(case):>20}  "
            f"{case.payload_profile:<11} SU={case.num_storage_units:<2} "
            f"{case.producer_mode:<9} ~{_estimate_logical_gib(case):.2f} GiB"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument(
        "--suite",
        choices=("smoke", "core", "production", "all"),
        default="core",
    )
    parser.add_argument(
        "--suite-name",
        help="Unique suite directory; reuse the name to resume.",
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--case",
        action="append",
        dest="selected_cases",
        help="Run only this case ID; repeat to select multiple cases.",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="Print the selected suite matrix and exit.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--benchmark-script",
        default=str(Path(__file__).with_name("tq_checkpoint_benchmark.py")),
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--phase-timeout-s", type=float, default=1800)
    parser.add_argument(
        "--verify-mode",
        choices=("none", "sample", "all"),
        default="sample",
    )
    parser.add_argument("--verify-samples", type=int, default=64)
    parser.add_argument("--producer-max-rows", type=int, default=65536)
    parser.add_argument("--producer-warmup-s", type=float, default=0.25)
    parser.add_argument("--producer-cooldown-s", type=float, default=0.25)
    parser.add_argument("--producer-sleep-ms", type=float, default=0.0)
    parser.add_argument("--torch-num-threads", type=int, default=1)
    parser.add_argument("--ray-address", default="")
    parser.add_argument("--skip-space-check", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.repetitions <= 0:
        raise ValueError("--repetitions must be > 0")
    if args.phase_timeout_s <= 0:
        raise ValueError("--phase-timeout-s must be > 0")
    if args.verify_samples < 0:
        raise ValueError("--verify-samples must be >= 0")
    if args.producer_max_rows < 0:
        raise ValueError("--producer-max-rows must be >= 0")
    benchmark_script = Path(args.benchmark_script)
    if not benchmark_script.is_file():
        raise FileNotFoundError(f"benchmark script not found: {benchmark_script}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        _validate_args(args)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))

    cases = build_cases(args.suite)
    if args.selected_cases:
        requested = set(args.selected_cases)
        known = {case.case_id for case in cases}
        unknown = sorted(requested - known)
        if unknown:
            parser.error(f"unknown --case values: {unknown}; use --list-cases")
        cases = [case for case in cases if case.case_id in requested]
    _print_cases(cases, args.repetitions)
    if args.list_cases or args.dry_run:
        return

    checkpoint_root = Path(args.checkpoint_root).expanduser().resolve()
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    suite_name = args.suite_name or f"{args.suite}-{time.strftime('%Y%m%d-%H%M%S')}"
    if suite_name in {"", ".", ".."} or Path(suite_name).name != suite_name:
        parser.error("--suite-name must be one directory name, not a path")
    suite_root = checkpoint_root / suite_name
    suite_root.mkdir(parents=True, exist_ok=True)
    runs_root = suite_root / "runs"
    logs_root = suite_root / "logs"
    runs_root.mkdir(exist_ok=True)
    logs_root.mkdir(exist_ok=True)
    state_path = suite_root / "state.json"

    if state_path.exists():
        state = _load_json(state_path)
        _validate_existing_state(
            state,
            suite=args.suite,
            repetitions=args.repetitions,
            cases=cases,
        )
        if _reconcile_completed_runs(state):
            state["updated_at"] = _utc_now()
            _atomic_json(state_path, state)
        print(f"Resuming suite at {suite_root}")
    else:
        if any(suite_root.iterdir()):
            # Only directories created immediately above are allowed here.
            unexpected = [
                path.name
                for path in suite_root.iterdir()
                if path.name not in {"runs", "logs"}
            ]
            if unexpected:
                raise FileExistsError(
                    f"suite directory is non-empty without state.json: {unexpected}"
                )
        state = _new_state(
            suite=args.suite,
            suite_name=suite_name,
            repetitions=args.repetitions,
            cases=cases,
        )
        _atomic_json(state_path, state)

    refresh_reports(suite_root, state, cases)
    failures = 0
    interrupted = False
    try:
        for case in cases:
            for repetition in range(1, args.repetitions + 1):
                if _successful_run(state, case.case_id, repetition):
                    print(
                        f"[skip] {case.case_id} repetition {repetition} already passed"
                    )
                    continue
                run_name = _next_run_name(
                    state,
                    case.case_id,
                    repetition,
                )
                log_path = logs_root / f"{run_name}.log"
                result_path = runs_root / run_name / "result.json"
                command = _command(
                    python=args.python,
                    benchmark_script=Path(args.benchmark_script).resolve(),
                    runs_root=runs_root,
                    run_name=run_name,
                    case=case,
                    args=args,
                )
                record = {
                    "case_id": case.case_id,
                    "repetition": repetition,
                    "run_name": run_name,
                    "status": "running",
                    "started_at": _utc_now(),
                    "finished_at": None,
                    "command": command,
                    "result_path": str(result_path),
                    "log_path": str(log_path),
                    "returncode": None,
                    "error": None,
                }
                state["runs"].append(record)
                state["updated_at"] = _utc_now()
                _atomic_json(state_path, state)
                print(
                    f"\n[run] {case.case_id} repetition "
                    f"{repetition}/{args.repetitions}: {run_name}"
                )
                returncode = _run_command(command, log_path)
                record["returncode"] = returncode
                record["finished_at"] = _utc_now()
                if returncode == 0 and result_path.exists():
                    record["status"] = "success"
                    print(f"[pass] {run_name}")
                else:
                    record["status"] = "failed"
                    record["error"] = (
                        f"benchmark exited {returncode}"
                        if returncode
                        else "benchmark exited successfully but result.json is missing"
                    )
                    failures += 1
                    print(f"[fail] {run_name}: {record['error']}")
                state["updated_at"] = _utc_now()
                _atomic_json(state_path, state)
                refresh_reports(suite_root, state, cases)
                if failures and args.fail_fast:
                    raise RuntimeError("stopping after first failed benchmark")
    except KeyboardInterrupt:
        interrupted = True
        for record in reversed(state["runs"]):
            if record["status"] == "running":
                record["status"] = "interrupted"
                record["finished_at"] = _utc_now()
                record["error"] = "runner interrupted"
                break
        state["updated_at"] = _utc_now()
        _atomic_json(state_path, state)
        refresh_reports(suite_root, state, cases)
    finally:
        refresh_reports(suite_root, state, cases)

    print(f"\nReport: {suite_root / 'report.md'}")
    print(f"JSON:   {suite_root / 'report.json'}")
    print(f"CSV:    {suite_root / 'summary.csv'}")
    if interrupted:
        raise SystemExit(130)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
