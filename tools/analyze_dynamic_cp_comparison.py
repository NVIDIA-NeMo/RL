#!/usr/bin/env python3
"""Build CSV/Markdown speedup sheets for matched Dynamic-CP/static-CP runs.

The launcher in ``tools/launch_dynamic_cp_comparison.sh`` writes runs as:

    <root>/<comparison-name>/<model>/<dynamic|static>/

This tool accepts the ``<comparison-name>`` directory. It reads TensorBoard
events when TensorBoard is installed and falls back to the human-readable
timing blocks in Slurm/Ray logs.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


MODEL_METADATA = {
    "qwen30": {
        "gbs": 512,
        "max_sequence_length": 8192,
        "static_cp": 2,
        "dynamic_cp": "1-2",
    },
    "qwen32": {
        "gbs": 512,
        "max_sequence_length": 16384,
        "static_cp": 4,
        "dynamic_cp": "1-4",
    },
    "nano": {
        "gbs": 64,
        "max_sequence_length": 8192,
        "static_cp": 4,
        "dynamic_cp": "4-16 effective (1-16 configured)",
    },
}

MODEL_ALIASES = {
    "qwen30": "qwen30",
    "qwen3-30b": "qwen30",
    "qwen3-30ba3b": "qwen30",
    "qwen32": "qwen32",
    "qwen3-32b": "qwen32",
    "nano": "nano",
    "nt3-nano": "nano",
    "nemotron3-nano": "nano",
}

PREFERRED_METRICS = [
    "timing/train/total_step_time",
    "timing/train/policy_training",
    "timing/train/policy_and_reference_logprobs",
    "timing/train/generation",
    "timing/train/prepare_for_generation/total",
    "timing/train/prepare_for_generation/transfer_and_update_weights",
    "timing/train/training_prep",
    "timing/train/logprob_inference_prep",
    "timing/train/reward_calculation",
    "timing/train/data_processing",
    "timing/train/valid_tokens_per_sec_per_gpu",
]

ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
STEP_RE = re.compile(r"\bStep\s+(\d+)\s*/\s*\d+", re.IGNORECASE)
TRAIN_STEP_RE = re.compile(r"\btrain step\s+(\d+)\s*/\s*\d+", re.IGNORECASE)
FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
TOTAL_TIME_RE = re.compile(rf"Total step time\s*:\s*({FLOAT})s\b", re.IGNORECASE)
TIMING_VALUE_RE = re.compile(rf"[•*-]\s*([A-Za-z0-9_./-]+)\s*:\s*({FLOAT})s(?:\s|$)")
FLAT_METRIC_RE = re.compile(
    rf"(timing/train/[A-Za-z0-9_./-]+)\s*[=:]\s*({FLOAT})"
)
DYNAMIC_CP_RE = re.compile(
    rf"Dynamic CP .*?samples_by_cp=(\{{.*?\}}).*?tasks_by_cp=(\{{.*?\}})"
    rf".*?packing_utilization=({FLOAT})"
)


@dataclass(frozen=True)
class RunSpec:
    model: str
    mode: str
    path: Path


@dataclass
class RunData:
    spec: RunSpec
    # metric -> step -> value. Later event/log files replace an earlier value.
    metrics: dict[str, dict[int, float]]
    packing_utilizations: list[float]
    samples_by_cp: dict[int, int]
    tasks_by_cp: dict[int, int]
    source: str


def canonical_model(value: str) -> str:
    key = value.strip().lower()
    if key not in MODEL_ALIASES:
        raise ValueError(f"unknown model '{value}'")
    return MODEL_ALIASES[key]


def canonical_mode(value: str) -> str:
    key = value.strip().lower()
    if key in {"dynamic", "dyncp", "dyn"}:
        return "dynamic"
    if key in {"static", "staticcp", "no-dyncp", "nodyncp"}:
        return "static"
    raise ValueError(f"unknown mode '{value}'")


def parse_run_arg(value: str) -> RunSpec:
    try:
        model, mode, path = value.split(":", 2)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--run must be MODEL:MODE:PATH (for example qwen30:dynamic:/logs/run)"
        ) from exc
    try:
        return RunSpec(canonical_model(model), canonical_mode(mode), Path(path))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def discover_runs(root: Path) -> list[RunSpec]:
    runs: list[RunSpec] = []
    for model in MODEL_METADATA:
        for mode in ("dynamic", "static"):
            path = root / model / mode
            if path.is_dir():
                runs.append(RunSpec(model, mode, path))
    return runs


def _record(
    metrics: dict[str, dict[int, float]], metric: str, step: int, value: float
) -> None:
    if math.isfinite(value):
        metrics.setdefault(metric, {})[step] = value


def read_tensorboard(path: Path) -> tuple[dict[str, dict[int, float]], bool]:
    event_files = sorted(
        path.rglob("events*tfevents*"), key=lambda item: item.stat().st_mtime
    )
    if not event_files:
        return {}, False

    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        print(
            "WARNING: TensorBoard event files exist, but the 'tensorboard' package "
            "is unavailable; falling back to text logs.",
            file=sys.stderr,
        )
        return {}, False

    size_guidance = {
        event_accumulator.SCALARS: 0,
        event_accumulator.TENSORS: 0,
    }
    metrics: dict[str, dict[int, float]] = {}
    for event_file in event_files:
        accumulator = event_accumulator.EventAccumulator(
            str(event_file), size_guidance=size_guidance
        )
        accumulator.Reload()
        for metric in accumulator.scalars.Keys():
            if not metric.startswith("timing/train/"):
                continue
            for scalar in accumulator.Scalars(metric):
                _record(metrics, metric, int(scalar.step), float(scalar.value))
    return metrics, bool(metrics)


def _parse_cp_dict(value: str) -> dict[int, int]:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, dict):
        return {}
    return {int(key): int(count) for key, count in parsed.items()}


def _merge_counts(target: dict[int, int], values: dict[int, int]) -> None:
    for key, value in values.items():
        target[key] = target.get(key, 0) + value


def _candidate_text_files(path: Path) -> list[Path]:
    candidates: set[Path] = set()
    for pattern in ("*.out", "*.log", "*.txt"):
        candidates.update(path.rglob(pattern))
    return sorted(candidates, key=lambda item: item.stat().st_mtime)


def _walk_json(value: object, prefix: str = "") -> Iterable[tuple[str, object]]:
    if not isinstance(value, dict):
        return
    for key, child in value.items():
        full_key = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(child, dict):
            yield from _walk_json(child, full_key)
        else:
            yield full_key, child


def read_text_logs(
    path: Path, *, read_timing: bool
) -> tuple[dict[str, dict[int, float]], list[float], dict[int, int], dict[int, int]]:
    metrics: dict[str, dict[int, float]] = {}
    packing_utilizations: list[float] = []
    samples_by_cp: dict[int, int] = {}
    tasks_by_cp: dict[int, int] = {}

    for log_path in _candidate_text_files(path):
        current_step: int | None = None
        inferred_step = 0
        in_timing = False
        try:
            handle = log_path.open("r", encoding="utf-8", errors="replace")
        except OSError as exc:
            print(f"WARNING: cannot read {log_path}: {exc}", file=sys.stderr)
            continue

        with handle:
            for raw_line in handle:
                line = ANSI_RE.sub("", raw_line).strip()
                step_match = STEP_RE.search(line) or TRAIN_STEP_RE.search(line)
                if step_match:
                    current_step = int(step_match.group(1))
                    in_timing = False

                cp_match = DYNAMIC_CP_RE.search(line)
                if cp_match:
                    try:
                        _merge_counts(samples_by_cp, _parse_cp_dict(cp_match.group(1)))
                        _merge_counts(tasks_by_cp, _parse_cp_dict(cp_match.group(2)))
                        packing_utilizations.append(float(cp_match.group(3)))
                    except (SyntaxError, TypeError, ValueError):
                        pass

                if not read_timing:
                    continue

                if "Timing:" in line:
                    in_timing = True
                    continue
                if "Performance Metrics:" in line or "Training Results:" in line:
                    in_timing = False

                total_match = TOTAL_TIME_RE.search(line)
                if total_match:
                    if current_step is None:
                        inferred_step += 1
                        current_step = inferred_step
                    _record(
                        metrics,
                        "timing/train/total_step_time",
                        current_step,
                        float(total_match.group(1)),
                    )
                    continue

                if in_timing and current_step is not None:
                    timing_match = TIMING_VALUE_RE.search(line)
                    if timing_match:
                        _record(
                            metrics,
                            f"timing/train/{timing_match.group(1)}",
                            current_step,
                            float(timing_match.group(2)),
                        )

                for flat_match in FLAT_METRIC_RE.finditer(line):
                    if current_step is not None:
                        _record(
                            metrics,
                            flat_match.group(1),
                            current_step,
                            float(flat_match.group(2)),
                        )

                if line.startswith("{") and len(line) < 1_000_000:
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(payload, dict):
                        continue
                    json_step = payload.get("step", payload.get("_step", current_step))
                    if not isinstance(json_step, (int, float)):
                        continue
                    for key, value in _walk_json(payload):
                        if key.startswith("timing/train/") and isinstance(
                            value, (int, float)
                        ):
                            _record(metrics, key, int(json_step), float(value))

    return metrics, packing_utilizations, samples_by_cp, tasks_by_cp


def load_run(spec: RunSpec) -> RunData:
    if not spec.path.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {spec.path}")

    metrics, used_tensorboard = read_tensorboard(spec.path)
    text_metrics, utilization, samples_by_cp, tasks_by_cp = read_text_logs(
        spec.path, read_timing=not used_tensorboard
    )
    if not used_tensorboard:
        metrics = text_metrics
    return RunData(
        spec=spec,
        metrics=metrics,
        packing_utilizations=utilization,
        samples_by_cp=samples_by_cp,
        tasks_by_cp=tasks_by_cp,
        source="tensorboard" if used_tensorboard else "text",
    )


def kept_values(steps: dict[int, float], warmup_steps: int) -> list[float]:
    ordered = [value for _, value in sorted(steps.items())]
    return ordered[warmup_steps:]


def metric_order(metric: str) -> tuple[int, str]:
    try:
        return PREFERRED_METRICS.index(metric), metric
    except ValueError:
        return len(PREFERRED_METRICS), metric


def higher_is_better(metric: str) -> bool:
    name = metric.lower()
    return "per_sec" in name or "throughput" in name or name.endswith("_tps")


def fmt(value: float | None, digits: int = 3) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def build_rows(
    runs: dict[tuple[str, str], RunData], warmup_steps: int
) -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    warnings: list[str] = []
    for model, metadata in MODEL_METADATA.items():
        dynamic = runs.get((model, "dynamic"))
        static = runs.get((model, "static"))
        if dynamic is None or static is None:
            missing = "dynamic" if dynamic is None else "static"
            warnings.append(f"{model}: missing {missing} run")
            continue

        common_metrics = set(dynamic.metrics) & set(static.metrics)
        if not common_metrics:
            warnings.append(f"{model}: dynamic/static runs have no shared timing metrics")
            continue

        for metric in sorted(common_metrics, key=metric_order):
            dynamic_values = kept_values(dynamic.metrics[metric], warmup_steps)
            static_values = kept_values(static.metrics[metric], warmup_steps)
            if not dynamic_values or not static_values:
                warnings.append(
                    f"{model}/{metric}: no samples remain after dropping "
                    f"{warmup_steps} warmup step(s)"
                )
                continue
            dynamic_mean = statistics.fmean(dynamic_values)
            static_mean = statistics.fmean(static_values)
            is_higher_better = higher_is_better(metric)
            if dynamic_mean == 0 or static_mean == 0:
                speedup = math.nan
                improvement = math.nan
            elif is_higher_better:
                speedup = dynamic_mean / static_mean
                improvement = (dynamic_mean - static_mean) / static_mean * 100
            else:
                speedup = static_mean / dynamic_mean
                improvement = (static_mean - dynamic_mean) / static_mean * 100
            rows.append(
                {
                    "model": model,
                    **metadata,
                    "metric": metric,
                    "direction": "higher is better" if is_higher_better else "lower is better",
                    "dynamic_samples": len(dynamic_values),
                    "static_samples": len(static_values),
                    "dynamic_mean": dynamic_mean,
                    "dynamic_median": statistics.median(dynamic_values),
                    "static_mean": static_mean,
                    "static_median": statistics.median(static_values),
                    "speedup_x": speedup,
                    "improvement_percent": improvement,
                }
            )
    return rows, warnings


def write_speedup_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "model",
        "gbs",
        "max_sequence_length",
        "static_cp",
        "dynamic_cp",
        "metric",
        "direction",
        "dynamic_samples",
        "static_samples",
        "dynamic_mean",
        "dynamic_median",
        "static_mean",
        "static_median",
        "speedup_x",
        "improvement_percent",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            for key in (
                "dynamic_mean",
                "dynamic_median",
                "static_mean",
                "static_median",
                "speedup_x",
                "improvement_percent",
            ):
                output[key] = fmt(float(output[key]), 6)
            writer.writerow(output)


def write_runs_csv(path: Path, runs: dict[tuple[str, str], RunData]) -> None:
    fields = [
        "model",
        "mode",
        "path",
        "metric_source",
        "timing_metrics",
        "timing_steps",
        "mean_packing_utilization",
        "samples_by_cp",
        "tasks_by_cp",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in sorted(runs):
            run = runs[key]
            all_steps = {step for values in run.metrics.values() for step in values}
            writer.writerow(
                {
                    "model": run.spec.model,
                    "mode": run.spec.mode,
                    "path": run.spec.path,
                    "metric_source": run.source,
                    "timing_metrics": len(run.metrics),
                    "timing_steps": len(all_steps),
                    "mean_packing_utilization": fmt(
                        statistics.fmean(run.packing_utilizations)
                        if run.packing_utilizations
                        else None,
                        6,
                    ),
                    "samples_by_cp": json.dumps(run.samples_by_cp, sort_keys=True),
                    "tasks_by_cp": json.dumps(run.tasks_by_cp, sort_keys=True),
                }
            )


def write_markdown(
    path: Path,
    rows: list[dict[str, object]],
    runs: dict[tuple[str, str], RunData],
    warnings: list[str],
    warmup_steps: int,
) -> None:
    lines = [
        "# Dynamic CP vs static CP speedup",
        "",
        f"Warmup steps excluded per metric: **{warmup_steps}**.",
        "Speedup for time metrics is `static mean / dynamic mean`; values above 1.0 favor Dynamic CP.",
        "The static arms use CP2/CP4/CP4—not CP1/no-CP.",
        "",
        "| Model | GBS | Max seq | Static CP | Dynamic CP | Metric | Dynamic | Static | Speedup | Improvement |",
        "|---|---:|---:|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        metric = str(row["metric"]).removeprefix("timing/train/")
        unit = "" if higher_is_better(str(row["metric"])) else " s"
        lines.append(
            f"| {row['model']} | {row['gbs']} | {row['max_sequence_length']} "
            f"| {row['static_cp']} | {row['dynamic_cp']} | `{metric}` "
            f"| {fmt(float(row['dynamic_mean']))}{unit} "
            f"| {fmt(float(row['static_mean']))}{unit} "
            f"| {fmt(float(row['speedup_x']))}x "
            f"| {fmt(float(row['improvement_percent']), 2)}% |"
        )

    lines.extend(["", "## Run diagnostics", ""])
    for key in sorted(runs):
        run = runs[key]
        all_steps = sorted({step for values in run.metrics.values() for step in values})
        utilization = (
            fmt(statistics.fmean(run.packing_utilizations), 4)
            if run.packing_utilizations
            else "n/a"
        )
        lines.append(
            f"- `{run.spec.model}/{run.spec.mode}`: source={run.source}, "
            f"steps={all_steps or 'none'}, mean Dynamic-CP packing utilization={utilization}, "
            f"tasks_by_cp={run.tasks_by_cp or 'n/a'}"
        )

    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        help="Comparison directory containing qwen30/, qwen32/, and nano/",
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        type=parse_run_arg,
        metavar="MODEL:MODE:PATH",
        help="Explicit run directory; repeat for each arm instead of --root",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: ROOT/analysis or ./dynamic-cp-analysis)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1,
        help="Discard this many earliest samples from each metric (default: 1)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero when a model/mode pair or usable metric is missing",
    )
    args = parser.parse_args()
    if args.root is None and not args.run:
        parser.error("provide --root or at least one --run")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps must be zero or greater")
    return args


def main() -> int:
    args = parse_args()
    specs = list(args.run)
    if args.root is not None:
        if not args.root.is_dir():
            print(f"ERROR: root directory does not exist: {args.root}", file=sys.stderr)
            return 2
        specs.extend(discover_runs(args.root))
    if not specs:
        print(
            "ERROR: no runs found; expected ROOT/{qwen30,qwen32,nano}/{dynamic,static}",
            file=sys.stderr,
        )
        return 2

    runs: dict[tuple[str, str], RunData] = {}
    for spec in specs:
        key = (spec.model, spec.mode)
        if key in runs:
            print(f"ERROR: duplicate run for {spec.model}/{spec.mode}", file=sys.stderr)
            return 2
        try:
            runs[key] = load_run(spec)
        except (FileNotFoundError, OSError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

    rows, warnings = build_rows(runs, args.warmup_steps)
    if not rows:
        warnings.append("no comparison rows were generated")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            args.root / "analysis" if args.root else Path("dynamic-cp-analysis")
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    speedup_csv = output_dir / "dynamic_cp_speedup.csv"
    runs_csv = output_dir / "dynamic_cp_runs.csv"
    markdown = output_dir / "dynamic_cp_speedup.md"
    write_speedup_csv(speedup_csv, rows)
    write_runs_csv(runs_csv, runs)
    write_markdown(markdown, rows, runs, warnings, args.warmup_steps)

    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    print(f"Wrote {speedup_csv}")
    print(f"Wrote {runs_csv}")
    print(f"Wrote {markdown}")
    if args.strict and warnings:
        return 1
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
