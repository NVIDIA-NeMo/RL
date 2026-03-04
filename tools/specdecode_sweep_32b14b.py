#!/usr/bin/env python3
"""Run a reproducible spec-decoding sweep for 32B target + 14B draft.

This script has two modes:
1) sweep mode (default): orchestrates baseline + speculative runs via subprocesses,
   parses logs, and writes CSV/Markdown summaries.
2) single-run mode (--single-run): executes one generation run and prints summary
   metrics in a parse-friendly format.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence


SUMMARY_RE = re.compile(r"^([a-zA-Z0-9_]+)=(.+)$")
SPEC_LINE_RE = re.compile(
    r"Accepted: (\d+) tokens, Drafted: (\d+) tokens, .*Avg Draft acceptance rate: ([0-9.]+)%"
)


@dataclass
class RunResult:
    mode: str
    spec_tokens: int
    log_path: Path
    elapsed_s: float
    num_prompts: int
    prompt_tokens: int
    output_tokens: int
    total_tokens: int
    requests_per_s: float
    output_tokens_per_s: float
    total_tokens_per_s: float
    tar_pct: float | None
    accepted_tokens: int
    drafted_tokens: int
    spec_line_count: int
    speedup_vs_baseline_total: float | None = None
    speedup_vs_baseline_output: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spec decoding sweep for 32B target + 14B draft."
    )
    parser.add_argument(
        "--single-run",
        action="store_true",
        help="Run a single benchmark instance (used internally by sweep mode).",
    )

    parser.add_argument(
        "--target-model",
        type=str,
        default="/home/scratch.shaunakj_other/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137",
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        default="/home/scratch.shaunakj_other/.cache/huggingface/hub/models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="/home/scratch.shaunakj_other/openmath-cached-sample-256.jsonl",
    )
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--spec-tokens", type=int, default=0)
    parser.add_argument(
        "--sweep-spec-tokens",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Values to sweep in orchestrator mode.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="/home/scratch.shaunakj_other/logs/specdecode-speedup-2026-02-18",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default="/home/scratch.shaunakj_other/results/specdecode-speedup-2026-02-18",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default="",
        help="Optional suffix for generated artifact names.",
    )
    return parser.parse_args()


def load_prompts(path: str, num_prompts: int) -> list[str]:
    prompts: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if "input" not in row:
                raise KeyError(f"Missing 'input' key in dataset row: {row.keys()}")
            prompts.append(row["input"])
            if len(prompts) >= num_prompts:
                break
    if len(prompts) < num_prompts:
        raise ValueError(
            f"Dataset {path} has only {len(prompts)} prompts, requested {num_prompts}."
        )
    return prompts


def run_single(args: argparse.Namespace) -> int:
    from vllm import LLM, SamplingParams  # Imported only in single-run mode.

    prompts = load_prompts(args.dataset, args.num_prompts)
    mode = "baseline" if args.spec_tokens <= 0 else "spec"

    llm_kwargs = dict(
        model=args.target_model,
        tokenizer=args.target_model,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        attention_backend="FLASH_ATTN",
        async_scheduling=False,
        disable_log_stats=False,
        seed=args.seed,
    )
    if mode == "spec":
        llm_kwargs["speculative_config"] = {
            "method": "draft_model",
            "model": args.draft_model,
            "num_speculative_tokens": args.spec_tokens,
            "draft_tensor_parallel_size": args.tp,
        }

    print(f"run_mode={mode}")
    print(f"spec_tokens={args.spec_tokens}")
    print(f"num_prompts={args.num_prompts}")
    print(f"dataset={args.dataset}")
    print(f"target_model={args.target_model}")
    if mode == "spec":
        print(f"draft_model={args.draft_model}")

    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
        ignore_eos=True,
    )

    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    elapsed = time.perf_counter() - start

    tok = llm.get_tokenizer()
    prompt_tokens = sum(len(tok.encode(p)) for p in prompts)
    output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    total_tokens = prompt_tokens + output_tokens

    print("=== SWEEP SUMMARY ===")
    print(f"elapsed_s={elapsed:.4f}")
    print(f"num_prompts={len(prompts)}")
    print(f"prompt_tokens={prompt_tokens}")
    print(f"output_tokens={output_tokens}")
    print(f"total_tokens={total_tokens}")
    print(f"requests_per_s={len(prompts)/elapsed:.4f}")
    print(f"output_tokens_per_s={output_tokens/elapsed:.4f}")
    print(f"total_tokens_per_s={total_tokens/elapsed:.4f}")
    return 0


def parse_log(log_path: Path, mode: str, spec_tokens: int) -> RunResult:
    summary: dict[str, str] = {}
    accepted_tokens = 0
    drafted_tokens = 0
    spec_line_count = 0

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            m = SUMMARY_RE.match(line)
            if m:
                summary[m.group(1)] = m.group(2)
            s = SPEC_LINE_RE.search(line)
            if s:
                accepted_tokens += int(s.group(1))
                drafted_tokens += int(s.group(2))
                spec_line_count += 1

    required = [
        "elapsed_s",
        "num_prompts",
        "prompt_tokens",
        "output_tokens",
        "total_tokens",
        "requests_per_s",
        "output_tokens_per_s",
        "total_tokens_per_s",
    ]
    missing = [k for k in required if k not in summary]
    if missing:
        raise RuntimeError(
            f"Missing summary keys in {log_path}: {missing}. "
            "Check the run for failures."
        )

    tar_pct = None
    if drafted_tokens > 0:
        tar_pct = 100.0 * accepted_tokens / drafted_tokens

    return RunResult(
        mode=mode,
        spec_tokens=spec_tokens,
        log_path=log_path,
        elapsed_s=float(summary["elapsed_s"]),
        num_prompts=int(summary["num_prompts"]),
        prompt_tokens=int(summary["prompt_tokens"]),
        output_tokens=int(summary["output_tokens"]),
        total_tokens=int(summary["total_tokens"]),
        requests_per_s=float(summary["requests_per_s"]),
        output_tokens_per_s=float(summary["output_tokens_per_s"]),
        total_tokens_per_s=float(summary["total_tokens_per_s"]),
        tar_pct=tar_pct,
        accepted_tokens=accepted_tokens,
        drafted_tokens=drafted_tokens,
        spec_line_count=spec_line_count,
    )


def run_subprocess(cmd: Sequence[str], env: dict[str, str], log_path: Path) -> None:
    with open(log_path, "w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=env,
        )
        code = proc.wait()
    if code != 0:
        raise RuntimeError(f"Run failed with exit code {code}. Log: {log_path}")


def write_csv(results: list[RunResult], out_path: Path) -> None:
    fields = [
        "mode",
        "spec_tokens",
        "elapsed_s",
        "num_prompts",
        "prompt_tokens",
        "output_tokens",
        "total_tokens",
        "requests_per_s",
        "output_tokens_per_s",
        "total_tokens_per_s",
        "tar_pct",
        "accepted_tokens",
        "drafted_tokens",
        "spec_line_count",
        "speedup_vs_baseline_output",
        "speedup_vs_baseline_total",
        "log_path",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "mode": r.mode,
                    "spec_tokens": r.spec_tokens,
                    "elapsed_s": f"{r.elapsed_s:.4f}",
                    "num_prompts": r.num_prompts,
                    "prompt_tokens": r.prompt_tokens,
                    "output_tokens": r.output_tokens,
                    "total_tokens": r.total_tokens,
                    "requests_per_s": f"{r.requests_per_s:.4f}",
                    "output_tokens_per_s": f"{r.output_tokens_per_s:.4f}",
                    "total_tokens_per_s": f"{r.total_tokens_per_s:.4f}",
                    "tar_pct": "" if r.tar_pct is None else f"{r.tar_pct:.2f}",
                    "accepted_tokens": r.accepted_tokens,
                    "drafted_tokens": r.drafted_tokens,
                    "spec_line_count": r.spec_line_count,
                    "speedup_vs_baseline_output": (
                        ""
                        if r.speedup_vs_baseline_output is None
                        else f"{r.speedup_vs_baseline_output:.4f}"
                    ),
                    "speedup_vs_baseline_total": (
                        ""
                        if r.speedup_vs_baseline_total is None
                        else f"{r.speedup_vs_baseline_total:.4f}"
                    ),
                    "log_path": str(r.log_path),
                }
            )


def write_markdown(results: list[RunResult], out_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Spec Decode Sweep Results")
    lines.append("")
    lines.append("| mode | spec_tokens | output tok/s | total tok/s | TAR % | speedup (output) | speedup (total) | log |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for r in results:
        tar_str = "-" if r.tar_pct is None else f"{r.tar_pct:.2f}"
        s_out = (
            "-" if r.speedup_vs_baseline_output is None else f"{r.speedup_vs_baseline_output:.4f}x"
        )
        s_tot = (
            "-" if r.speedup_vs_baseline_total is None else f"{r.speedup_vs_baseline_total:.4f}x"
        )
        lines.append(
            "| "
            f"{r.mode} | {r.spec_tokens} | {r.output_tokens_per_s:.4f} | {r.total_tokens_per_s:.4f} | "
            f"{tar_str} | {s_out} | {s_tot} | {r.log_path} |"
        )
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def orchestrate_sweep(args: argparse.Namespace) -> int:
    # Keep the venv shim path intact; resolving symlinks can drop us to the
    # system interpreter and lose installed deps like vllm.
    py = Path(sys.executable)
    script = Path(__file__).resolve()
    log_dir = Path(args.log_dir).resolve()
    result_dir = Path(args.result_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = f"-{args.run_tag}" if args.run_tag else ""
    prefix = f"sweep-32b14b-openmath64-greedy-{timestamp}{tag}"

    env = os.environ.copy()
    env["VLLM_LOG_STATS_INTERVAL"] = "1"
    env.setdefault("PYTHONUNBUFFERED", "1")

    all_results: list[RunResult] = []

    base_log = log_dir / f"{prefix}-baseline.log"
    base_cmd = [
        str(py),
        str(script),
        "--single-run",
        "--target-model",
        args.target_model,
        "--draft-model",
        args.draft_model,
        "--dataset",
        args.dataset,
        "--num-prompts",
        str(args.num_prompts),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--max-model-len",
        str(args.max_model_len),
        "--tp",
        str(args.tp),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--seed",
        str(args.seed),
        "--spec-tokens",
        "0",
    ]
    print(f"[sweep] baseline -> {base_log}")
    run_subprocess(base_cmd, env, base_log)
    baseline = parse_log(base_log, mode="baseline", spec_tokens=0)
    all_results.append(baseline)

    for spec_tokens in args.sweep_spec_tokens:
        spec_log = log_dir / f"{prefix}-spec{spec_tokens}.log"
        spec_cmd = [
            str(py),
            str(script),
            "--single-run",
            "--target-model",
            args.target_model,
            "--draft-model",
            args.draft_model,
            "--dataset",
            args.dataset,
            "--num-prompts",
            str(args.num_prompts),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--max-model-len",
            str(args.max_model_len),
            "--tp",
            str(args.tp),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--seed",
            str(args.seed),
            "--spec-tokens",
            str(spec_tokens),
        ]
        print(f"[sweep] spec_tokens={spec_tokens} -> {spec_log}")
        run_subprocess(spec_cmd, env, spec_log)
        spec_res = parse_log(spec_log, mode="spec", spec_tokens=spec_tokens)
        all_results.append(spec_res)

    # Compute speedups relative to baseline.
    for r in all_results:
        r.speedup_vs_baseline_output = r.output_tokens_per_s / baseline.output_tokens_per_s
        r.speedup_vs_baseline_total = r.total_tokens_per_s / baseline.total_tokens_per_s

    csv_path = result_dir / f"{prefix}.csv"
    md_path = result_dir / f"{prefix}.md"
    write_csv(all_results, csv_path)
    write_markdown(all_results, md_path)

    print("\n[sweep] summary")
    for r in all_results:
        tar_s = "-" if r.tar_pct is None else f"{r.tar_pct:.2f}%"
        print(
            f"mode={r.mode:8s} spec={r.spec_tokens:<2d} "
            f"out_tps={r.output_tokens_per_s:.4f} total_tps={r.total_tokens_per_s:.4f} "
            f"TAR={tar_s} speedup_out={r.speedup_vs_baseline_output:.4f}x"
        )

    best = max(all_results[1:], key=lambda x: x.output_tokens_per_s)
    print(
        "\n[sweep] best speculative point by output tok/s: "
        f"spec_tokens={best.spec_tokens}, output_tps={best.output_tokens_per_s:.4f}, "
        f"speedup_vs_baseline={best.speedup_vs_baseline_output:.4f}x, "
        f"TAR={'-' if best.tar_pct is None else f'{best.tar_pct:.2f}%'}"
    )
    print(f"[sweep] csv={csv_path}")
    print(f"[sweep] markdown={md_path}")
    return 0


def main() -> int:
    args = parse_args()
    if args.single_run:
        return run_single(args)
    return orchestrate_sweep(args)


if __name__ == "__main__":
    raise SystemExit(main())
