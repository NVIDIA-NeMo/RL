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
"""Diagnose rollout performance of a NeMo-RL run, or compare two arms.

Reads the ``metrics.json`` written by ``tests/json_dump_tb_logs.py`` (metric ->
{step: value}) and, optionally, the Ray driver log, from which it extracts the
effective settings that vLLM, SGLang, TRT-LLM, Megatron inference and Dynamo
log at startup. Standard library only.

Diagnose one run:

    uv run --no-project python rollout_perf_report.py RUN/metrics.json \
        --log RUN/<jobid>-logs/ray-driver.log

Compare a control and a treatment of the same test, with one or more runs per
arm (steps are paired; with several runs, each step uses the median across
runs):

    uv run --no-project python rollout_perf_report.py \
        --base BASE1/metrics.json BASE2/metrics.json \
        --treat TREAT1/metrics.json TREAT2/metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections.abc import Callable, Iterable
from pathlib import Path

GEN = "timing/train/generation"
STEP = "timing/train/total_step_time"
VAL = "timing/validation/total_validation_time"
SETUP = "timing/setup/total_setup_time_s"
MAX_GEN = "train/max_gen_tokens_per_sample"
MEAN_GEN = "train/mean_gen_tokens_per_sample"
PROMPT = "train/mean_prompt_length"
TURNS = "train/avg_turns_per_sample"
TRUNC = "train/truncation_rate"
TMPE = "train/token_mult_prob_error"
GEN_KL = "train/gen_kl_error"
VAL_ACC = "validation/accuracy"

# Top-level children of timing/train/total_step_time in synchronous GRPO; they
# do not overlap one another. Async GRPO logs exposed_generation and
# weight_sync instead. Nested timers (e.g. prepare_for_generation/*,
# policy_training/*) are deliberately not summed.
PHASES = {
    "generation": GEN,
    "exposed generation (async)": "timing/train/exposed_generation",
    "refit": "timing/train/prepare_for_generation/total",
    "weight sync (async)": "timing/train/weight_sync",
    "reward": "timing/train/reward_calculation",
    "logprobs": "timing/train/policy_and_reference_logprobs",
    "training": "timing/train/policy_training",
    "checkpointing": "timing/train/checkpointing",
}

# Effective settings from the engines' own startup lines. All patterns use the
# `key=value` form of the engines' config reprs; the driver's config echoes use
# `'key': value` (or are `Overrides:` / `MasterConfig(` lines, skipped below),
# so they show what was requested, not what ran. Engine log formats change
# across versions: a missing value means "not found", not "default".
ENGINE_PATTERNS: dict[str, dict[str, str]] = {
    "vllm": {
        "version": r"Initializing a V1 LLM engine \(v([\w.+-]+)\)",
        "enforce_eager": r"\benforce_eager=(True|False)",
        "cudagraph_mode": r"'cudagraph_mode': <CUDAGraphMode\.(\w+)",
        "enable_prefix_caching": r"\benable_prefix_caching=(True|False)",
        "max_num_batched_tokens": r"Chunked prefill is enabled with max_num_batched_tokens=(\d+)",
        "kv_cache_tokens": r"GPU KV cache size: ([\d,]+) tokens",
        "max_concurrency": r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x",
        "moe_backend": r"Using (.+?) MoE backend",
        "attention_backend": r"Using (\S+) attention backend",
    },
    "sglang": {
        "disable_cuda_graph": r"\bdisable_cuda_graph=(True|False)",
        "disable_radix_cache": r"\bdisable_radix_cache=(True|False)",
        "chunked_prefill_size": r"\bchunked_prefill_size=(-?\d+)",
        "max_running_requests": r"\bmax_running_requests=(\w+)",
        "max_total_num_tokens": r"\bmax_total_num_tokens=(\d+)",
        "mem_fraction_static": r"\bmem_fraction_static=([\d.]+)",
    },
    "trtllm": {
        "max_num_tokens": r"\bmax_num_tokens=(\d+)",
        "max_batch_size": r"\bmax_batch_size=(\d+)",
        "enable_block_reuse": r"\benable_block_reuse=(True|False)",
        "enable_chunked_prefill": r"\benable_chunked_prefill=(True|False)",
        "allreduce_strategy": r"\ballreduce_strategy=['\"]?([\w.]+)",
        "free_gpu_memory_fraction": r"\bfree_gpu_memory_fraction=([\d.]+)",
    },
    "megatron": {
        "engine": r"(Initialized persistent inference engine)",
        "async_sched_steps": r"mcore async scheduling steps \(cumul\): (\d+)",
        "graph_count": r"\[graph \d+/(\d+)\]",
        "largest_graph_tokens": r"\[graph 1/\d+\] \[(\d+)\]",
        "graph_build_seconds": r"> built cuda graph\(s\) in ([\d.]+) sec",
    },
    "dynamo": {
        "worker_argv": r"\[Dynamo:[^\]]*\] launching argv=(.{0,200})",
        "frontend_argv": r"\[Dynamo\] launching frontend argv=(.{0,200})",
    },
}
# A backend's patterns are applied only if one of its anchor lines appears, so
# that a shared field name (e.g. enable_chunked_prefill=) is not attributed to
# the wrong engine. Dynamo runs also show vLLM worker lines.
ENGINE_ANCHORS = {
    "vllm": ("Initializing a V1 LLM engine", "non-default args:"),
    "sglang": ("ServerArgs(", "Launch HttpServerEngineAdapter"),
    "trtllm": ("TrtllmAsyncWorker", "LLM Args"),
    "megatron": ("Initialized persistent inference engine", "mcore async scheduling"),
    "dynamo": ("[Dynamo",),
}
# Running totals: keep the last value instead of the first.
LAST_VALUE = {"async_sched_steps"}
# Last-seen values from vLLM's periodic engine stats line.
VLLM_STATS = re.compile(
    r"Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([\d.]+)%"
    r"(?:, Prefix cache hit rate: ([\d.]+)%)?"
)
# Requested values from the driver's config echo (not proof of what ran).
CONFIG_PATTERNS = {
    "max_new_tokens": r"'max_new_tokens': (\d+)",
    "backend": r"'generation': \{'backend': '(\w+)'",
}

# Engine-neutral view of the effective settings. Each lever maps to the
# per-backend fields above; a value is "on"/"off" or a number, and the source
# says which engine line proved it. Fields that a backend does not log stay
# unproven (see references/engines.md).
GRAPH_KNOB = {
    "vllm": "vllm_cfg.enforce_eager / vllm_kwargs.compilation_config.cudagraph_mode",
    "dynamo": "vllm_cfg.enforce_eager / vllm_kwargs.compilation_config",
    "sglang": "sglang_cfg.disable_cuda_graph",
    "trtllm": "trtllm_kwargs.cuda_graph_config",
    "megatron": "mcore_generation_config.cuda_graph_impl",
}
REUSE_KNOB = {
    "vllm": "vllm_cfg.enable_prefix_caching",
    "dynamo": "vllm_kwargs.enable_prefix_caching",
    "sglang": "radix cache (sglang_cfg.disable_radix_cache is not forwarded today)",
    "trtllm": "trtllm_kwargs.kv_cache_config.enable_block_reuse",
    "megatron": "mcore_generation_config.enable_prefix_caching",
}


def normalize(log: dict[str, dict[str, str]]) -> dict[str, tuple[str, str]]:
    """Map per-engine log fields onto engine-neutral levers."""
    vllm, sglang, trtllm = (log.get(e, {}) for e in ("vllm", "sglang", "trtllm"))
    out: dict[str, tuple[str, str]] = {}

    def put(field: str, value: str | None, source: str) -> None:
        if value is not None and field not in out:
            out[field] = (value, source)

    if vllm.get("enforce_eager") == "True" or vllm.get("cudagraph_mode") == "NONE":
        put("cuda_graphs", "off", "vllm")
    elif "cudagraph_mode" in vllm:
        put("cuda_graphs", f"on ({vllm['cudagraph_mode']})", "vllm")
    megatron = log.get("megatron", {})
    if "graph_count" in megatron:
        put(
            "cuda_graphs",
            f"on ({megatron['graph_count']} graphs, up to "
            f"{megatron.get('largest_graph_tokens', '?')} tokens)",
            "megatron",
        )
    # No `[graph` lines is not proof of eager mode: Ray log deduplication or
    # the MCore log level can hide them, so leave the field unproven.
    if "disable_cuda_graph" in sglang:
        put(
            "cuda_graphs",
            "off" if sglang["disable_cuda_graph"] == "True" else "on",
            "sglang",
        )
    on_off = {"True": "on", "False": "off"}
    put("prefix_reuse", on_off.get(vllm.get("enable_prefix_caching", "")), "vllm")
    if "disable_radix_cache" in sglang:
        put(
            "prefix_reuse",
            "off" if sglang["disable_radix_cache"] == "True" else "on",
            "sglang",
        )
    put("prefix_reuse", on_off.get(trtllm.get("enable_block_reuse", "")), "trtllm")
    put("token_budget", vllm.get("max_num_batched_tokens"), "vllm")
    put("token_budget", sglang.get("chunked_prefill_size"), "sglang")
    put("token_budget", trtllm.get("max_num_tokens"), "trtllm")
    if "max_num_batched_tokens" in vllm:
        put("chunked_prefill", "on", "vllm")
    put(
        "chunked_prefill",
        on_off.get(trtllm.get("enable_chunked_prefill", "")),
        "trtllm",
    )
    if "chunked_prefill_size" in sglang:
        put(
            "chunked_prefill",
            "off" if sglang["chunked_prefill_size"] == "-1" else "on",
            "sglang",
        )
    cap = sglang.get("max_running_requests")
    put("admission_cap", "auto" if cap == "None" else cap, "sglang")
    put("admission_cap", trtllm.get("max_batch_size"), "trtllm")
    put("memory_fraction", sglang.get("mem_fraction_static"), "sglang")
    put("memory_fraction", trtllm.get("free_gpu_memory_fraction"), "trtllm")
    put("kv_capacity", vllm.get("max_concurrency"), "vllm")
    if "max_total_num_tokens" in sglang:
        put("kv_capacity", f"{sglang['max_total_num_tokens']} tokens", "sglang")
    return out


def series(metrics: dict, key: str) -> dict[int, float]:
    return {int(step): float(value) for step, value in metrics.get(key, {}).items()}


def med(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.median(values) if values else math.nan


def rel_change(before: float, after: float) -> str:
    return "n/a" if before == 0 else f"{(after / before - 1) * 100:+.1f}%"


def scan_log(path: Path) -> dict[str, dict[str, str]]:
    found: dict[str, dict[str, str]] = {}
    config: dict[str, str] = {}
    captured = False
    stats: list[tuple[int, int, float, float | None]] = []
    anchored: set[str] = set()
    with path.open(errors="replace") as handle:
        for line in handle:
            anchored.update(
                engine
                for engine, anchors in ENGINE_ANCHORS.items()
                if any(anchor in line for anchor in anchors)
            )
            for name, pattern in CONFIG_PATTERNS.items():
                if name not in config and (match := re.search(pattern, line)):
                    config[name] = match.group(1)
            if "Overrides:" in line or "MasterConfig(" in line:
                continue
            if "Capturing CUDA graphs" in line:
                captured = True
            if match := VLLM_STATS.search(line):
                hit = match.group(4)
                stats.append(
                    (
                        int(match.group(1)),
                        int(match.group(2)),
                        float(match.group(3)),
                        float(hit) if hit else None,
                    )
                )
            for engine, patterns in ENGINE_PATTERNS.items():
                seen = found.setdefault(engine, {})
                for name, pattern in patterns.items():
                    if (name in seen and name not in LAST_VALUE) or not (
                        match := re.search(pattern, line)
                    ):
                        continue
                    if name == "max_concurrency":
                        seen[name] = (
                            f"{match.group(2)}x at {match.group(1)} tokens/request"
                        )
                    else:
                        seen[name] = match.group(1)
    found = {
        engine: values
        for engine, values in found.items()
        if values and engine in anchored
    }
    if "vllm" in found:
        found["vllm"]["cuda_graph_capture_seen"] = str(captured)
    if stats:
        found.setdefault("vllm", {})["engine_stats"] = (
            f"{len(stats)} samples; running max {max(s[0] for s in stats)}, "
            f"waiting max {max(s[1] for s in stats)} "
            f"(>0 in {sum(s[1] > 0 for s in stats) / len(stats):.0%} of samples), "
            f"KV usage max {max(s[2] for s in stats):.1f}%"
            + (
                f", last prefix-cache hit rate {stats[-1][3]:.1f}%"
                if stats[-1][3] is not None
                else ""
            )
        )
        found["vllm"]["_waiting_frac"] = str(sum(s[1] > 0 for s in stats) / len(stats))
        found["vllm"]["_kv_max"] = str(max(s[2] for s in stats))
    found["_config"] = config
    return found


def diagnose(
    metrics: dict, log: dict[str, dict[str, str]], max_new_tokens: int | None
) -> None:
    step = series(metrics, STEP)
    is_async = not series(metrics, GEN)
    gen = series(metrics, GEN) or series(metrics, PHASES["exposed generation (async)"])
    gen_label = "Exposed (non-overlapped) generation" if is_async else "Generation"
    steps = sorted(s for s in set(gen) & set(step) if step[s] > 0)
    print(f"steps with timing: {len(steps)}")
    share = med(gen[s] / step[s] for s in steps)
    iqr = math.nan
    if steps:
        gen_vals = [gen[s] for s in steps]
        gen_med = med(gen_vals)
        quartiles = (
            statistics.quantiles(gen_vals, n=4) if len(gen_vals) >= 2 else [gen_med] * 3
        )
        iqr = (quartiles[2] - quartiles[0]) / gen_med if gen_med > 0 else math.nan
        print(f"total step time, median: {med(step[s] for s in steps):.1f} s")
        print(
            f"{gen_label.lower()} per step: median {gen_med:.1f} s, max {max(gen_vals):.1f} s "
            f"(interquartile spread {iqr:.0%})"
        )
        print("step breakdown (median share of total_step_time):")
        for label, key in PHASES.items():
            values = series(metrics, key)
            shares = [values[s] / step[s] for s in steps if s in values]
            if shares:
                print(f"  {label:<28} {med(shares):>6.1%}")
    val = series(metrics, VAL)
    if val:
        print(f"validation: {len(val)} runs, {sum(val.values()):.1f} s total")
    setup = series(metrics, SETUP)
    if setup:
        print(f"setup: {sum(setup.values()):.1f} s")
    prompt, mean_gen, max_gen = (
        series(metrics, PROMPT),
        series(metrics, MEAN_GEN),
        series(metrics, MAX_GEN),
    )
    if prompt and mean_gen:
        print(
            f"tokens per sample: prompt {med(prompt.values()):.0f}, "
            f"generated mean {med(mean_gen.values()):.0f}"
        )
    turns = series(metrics, TURNS)
    if turns:
        print(f"turns per sample, median: {med(turns.values()):.1f}")
    trunc = series(metrics, TRUNC)
    if trunc:
        print(f"truncation rate, median: {med(trunc.values()):.1%}")
    config = log.get("_config", {})
    cap = max_new_tokens or (
        int(config["max_new_tokens"]) if "max_new_tokens" in config else None
    )
    hits = None
    if max_gen and cap:
        hits = sum(v >= cap for v in max_gen.values()) / len(max_gen)
        print(f"steps whose longest sample hit max_new_tokens={cap}: {hits:.0%}")
    tok_s = series(metrics, "performance/generation_tokens_per_sec")
    if tok_s:
        print(
            f"generation tokens/s, median: {med(tok_s.values()):.0f} "
            "(NeMo-RL counts prompt + generated tokens)"
        )
    engines = [e for e in log if not e.startswith("_")]
    # Dynamo runs also show vLLM worker lines; prefer the more specific backend.
    backend = config.get("backend") or (
        "dynamo" if "dynamo" in engines else next(iter(engines), None)
    )
    if backend:
        print(f"generation backend: {backend}")
    for engine, values in log.items():
        if engine.startswith("_"):
            continue
        for name, value in values.items():
            if not name.startswith("_"):
                print(f"log [{engine}]: {name} = {value}")
    norm = normalize(log)
    if log:
        print("engine-neutral settings (from engine log lines):")
        for field in (
            "cuda_graphs",
            "prefix_reuse",
            "token_budget",
            "chunked_prefill",
            "admission_cap",
            "memory_fraction",
            "kv_capacity",
        ):
            value, source = norm.get(field, ("unproven", "-"))
            print(f"  {field:<16} {value:<24} [{source}]")

    print("\ndiagnosis:")
    if math.isnan(share):
        print("- No generation timing in metrics.json; cannot diagnose the rollout.")
        return
    refit = series(metrics, PHASES["refit"]) or series(
        metrics, PHASES["weight sync (async)"]
    )
    refit_share = med(refit[s] / step[s] for s in steps if s in refit)
    if not math.isnan(refit_share) and refit_share >= 0.15:
        print(
            f"- Refit takes {refit_share:.0%} of the step: check refit transport and buffer settings."
        )
    if share < 0.5:
        print(
            f"- {gen_label} is {share:.0%} of step time, which bounds what rollout tuning can "
            "save; look at the other phases too."
        )
    else:
        print(f"- Generation-bound ({gen_label.lower()} is {share:.0%} of step time).")
    if hits is not None and hits >= 0.8 and not iqr >= 0.1:
        print(
            "- Decode-tail-bound: the longest response sets the step. Levers: CUDA graphs, "
            "more and smaller replicas, kernel backend."
        )
    prefill_heavy = bool(
        prompt and mean_gen and med(prompt.values()) > 4 * med(mean_gen.values())
    ) or bool(turns and med(turns.values()) > 1.5)
    if prefill_heavy:
        print(
            "- Prefill-heavy or multi-turn: check the scheduler token budget, prefix/KV reuse "
            "and session affinity."
        )
    graphs = norm.get("cuda_graphs", ("unproven", ""))[0]
    reuse = norm.get("prefix_reuse", ("unproven", ""))[0]
    if graphs == "off":
        hint = (
            " and A/B the graph mode (default FULL_AND_PIECEWISE vs PIECEWISE)"
            if backend in ("vllm", "dynamo")
            else ""
        )
        print(
            f"- {backend or 'The engine'} runs without CUDA graphs: A/B enabling them "
            f"({GRAPH_KNOB.get(backend or '', 'see references/engines.md')}){hint}."
        )
    elif graphs == "unproven" and backend:
        print(
            f"- CUDA-graph state is not proven from the {backend} log; check "
            f"{GRAPH_KNOB.get(backend, 'the engine config')} and references/engines.md."
        )
    if prefill_heavy and reuse != "on":
        print(
            f"- Prefix/KV reuse is {reuse}: for repeated prefixes, A/B it "
            f"({REUSE_KNOB.get(backend or '', 'see references/levers.md §3')})."
        )
    unproven = [
        field
        for field in ("cuda_graphs", "prefix_reuse", "token_budget", "admission_cap")
        if field not in norm
    ]
    if backend and len(unproven) > ("cuda_graphs" in unproven):
        print(
            f"- The {backend} log does not prove: {', '.join(unproven)}. Treat the resolved "
            "config values as requested only (references/engines.md)."
        )
    vllm = log.get("vllm", {})
    if "max_concurrency" in vllm:
        print(
            f"- vLLM KV capacity per replica: {vllm['max_concurrency']}; memory knobs only help "
            "if this is below the per-replica load."
        )
    if "_waiting_frac" in vllm:
        waiting, kv = float(vllm["_waiting_frac"]), float(vllm["_kv_max"])
        if waiting == 0 and kv < 20:
            print(
                f"- Engine never queued requests and KV use peaked at {kv:.0f}%: budget and memory "
                "knobs will not help; if GPUs idle, look at the feeder (tokenizer, HTTP workers, harness)."
            )
        elif waiting >= 0.2:
            print(
                f"- Requests waited in the engine queue in {waiting:.0%} of stats samples: check "
                "the admission cap, token budget and KV capacity."
            )
    if not any(engine for engine in log if not engine.startswith("_")) and log:
        print(
            "- No engine startup lines found in the log; prove the effective config by hand "
            "(see references/engines.md)."
        )


def load_arm(paths: list[Path]) -> list[dict]:
    return [json.loads(path.read_text()) for path in paths]


def arm_series(runs: list[dict], key: str) -> dict[int, float]:
    """Per-step median across the runs of one arm, over steps all runs logged."""
    per_run = [series(run, key) for run in runs]
    common = set.intersection(*(set(s) for s in per_run)) if per_run else set()
    return {step: med(s[step] for s in per_run) for step in sorted(common)}


def compare(base: list[dict], treat: list[dict]) -> None:
    print(f"runs per arm: base {len(base)}, treat {len(treat)}")
    if len(treat) < 3:
        print(
            "  note: fewer than 3 treatment runs; grade C evidence at best "
            "(see references/measurement.md)."
        )
    elif len(base) < 3:
        print("  note: fewer than 3 control runs; grade B evidence at best.")
    gb, gt = arm_series(base, GEN), arm_series(treat, GEN)
    paired = sorted(s for s in set(gb) & set(gt) if gb[s] > 0)
    if not paired:
        print("warning: no steps with generation timing are shared by both arms.")
    if paired:
        ratios = [gt[s] / gb[s] for s in paired]
        print(
            f"paired steps: {len(paired)}; generation ratio treat/base median "
            f"{med(ratios):.3f} (min {min(ratios):.3f}, max {max(ratios):.3f})"
        )
        for name, arm in (("base", gb), ("treat", gt)):
            values = [arm[s] for s in paired]
            p90 = (
                statistics.quantiles(values, n=10)[-1]
                if len(values) >= 2
                else values[0]
            )
            print(
                f"  [{name}] generation per step p50 {med(values):.1f} s, "
                f"p90 {p90:.1f} s, max {max(values):.1f} s"
            )
    rows: list[tuple[str, str, Callable[[list[float]], float]]] = [
        ("generation per step, median (s)", GEN, med),
        ("generation, sum over paired steps (s)", GEN, sum),
        ("total step time, sum over paired steps (s)", STEP, sum),
        ("refit, sum over paired steps (s)", PHASES["refit"], sum),
        ("validation, sum (s)", VAL, sum),
        ("setup (s)", SETUP, sum),
        ("generated tokens per sample, median", MEAN_GEN, med),
        ("token_mult_prob_error, median", TMPE, med),
        ("gen_kl_error, mean", GEN_KL, statistics.mean),
    ]
    for label, key, fn in rows:
        b, t = arm_series(base, key), arm_series(treat, key)
        common = sorted(set(b) & set(t))
        if not common:
            continue
        vb, vt = fn([b[s] for s in common]), fn([t[s] for s in common])
        print(f"{label:<44} {vb:>12.5g} -> {vt:>12.5g} ({rel_change(vb, vt)})")
        if key == MEAN_GEN and vb > 0 and abs(vt / vb - 1) > 0.05:
            print(
                "  warning: generated tokens per sample moved by more than 5%; "
                "the workload changed, not just the engine."
            )
    for name, runs in (("base", base), ("treat", treat)):
        for index, run in enumerate(runs, 1):
            tmpe = series(run, TMPE)
            acc = series(run, VAL_ACC)
            spikes = sum(v >= 1.04 for v in tmpe.values())
            accs = ", ".join(f"step {s}: {acc[s]:.3f}" for s in sorted(acc))
            print(
                f"[{name} run {index}] steps with token_mult_prob_error >= 1.04: "
                f"{spikes}/{len(tmpe)}; validation accuracy {accs or 'n/a'}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "metrics", nargs="?", type=Path, help="metrics.json of one run to diagnose"
    )
    parser.add_argument(
        "--log", type=Path, help="ray-driver.log of the same run (optional)"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, help="generation cap, if not found in the log"
    )
    parser.add_argument(
        "--base", type=Path, nargs="+", help="control metrics.json file(s)"
    )
    parser.add_argument(
        "--treat", type=Path, nargs="+", help="treatment metrics.json file(s)"
    )
    args = parser.parse_args()
    if args.base and args.treat:
        compare(load_arm(args.base), load_arm(args.treat))
    elif args.metrics:
        log = scan_log(args.log) if args.log else {}
        diagnose(json.loads(args.metrics.read_text()), log, args.max_new_tokens)
    else:
        parser.error("pass one metrics.json, or --base and --treat")


if __name__ == "__main__":
    main()
