# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Offline inference smoke test for DeepSeek-V4-Flash through NeMo-RL's vLLM path.

Constructs the same ``llm_kwargs`` dict that
``nemo_rl.models.generation.vllm.vllm_worker.BaseVllmGenerationWorker`` builds at
vllm_worker.py:539-563, calls ``vllm.LLM(**llm_kwargs)`` directly (no Ray actor
wrapper), and runs a short greedy completion. Mirrors ``generate_text``
(vllm_worker.py:826-886) and ``_build_sampling_params`` (vllm_worker.py:591-616).

DSV4-specific flags come from the recipes.vllm.ai DSV4 recipe we validated via
``vllm serve``: ``--tokenizer-mode deepseek_v4``, ``--enable-expert-parallel``,
``--enforce-eager``, ``--kv-cache-dtype fp8``, ``--max-model-len 16384``,
``--gpu-memory-utilization 0.85``, ``--trust-remote-code``. The
``--reasoning-parser`` flag is server-only and intentionally skipped.

Run inside the baked NeMo-RL DSV4 sqsh with the VllmGenerationWorker venv:

    /opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/python \\
        tools/test_vllm_dsv4_inference.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


DEFAULT_MODEL = (
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/"
    "users/shuangy/models/deepseek-ai/DeepSeek-V4-Flash"
)
DEFAULT_PROMPT = "The capital of France is"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default=DEFAULT_MODEL, help="Model path or HF id")
    p.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="Prompt string (repeat for multiple). Default: one canonical prompt.",
    )
    p.add_argument(
        "--prompts-file",
        type=Path,
        default=None,
        help="Optional file with one prompt per line; combined with --prompt.",
    )
    p.add_argument("--tensor-parallel-size", type=int, default=8)
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument(
        "--no-enable-expert-parallel",
        dest="enable_expert_parallel",
        action="store_false",
    )
    p.set_defaults(enable_expert_parallel=True)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--distributed-executor-backend",
        default=None,
        choices=[None, "mp", "ray", "uni", "external_launcher"],
        help="None lets vLLM pick (mp when TP>1 outside Ray). Override if needed.",
    )
    return p.parse_args()


def collect_prompts(args: argparse.Namespace) -> list[str]:
    prompts: list[str] = list(args.prompt or [])
    if args.prompts_file is not None:
        prompts.extend(
            line.rstrip("\n")
            for line in args.prompts_file.read_text().splitlines()
            if line.strip()
        )
    if not prompts:
        prompts = [DEFAULT_PROMPT]
    return prompts


def build_llm_kwargs(args: argparse.Namespace) -> dict:
    """Mirror nemo_rl.models.generation.vllm.vllm_worker.py:539-563.

    DSV4-specific fields go through ``vllm_kwargs`` passthrough (same as NeMo-RL
    recipes do via ``policy.generation.vllm_kwargs``).
    """
    vllm_kwargs: dict = {
        # From recipes.vllm.ai DSV4 recipe.
        "kv_cache_dtype": "fp8",
        "tokenizer_mode": "deepseek_v4",
    }
    if args.distributed_executor_backend is not None:
        # NeMo-RL sets this to "ray" when TP*PP>1 inside its Ray cluster; we're
        # standalone, so we either let vLLM pick (None) or force "mp".
        vllm_kwargs["distributed_executor_backend"] = args.distributed_executor_backend

    return dict(
        model=args.model,
        served_model_name=args.model,
        load_format="auto",
        skip_tokenizer_init=False,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=1,
        enable_expert_parallel=args.enable_expert_parallel,
        gpu_memory_utilization=args.gpu_memory_utilization,
        # NeMo-RL defaults this based on CC>=8; hard-code True (we target H100/H200).
        enable_prefix_caching=True,
        dtype=args.dtype,
        seed=args.seed,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        worker_extension_cls=(
            "nemo_rl.models.generation.vllm.vllm_backend.VllmInternalWorkerExtension"
        ),
        enable_sleep_mode=True,
        disable_log_stats=False,
        logprobs_mode="processed_logprobs",
        **vllm_kwargs,
    )


def build_sampling_params(args: argparse.Namespace):
    """Mirror BaseVllmGenerationWorker._build_sampling_params (vllm_worker.py:591-616)."""
    import vllm

    greedy = args.temperature == 0.0
    return vllm.SamplingParams(
        temperature=0.0 if greedy else args.temperature,
        top_p=args.top_p,
        top_k=1 if greedy else -1,
        max_tokens=args.max_tokens,
        logprobs=0,
        stop_token_ids=None,
        stop=None,
        include_stop_str_in_output=True,
    )


def main() -> int:
    args = parse_args()
    prompts = collect_prompts(args)

    print(f"[info] python     = {sys.executable}")
    print(f"[info] model      = {args.model}")
    print(f"[info] TP         = {args.tensor_parallel_size}")
    print(f"[info] EP         = {args.enable_expert_parallel}")
    print(f"[info] max_len    = {args.max_model_len}")
    print(f"[info] gpu_mem    = {args.gpu_memory_utilization}")
    print(f"[info] max_tokens = {args.max_tokens}")
    print(f"[info] prompts    = {len(prompts)}")

    # Import here so the log line above lands even if vLLM fails to import.
    import vllm

    print(f"[info] vllm       = {vllm.__version__}")

    llm_kwargs = build_llm_kwargs(args)
    sampling_params = build_sampling_params(args)

    print("[info] llm_kwargs:")
    for k, v in sorted(llm_kwargs.items()):
        print(f"  {k}: {v!r}")

    t0 = time.time()
    print(f"[info] constructing vllm.LLM ... ({time.strftime('%H:%M:%S')})")
    llm = vllm.LLM(**llm_kwargs)
    print(f"[info] vllm.LLM ready in {time.time() - t0:.1f}s")

    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0

    print(f"\n=== generations ({elapsed:.1f}s total) ===")
    for prompt, out in zip(prompts, outputs):
        text = out.outputs[0].text
        ntok = len(out.outputs[0].token_ids)
        print(f"[prompt]  {prompt!r}")
        print(f"[output]  tokens={ntok}  text={text!r}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
