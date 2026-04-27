# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Offline inference smoke test for DeepSeek-V4-Flash-**Base** (FP8-block-quant).

Counterpart to ``tools/test_vllm_dsv4_inference.py`` (which targets the
FP4-packed Flash variant). Drives Base via the env-gated quick-patch in
``tools/patch_vllm_dsv4_base_fp8_quick.sh``: routed experts go through
``Fp8MoEMethod`` (block-quant), ``.scale`` -> ``.weight_scale_inv``,
``indexer.k_norm`` -> ``Identity``, ``use_mega_moe`` forced ``False``.

Sets ``VLLM_DSV4_BASE_FP8=1`` BEFORE importing vLLM. Assumes the patch
is already baked into the worker venv (env_refresh_dsv4.sh applies it
during sqsh bake). Runs ``patch_vllm_dsv4_base_fp8_quick.sh`` inline
again as a defensive idempotent re-application — safe no-op if already
patched.

Run inside the baked NeMo-RL DSV4 sqsh with the VllmGenerationWorker venv:

    /opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/python \\
        tools/test_vllm_dsv4_base_inference.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


# Set BEFORE any vllm import — the patched code paths gate on this env var.
os.environ.setdefault("VLLM_DSV4_BASE_FP8", "1")


DEFAULT_MODEL = (
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/"
    "users/shuangy/models/deepseek-ai/DeepSeek-V4-Flash-Base"
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
    p.add_argument(
        "--skip-patch-reapply",
        action="store_true",
        help="Skip the idempotent re-application of patch_vllm_dsv4_base_fp8_quick.sh.",
    )
    return p.parse_args()


def reapply_base_patch() -> None:
    """Idempotent: prints `already patched` if applied during sqsh bake."""
    repo_root = Path(__file__).resolve().parents[1]
    patch = repo_root / "tools" / "patch_vllm_dsv4_base_fp8_quick.sh"
    if not patch.exists():
        print(f"[warn] {patch} missing; skipping re-application", file=sys.stderr)
        return
    print(f"[info] re-applying {patch.name} against {sys.executable} (idempotent)")
    res = subprocess.run(
        ["bash", str(patch), sys.executable],
        capture_output=True,
        text=True,
    )
    sys.stdout.write(res.stdout)
    if res.returncode != 0:
        sys.stderr.write(res.stderr)
        raise SystemExit(f"patch script failed (rc={res.returncode})")


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
    """Same vllm_kwargs shape as Flash test, plus env gate set above."""
    vllm_kwargs: dict = {
        "kv_cache_dtype": "fp8",
        "tokenizer_mode": "deepseek_v4",
    }
    if args.distributed_executor_backend is not None:
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

    print(f"[info] python                = {sys.executable}")
    print(f"[info] VLLM_DSV4_BASE_FP8    = {os.environ.get('VLLM_DSV4_BASE_FP8')!r}")
    print(f"[info] model                 = {args.model}")
    print(f"[info] TP                    = {args.tensor_parallel_size}")
    print(f"[info] EP                    = {args.enable_expert_parallel}")
    print(f"[info] max_len               = {args.max_model_len}")
    print(f"[info] gpu_mem               = {args.gpu_memory_utilization}")
    print(f"[info] max_tokens            = {args.max_tokens}")
    print(f"[info] prompts               = {len(prompts)}")

    if not args.skip_patch_reapply:
        reapply_base_patch()

    import vllm

    print(f"[info] vllm                  = {vllm.__version__}")

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
