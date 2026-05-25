#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""Smoke test: bring up sglang Engine on the sliced 4-layer Qwen3-30B-A3B
checkpoint with a configurable ``moe_runner_backend`` and run one greedy
generate. No weight update at all — just verifies the engine can stand up
under this backend.
"""

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--moe-runner-backend",
        type=str,
        default="flashinfer_trtllm_routed",
        help="sglang moe_runner_backend (e.g. flashinfer_trtllm, flashinfer_trtllm_routed, triton, auto).",
    )
    args = parser.parse_args()
    ckpt = (
        Path(os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"))
        / "hub"
        / "qwen3-30b-a3b-sliced-4"
    )
    if not ckpt.is_dir():
        sys.exit(f"sliced ckpt not found at {ckpt}")
    print(f"[smoke] ckpt={ckpt}", flush=True)
    print(
        f"[smoke] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
        flush=True,
    )
    print(f"[smoke] moe_runner_backend={args.moe_runner_backend}", flush=True)

    from sglang.srt.entrypoints.engine import Engine

    engine = Engine(
        model_path=str(ckpt),
        tp_size=1,
        pp_size=1,
        dp_size=1,
        mem_fraction_static=0.5,
        log_level="info",
        random_seed=42,
        disable_cuda_graph=True,
        dtype="bfloat16",
        moe_runner_backend=args.moe_runner_backend,
    )
    print("[smoke] engine up", flush=True)

    # Confirm the actually-resolved dtype + MoE backend the engine picked.
    try:
        sa = engine.tokenizer_manager.server_args
        print(
            f"[smoke] resolved dtype={getattr(sa, 'dtype', '<?>')!r} "
            f"moe_runner_backend={getattr(sa, 'moe_runner_backend', '<?>')!r}",
            flush=True,
        )
    except Exception as e:
        print(f"[smoke] could not introspect server_args: {e!r}", flush=True)

    prompt = "The capital of France is"
    out = engine.generate(
        prompt, sampling_params={"max_new_tokens": 16, "temperature": 0.0}
    )
    print(f"[smoke] generate output: {out!r}", flush=True)
    print("[smoke] PASS — engine started and produced output", flush=True)

    engine.shutdown()


if __name__ == "__main__":
    main()
