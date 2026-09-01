#!/usr/bin/env python3
"""Fail closed when an aligned-v1 OSWorld Molt recipe drifts."""

from __future__ import annotations

import os
import sys


REFERENCE_REPO = (
    "/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_reasoning/"
    "users/jianh/projects/OpenRLHF-main"
)
REFERENCE_COMMIT = "26f086ddabbb45a09e00c0b1f9962cdd2863159c"
BASE_MODEL = f"{REFERENCE_REPO}/ckpts/rfc0037-sft-step400"

PROFILES = {
    "b1k1": {
        "OSWORLD_NUM_PROMPTS_PER_STEP": "1",
        "MOLT_VLLM_GENERATE_BATCH_SIZE": "1",
        "OSWORLD_NEMO_GYM_NUM_WORKERS": "8",
        "OSWORLD_MAX_PARALLEL_ROLLOUTS": "8",
    },
    "b4k4": {
        "OSWORLD_NUM_PROMPTS_PER_STEP": "4",
        "MOLT_VLLM_GENERATE_BATCH_SIZE": "4",
        "OSWORLD_NEMO_GYM_NUM_WORKERS": "32",
        "OSWORLD_MAX_PARALLEL_ROLLOUTS": "32",
    },
    "b8k8": {
        "OSWORLD_NUM_PROMPTS_PER_STEP": "8",
        "MOLT_VLLM_GENERATE_BATCH_SIZE": "8",
        "OSWORLD_NEMO_GYM_NUM_WORKERS": "32",
        "OSWORLD_MAX_PARALLEL_ROLLOUTS": "32",
    },
}

COMMON = {
    "NANO_OMNI_MODEL_NAME": BASE_MODEL,
    "OSWORLD_NUM_GENERATIONS": "8",
    "MOLT_ASYNC_QUEUE_SIZE": "2",
    "MOLT_MAX_STALENESS": "1",
    "NUM_NODES": "4",
    "MOLT_INFERENCE_NODES": "3",
    "MOLT_TENSOR_PARALLEL_SIZE": "1",
    "MOLT_EXPERT_MODEL_PARALLEL_SIZE": "8",
    "MOLT_CONTEXT_PARALLEL_SIZE": "8",
    "MOLT_PIPELINE_PARALLEL_SIZE": "1",
    "OSWORLD_LEARNING_RATE": "5e-6",
    "MOLT_ADAM_BETA1": "0.9",
    "MOLT_ADAM_BETA2": "0.99",
    "MOLT_LR_WARMUP_ITERS": "9",
    "MOLT_ROUTER_REPLAY_ENABLED": "true",
    "MOLT_IS_RATIO_MIN": "0.99",
    "MOLT_IS_RATIO_MAX": "1.01",
    "OSWORLD_TEMPERATURE": "1.0",
    "OSWORLD_TOP_P": "1.0",
    "OSWORLD_MAX_MODEL_LEN": "49152",
    "OSWORLD_MAX_NEW_TOKENS": "16384",
    "OSWORLD_MAX_STEPS": "150",
    "OSWORLD_ACTION_TIMEOUT_S": "60",
    "OSWORLD_LLM_TIMEOUT_S": "900",
    "OSWORLD_ROLLOUT_TIMEOUT_S": "1200",
    "MOLT_ROLLOUT_BATCH_TIMEOUT_S": "3600",
    "OSWORLD_MAX_IMAGE_HISTORY_LENGTH": "3",
    "OSWORLD_MAX_ACTIVE_IMAGES": "10",
    "OSWORLD_SLEEP_AFTER_EXECUTION": "5",
    "OSWORLD_VLLM_GPU_MEMORY_UTILIZATION": "0.8",
    "OSWORLD_VLLM_ENABLE_PREFIX_CACHING": "true",
    "OSWORLD_VLLM_ENABLE_CHUNKED_PREFILL": "true",
    "OSWORLD_VLLM_MAX_NUM_BATCHED_TOKENS": "4096",
    "OSWORLD_VLLM_EXPERT_PARALLEL_SIZE": "2",
    "MOLT_SEQUENCE_PACKING_ENABLED": "true",
    "MOLT_TRAIN_MB_TOKENS": "49152",
    "MOLT_LOGPROB_MB_TOKENS": "49152",
    "NEMOTRON_OMNI_VISION_CHUNK_SIZE": "1",
    "NEMOTRON_OMNI_VISION_CACHE_MAX_ENTRIES": "0",
    "RAY_ENABLE_ZERO_COPY_TORCH_TENSORS": "1",
    "CHECKPOINT_SAVE_PERIOD": "1",
    "CHECKPOINT_KEEP_TOP_K": "2",
    "MOLT_SAVE_OPTIMIZER": "true",
}


def main() -> int:
    profile = os.environ.get("MOLT_ALIGNMENT_PROFILE", "")
    if profile not in PROFILES:
        print(f"ABORT: unknown MOLT_ALIGNMENT_PROFILE={profile!r}", file=sys.stderr)
        return 2

    expected = COMMON | PROFILES[profile]
    mismatches = [
        f"{key}: expected {value!r}, got {os.environ.get(key)!r}"
        for key, value in expected.items()
        if os.environ.get(key) != value
    ]

    run_name = os.environ.get("MOLT_RUN_NAME", "")
    expected_suffix = f"aligned-v1-{profile}"
    if expected_suffix not in run_name:
        mismatches.append(
            f"MOLT_RUN_NAME must contain {expected_suffix!r}, got {run_name!r}"
        )

    if mismatches:
        print("ABORT: aligned-v1 recipe drift detected:", file=sys.stderr)
        for mismatch in mismatches:
            print(f"  - {mismatch}", file=sys.stderr)
        return 2

    print(
        f"Aligned-v1 preflight passed: profile={profile} "
        f"reference={REFERENCE_COMMIT[:12]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
