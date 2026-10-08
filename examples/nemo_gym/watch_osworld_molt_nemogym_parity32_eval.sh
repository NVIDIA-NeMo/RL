#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
RUNTIME_ROOT="${OSWORLD_RUNTIME_ROOT:-$(dirname "${ROOT}")/osworld-cc-runtime}"
RUN_NAME=osworld361-molt-nemogym-parity32

export MOLT_RUN_NAME="${RUN_NAME}"
export MOLT_CHECKPOINT_DIR="${RUNTIME_ROOT}/results/${RUN_NAME}/checkpoints"
export OSWORLD_GRPO_VAL_DATA="${RUNTIME_ROOT}/data/overfit361-cc-v2/validation-1x.jsonl"
export EVAL_EVERY=10
export EVAL_POLL_SECONDS=60

# Keep evaluation generation identical to Jianh's 361-task mean@4 protocol.
export OSWORLD_VLLM_GPU_MEMORY_UTILIZATION=0.8
export OSWORLD_VLLM_ENABLE_PREFIX_CACHING=true
export OSWORLD_VLLM_ENABLE_CHUNKED_PREFILL=true
export OSWORLD_VLLM_MAX_NUM_BATCHED_TOKENS=4096
export OSWORLD_VLLM_EXPERT_PARALLEL_SIZE=2
export OSWORLD_VLLM_ATTENTION_BACKEND=TRITON_ATTN
export OSWORLD_VLLM_MM_ENCODER_ATTN_BACKEND=TORCH_SDPA
export OSWORLD_VLLM_GDN_PREFILL_BACKEND=triton

exec "${ROOT}/examples/nemo_gym/watch_osworld_molt_eval.sh"
