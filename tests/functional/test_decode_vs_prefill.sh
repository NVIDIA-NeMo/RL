#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")

cd "$PROJECT_ROOT"

# vLLM sizes the KV cache from gpu_memory_utilization after its profiling run, and the
# FlashInfer TRT-LLM MoE kernels allocate their workspace (~4 GiB for this model) lazily
# after that. On 186 GiB GB200 parts 0.8 fills the card with KV cache and that workspace
# allocation OOMs at the first MoE forward; on 80 GiB H100 parts the model (29.5 GiB per
# TP rank) plus the profiled activation peak already take ~58 GiB, so 0.7 leaves no room
# for any KV cache at all. Pick the fraction from the device size.
GPU_MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
if [ "${GPU_MEM_MIB:-0}" -gt 120000 ]; then
    GPU_MEM_UTIL=0.7
else
    GPU_MEM_UTIL=0.8
fi

uv run --extra vllm coverage run -a --data-file="$PROJECT_ROOT/tests/.coverage" --source="$PROJECT_ROOT/nemo_rl" \
    tools/model_diagnostics/2.long_generation_decode_vs_prefill.py \
    --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16 \
    --prompts arc \
    --max-tokens 8192 \
    --num-batches 4 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
