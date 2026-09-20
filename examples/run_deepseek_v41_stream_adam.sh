#!/bin/bash
# Source-integrated launcher. Requires a matching CUDA/Torch/AutoModel/vLLM environment.
set -euo pipefail
: "${DS41_FULL_CHECKPOINT:?Set a verified full checkpoint directory}"
: "${DS41_SMOKE_DATA:?Set the full smoke JSONL data path}"
export DS41_DEFER_GRAD_OFFLOAD=1 DS41_GPU_GRAD_NORM=1 DS41_STREAM_ADAM=1
export NRL_REFIT_SERIAL_COLLECTIVES=1
export NRL_REFIT_BUFFER_MEMORY_RATIO=0.3 NRL_REFIT_BUFFER_MAX_GB=4
cd "$(dirname "$0")/.."
exec uv run examples/run_grpo.py \
  --config examples/configs/recipes/llm/grpo-deepseek-v4.1-10n4g-fsdp2tp1-noncolocated-stream-adam.yaml "$@"
