#!/usr/bin/env bash
set -euo pipefail
umask 077
cd /flint-work/RL
export PYTHONPATH=/flint-work/RL:/flint-work/tools:/flint-work/sglang-plugin/src
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=disabled RAY_USAGE_STATS_ENABLED=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export SGLANG_EXTERNAL_MODEL_PACKAGE=flint_sglang FLASHINFER_USE_CUDA_NORM=1
export FLINT_TRITON_CACHE_BASE=/flint-work/cache/triton
export TMPDIR=/tmp/flint-grpo-8075956
export XDG_CACHE_HOME=/flint-work/cache
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
unset RAY_ADDRESS NRL_IGNORE_VERSION_MISMATCH
flint_config=examples/configs/recipes/llm/grpo-flint-4layer-1n4g-automodel-sglang-quick.yaml
if [[ "${FLINT_SYNTHETIC_SMOKE:-0}" == 1 ]]; then
  flint_config=examples/configs/recipes/llm/grpo-flint-4layer-1n4g-automodel-sglang-synthetic-quick.yaml
fi
exec /opt/flint/bin/python-driver /flint-work/tools/launch_grpo.py \
  --config "$flint_config" "$@"
