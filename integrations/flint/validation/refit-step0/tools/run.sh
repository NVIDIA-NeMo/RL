#!/usr/bin/env bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
umask 077
cd /flint-work/RL
export PYTHONPATH=/flint-audit/tools:/flint-work/RL:/flint-work/tools:/flint-work/sglang-plugin/src
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=disabled RAY_USAGE_STATS_ENABLED=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export SGLANG_EXTERNAL_MODEL_PACKAGE=flint_sglang FLASHINFER_USE_CUDA_NORM=1
export FLINT_TRITON_CACHE_BASE=/flint-audit/cache/triton
export TMPDIR=/tmp/flint-step0-audit
export XDG_CACHE_HOME=/flint-audit/cache
export FLINT_STEP0_SOURCE=/flint-work/checkpoint
export FLINT_STEP0_OUTPUT=/flint-audit/tensors
export FLINT_SYNTHETIC_SMOKE=1
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" /flint-audit/evidence
unset RAY_ADDRESS NRL_IGNORE_VERSION_MISMATCH FLINT_STEP0_ENABLE
/opt/flint/bin/python-sglang /flint-audit/tools/test_step0_audit.py > /flint-audit/evidence/unit-tests.log 2>&1
/opt/flint/bin/python-driver /flint-audit/tools/launch_audit.py > /flint-audit/evidence/training.log 2>&1 &
flint_driver=$!
trap 'kill "$flint_driver" 2>/dev/null || true' TERM INT
set +e
wait "$flint_driver"
flint_status=$?
set -e
printf '%s\n' "$flint_status" > /flint-audit/evidence/training.exit-code
exit "$flint_status"
