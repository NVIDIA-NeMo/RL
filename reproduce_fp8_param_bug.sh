#!/usr/bin/env bash
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
#
# Reproducer for the fp8_param convergence bug in mcore with MXFP8:
#   https://github.com/NVIDIA-NeMo/RL/issues/1164
#
# With deterministic_mode=true every source of non-determinism is eliminated,
# so the three modes below should be directly comparable across runs.
#
# Modes:
#   mxfp8-fp8param   MXFP8 recipe with fp8_param=true   (buggy — diverges)
#   mxfp8-noparam    MXFP8 recipe with fp8_param=false  (baseline)
#   bf16             FP8 disabled entirely              (pure bf16 reference)
#
# Prerequisites:
#   - Single node with 8 B100/B200 GPUs (Blackwell, CUDA >= 12.9)
#     MXFP8 is only supported on Blackwell architecture (compute capability >= 10.0)
#   - Run from the root of the NeMo-RL repo inside the nemo-rl container
#
# Usage:
#   bash reproduce_fp8_param_bug.sh --mode mxfp8-fp8param
#   bash reproduce_fp8_param_bug.sh --mode mxfp8-noparam
#   bash reproduce_fp8_param_bug.sh --mode bf16

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="${SCRIPT_DIR}/examples/configs/fp8_param_bug_reproducer.yaml"

# ---------- defaults ----------
MODE="mxfp8-noparam"

# ---------- arg parsing ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ---------- mode → config overrides ----------
case "${MODE}" in
    mxfp8-fp8param)
        FP8_ENABLED="true"
        FP8_PARAM="true"
        ;;
    mxfp8-noparam)
        FP8_ENABLED="true"
        FP8_PARAM="false"
        ;;
    bf16)
        # fp8_param value is irrelevant when fp8_cfg.enabled=false, but pass
        # something valid so the override doesn't trip type checks downstream.
        FP8_ENABLED="false"
        FP8_PARAM="false"
        ;;
    *)
        echo "Error: --mode must be one of: mxfp8-fp8param, mxfp8-noparam, bf16" >&2
        exit 1
        ;;
esac

WANDB_PROJECT="nrl_fp8_param_debug"
WANDB_RUN="${MODE}"
TB_DIR="tb_logs-${MODE}"
LOG_DIR="logs/fp8_param_bug/run_${MODE}"

echo "================================================================"
echo "  FP8 param bug reproducer"
echo "  mode           : ${MODE}"
echo "  fp8_cfg.enabled: ${FP8_ENABLED}"
echo "  fp8_param      : ${FP8_PARAM}"
echo "  wandb project  : ${WANDB_PROJECT}"
echo "  wandb run name : ${WANDB_RUN}"
echo "  TensorBoard dir: ${TB_DIR}"
echo "  Config         : ${CFG}"
echo "================================================================"

uv run "${SCRIPT_DIR}/examples/run_sft.py" \
    --config "${CFG}" \
    "policy.megatron_cfg.fp8_cfg.enabled=${FP8_ENABLED}" \
    "policy.megatron_cfg.fp8_cfg.fp8_param=${FP8_PARAM}" \
    "logger.tensorboard.log_dir=${TB_DIR}" \
    "logger.log_dir=${LOG_DIR}" \
    "logger.wandb_enabled=true" \
    "logger.wandb.project=${WANDB_PROJECT}" \
    "logger.wandb.name=${WANDB_RUN}"
