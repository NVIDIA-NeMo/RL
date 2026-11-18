#!/bin/bash
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

# Golden values on a 4-layer, 8-expert Nemotron 3.5 Super VL with random weights:
# record two steps with Megatron-Bridge + Megatron-Core on one GPU (nothing to
# reduce), then check that NeMo-RL's Megatron training path reproduces them at
# TP=2 and at DP=2. A gradient that is not reduced across ranks fails the check.
# See tools/golden_values/README.md.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/../..")

GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if ((GPU_COUNT < 2)); then
    echo "SKIP: golden-value check needs two GPUs"
    exit 0
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

EXP_NAME=$(basename "$0" .sh)
EXP_DIR="${SCRIPT_DIR}/${EXP_NAME}"
CHECKPOINT="${EXP_DIR}/checkpoint"
GOLDEN="${EXP_DIR}/golden.json.gz"
rm -rf "${EXP_DIR}"
mkdir -p "${EXP_DIR}"
# NeMo-RL caches its HF->Megatron conversion by checkpoint path, and the
# checkpoint below is rebuilt at the same path on every run.
export NRL_MEGATRON_CHECKPOINT_DIR="${EXP_DIR}/megatron_checkpoints"
cd "${PROJECT_ROOT}"

# Config, remote code and tokenizer only; the 120B weights are never downloaded.
uv run --extra mcore torchrun --nproc-per-node 1 tools/golden_values/make_truncated_checkpoint.py \
    --hf-model nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16 \
    --out "${CHECKPOINT}" \
    --set 'llm_config.layers_block_type=["mamba","moe","attention","moe"]' \
    --set llm_config.n_routed_experts=8 \
    --set llm_config.num_experts_per_tok=2

uv run --extra mcore torchrun --nproc-per-node 1 tools/golden_values/record_mcore_goldens.py \
    --hf-model "${CHECKPOINT}" --out "${GOLDEN}" --global-batch-size 4 --seq-length 128

check() {
    uv run --extra mcore coverage run -a --data-file="${PROJECT_ROOT}/tests/.coverage" \
        --source="${PROJECT_ROOT}/nemo_rl" \
        tools/golden_values/check_nemo_rl_goldens.py \
        --golden "${GOLDEN}" --hf-model "${CHECKPOINT}" --gpus-per-node 2 "$@"
}
# DP=2 is the layout the Omni gradient-sync bug broke; default tolerances.
check --tp 1
# Under TP=2 with sequence parallelism, bf16 near-ties in MoE routing flip and
# moved individual experts' updates by 22-38% (4-5% overall) in three runs on
# 2026-10-20, so the weight tolerances are looser. Loss and grad norm keep their
# defaults.
check --tp 2 --tolerance weight_update=0.15 --tolerance weight_update_per_tensor=0.8
