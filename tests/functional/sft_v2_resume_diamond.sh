#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/../..")
GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if (( GPU_COUNT < 2 )); then
    echo "SKIP: Qwen Energon SFTv2 resume requires at least two GPUs"
    exit 0
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

EXP_NAME=$(basename "$0" .sh)
RECIPE="examples/configs/recipes/vlm/vlm_sft-qwen2.5-vl-3b-instruct-clevr-1n2g-megatrontp1-energon.v1.yaml"
if [[ "${SFT_V2_RESUME_PACKED:-0}" == "1" ]]; then
    EXP_NAME="${EXP_NAME}_packed"
    RECIPE="${RECIPE%.yaml}.packing.yaml"
fi
EXP_DIR="${SCRIPT_DIR}/${EXP_NAME}"
DATA_DIR="${EXP_DIR}/clevr-energon"
LOG_DIR="${EXP_DIR}/logs"
CKPT_DIR="${EXP_DIR}/checkpoints"
MCORE_VENV_ROOT="${NEMO_RL_VENV_DIR:-${PROJECT_ROOT}/venvs}"
MCORE_PYTHON="${MCORE_VENV_ROOT}/nemo_rl.data.energon.sft_worker.SFTMegatronPolicyWorker/bin/python"

rm -rf "${EXP_DIR}"
mkdir -p "${EXP_DIR}"
trap 'rm -rf "${DATA_DIR}" "${CKPT_DIR}"' EXIT
cd "${PROJECT_ROOT}"

# Reuse a materialized worker environment when the container already provides it.
if [[ ! -x "${MCORE_PYTHON}" ]]; then
    uv run --no-sync nemo_rl/utils/prefetch_venvs.py SFTMegatronPolicyWorker
fi
"${MCORE_PYTHON}" nemo_rl/data/energon/scripts/prepare_energon_dataset.py \
    --output-dir "${DATA_DIR}" --splits train \
    --max-samples 16 --max-samples-per-shard 16 \
    --num-workers 1 --download-workers 1 --image-workers 1

train_run() {
    local phase="$1"
    shift
    uv run --no-sync python examples/run_sft_v2.py \
        --config "${RECIPE}" \
        policy.train_global_batch_size=2 policy.train_micro_batch_size=1 \
        sft.max_num_steps=2 \
        data.train.path="${DATA_DIR}" data.train.virtual_epoch_length=6 \
        data.energon.num_workers=0 data.energon.shuffle_buffer_size=16 \
        logger.tensorboard_enabled=true logger.log_dir="${LOG_DIR}/${phase}" \
        logger.wandb_enabled=false logger.monitor_gpus=false \
        checkpointing.enabled=false checkpointing.save_optimizer=true \
        checkpointing.checkpoint_dir="${CKPT_DIR}" \
        "$@" 2>&1 | tee "${EXP_DIR}/${phase}.log"
    uv run --no-sync tests/json_dump_tb_logs.py "${LOG_DIR}/${phase}" \
        --output_path "${EXP_DIR}/${phase}.json" --error-on-conflicts
}

train_run baseline "$@"
# Keep the same two-step scheduler budget; the timeout saves and exits after step 1.
train_run interrupted "$@" checkpointing.enabled=true \
    checkpointing.checkpoint_must_save_by=0:0:0:1
train_run resumed "$@" checkpointing.enabled=true

uv run --no-sync python - "${EXP_DIR}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
runs = {
    name: json.loads((root / f"{name}.json").read_text())
    for name in ("baseline", "interrupted", "resumed")
}
for metric in ("loss", "global_valid_toks"):
    assert set(runs["baseline"][metric]) == {"1", "2"}, (
        metric,
        runs["baseline"][metric],
    )
    assert set(runs["interrupted"][metric]) == {"1"}, (
        metric,
        runs["interrupted"][metric],
    )
    # A restart from the beginning would emit steps 1 and 2, so reject it.
    assert set(runs["resumed"][metric]) == {"2"}, (metric, runs["resumed"][metric])
    for phase, step in (("interrupted", "1"), ("resumed", "2")):
        expected = runs["baseline"][metric][step]
        actual = runs[phase][metric][step]
        assert np.isfinite(expected) and np.isfinite(actual), (metric, phase, actual)
        if metric == "global_valid_toks":
            assert actual == expected and expected > 0, (
                metric,
                phase,
                actual,
                expected,
            )
        else:
            # Match the SFTv1 diamond tolerance for BF16 checkpoint restore.
            assert np.isclose(actual, expected, rtol=0.05, atol=1e-6), (
                metric,
                phase,
                actual,
                expected,
            )
        print(f"PASS: {phase} step {step} {metric}: {actual} (baseline {expected})")
print("PASS: SFTv2 checkpoint restore preserves loss and the Energon stream.")
PY
