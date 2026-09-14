#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/../..")

GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if (( GPU_COUNT < 2 )); then
    echo "SKIP: Qwen Energon SFTv2 smoke requires at least two GPUs"
    exit 0
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

EXP_NAME=$(basename "$0" .sh)
EXP_DIR="${SCRIPT_DIR}/${EXP_NAME}"
DATA_DIR="${EXP_DIR}/clevr-energon"
LOG_DIR="${EXP_DIR}/logs"
CKPT_DIR="${EXP_DIR}/checkpoints"
JSON_METRICS="${EXP_DIR}/metrics.json"
RUN_LOG="${EXP_DIR}/run.log"
MCORE_VENV_ROOT="${NEMO_RL_VENV_DIR:-${PROJECT_ROOT}/venvs}"
MCORE_PYTHON="${MCORE_VENV_ROOT}/nemo_rl.data.energon.sft_worker.SFTMegatronPolicyWorker/bin/python"

rm -rf "${EXP_DIR}"
mkdir -p "${EXP_DIR}"
trap 'rm -rf "${DATA_DIR}" "${CKPT_DIR}"' EXIT

cd "${PROJECT_ROOT}"

uv run --no-sync nemo_rl/utils/prefetch_venvs.py SFTMegatronPolicyWorker
"${MCORE_PYTHON}" nemo_rl/data/energon/scripts/prepare_energon_dataset.py \
    --output-dir "${DATA_DIR}" \
    --splits train \
    --max-samples 16 \
    --max-samples-per-shard 16 \
    --num-workers 1 \
    --download-workers 1 \
    --image-workers 1

uv run --no-sync python examples/run_sft_v2.py \
    --config examples/configs/recipes/vlm/vlm_sft-qwen2.5-vl-3b-instruct-clevr-1n2g-megatrontp1-energon.v1.yaml \
    policy.train_global_batch_size=2 \
    policy.train_micro_batch_size=1 \
    sft.max_num_steps=3 \
    data.train.path="${DATA_DIR}" \
    data.train.virtual_epoch_length=6 \
    data.energon.num_workers=0 \
    data.energon.shuffle_buffer_size=16 \
    logger.tensorboard_enabled=true \
    logger.log_dir="${LOG_DIR}" \
    logger.wandb_enabled=false \
    logger.monitor_gpus=false \
    checkpointing.enabled=true \
    checkpointing.save_period=3 \
    checkpointing.save_optimizer=false \
    checkpointing.checkpoint_dir="${CKPT_DIR}" \
    "$@" 2>&1 | tee "${RUN_LOG}"

uv run --no-sync tests/json_dump_tb_logs.py \
    "${LOG_DIR}" --output_path "${JSON_METRICS}"
uv run --no-sync tests/check_metrics.py "${JSON_METRICS}" \
    'all_finite(data["loss"])'
