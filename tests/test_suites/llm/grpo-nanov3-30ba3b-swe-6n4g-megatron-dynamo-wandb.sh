#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "${SCRIPT_DIR}/common.env"

# ===== BEGIN CONFIG =====
NUM_NODES=6
GPUS_PER_NODE=4
STEPS_PER_RUN=4
MAX_STEPS=4
NUM_RUNS=1
NUM_MINUTES=240
USES_SANDBOX=1
USE_GYM_CONTAINER=true
# ===== END CONFIG =====

exit_if_max_steps_reached

: "${TRAIN_PATH:?Set TRAIN_PATH to the SWE training JSONL}"
: "${VAL_PATH:?Set VAL_PATH to the SWE validation JSONL}"
: "${SIF_FORMATTERS:?Set SIF_FORMATTERS to a Hydra list of SWE image format strings}"

cd "${PROJECT_ROOT}"
uv run examples/nemo_gym/run_grpo_nemo_gym.py \
    --config "${CONFIG_PATH}" \
    grpo.max_num_steps="${MAX_STEPS}" \
    data.train.data_path="${TRAIN_PATH}" \
    data.validation.data_path="${VAL_PATH}" \
    env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.container_formatter="${SIF_FORMATTERS}" \
    logger.log_dir="${LOG_DIR}" \
    logger.wandb_enabled=true \
    logger.wandb.project=nemo-rl-dynamo-swe \
    logger.wandb.name="${EXP_NAME}" \
    "$@" \
    2>&1 | tee "${RUN_LOG}"

uv run tests/json_dump_tb_logs.py "${LOG_DIR}" --output_path "${JSON_METRICS}"
