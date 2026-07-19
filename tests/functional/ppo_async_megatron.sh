#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/../..")
EXP_NAME=$(basename "$0" .sh)
EXP_DIR="${SCRIPT_DIR}/${EXP_NAME}"
LOG_DIR="${EXP_DIR}/logs"
METRICS="${EXP_DIR}/metrics.json"
RUN_LOG="${EXP_DIR}/run.log"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

rm -rf "${EXP_DIR}"
mkdir -p "${LOG_DIR}"

cd "${PROJECT_ROOT}"
uv run coverage run -a \
    --data-file="${PROJECT_ROOT}/tests/.coverage" \
    --source="${PROJECT_ROOT}/nemo_rl" \
    "${PROJECT_ROOT}/examples/run_ppo.py" \
    --config "${PROJECT_ROOT}/examples/configs/ppo_math_1B_megatron.yaml" \
    policy.model_name=Qwen/Qwen2.5-0.5B \
    value.model_name=Qwen/Qwen2.5-0.5B \
    ppo.num_prompts_per_step=2 \
    ppo.num_generations_per_prompt=4 \
    ppo.ppo_epochs=2 \
    ppo.policy_training_start_step=1 \
    ppo.max_num_steps=2 \
    ppo.val_at_start=false \
    ppo.val_period=0 \
    ppo.val_at_end=true \
    ppo.max_val_samples=8 \
    ppo.val_batch_size=8 \
    ppo.reward_scaling.enabled=false \
    ppo.reward_shaping.enabled=false \
    ppo.seq_logprob_error_threshold=1000 \
    ppo.async_ppo.enabled=true \
    ppo.async_ppo.max_trajectory_age_steps=1 \
    ppo.async_ppo.warmup_max_trajectory_age_steps=2 \
    policy.train_global_batch_size=4 \
    policy.logprob_batch_size=4 \
    policy.train_micro_batch_size=1 \
    policy.generation.colocated.enabled=false \
    policy.generation.colocated.resources.gpus_per_node=1 \
    policy.generation.colocated.resources.num_nodes=1 \
    policy.generation.vllm_cfg.async_engine=true \
    loss_fn.use_importance_sampling_correction=true \
    value.train_global_batch_size=4 \
    value.train_micro_batch_size=1 \
    cluster.gpus_per_node=2 \
    logger.tensorboard_enabled=true \
    logger.log_dir="${LOG_DIR}" \
    logger.wandb_enabled=false \
    logger.monitor_gpus=true \
    checkpointing.enabled=false \
    "$@" \
    2>&1 | tee "${RUN_LOG}"

grep -q "Separate PPO clusters initialized" "${RUN_LOG}"
grep -q "Using vllm in-flight weight update" "${RUN_LOG}"
grep -q "Updated generation window: version=0, lead=2, max_age=2" "${RUN_LOG}"
grep -q "Updated generation window: version=1, lead=1, max_age=2" "${RUN_LOG}"

uv run tests/json_dump_tb_logs.py "${LOG_DIR}" --output_path "${METRICS}"
uv run tests/check_metrics.py "${METRICS}" \
    'len(data["train/loss"]) == 1' \
    'len(data["train/critic/loss"]) == 2' \
    'min(data["train/probs_ratio_clamped_min"]) > 0.79' \
    'max(data["train/probs_ratio_clamped_min"]) < 1.21' \
    'min(data["train/probs_ratio_clamped_max"]) > 0.79' \
    'max(data["train/probs_ratio_clamped_max"]) < 1.29' \
    'max(data["train/token_mult_prob_error"]) < 1.05' \
    'max(data["train/avg_trajectory_age"]) <= 1' \
    'len(data["validation/accuracy"]) == 1'
