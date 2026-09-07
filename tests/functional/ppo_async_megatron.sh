#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/../..")
EXP_NAME=$(basename "$0" .sh)
EXP_DIR="${SCRIPT_DIR}/${EXP_NAME}"
CKPT_DIR="${EXP_DIR}/checkpoints"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

rm -rf "${EXP_DIR}"
mkdir -p "${EXP_DIR}"

TRAIN_CMD=(
    uv run coverage run -a
    --data-file="${PROJECT_ROOT}/tests/.coverage"
    --source="${PROJECT_ROOT}/nemo_rl"
    "${PROJECT_ROOT}/examples/run_ppo.py"
    --config "${PROJECT_ROOT}/examples/configs/ppo_math_1B_megatron.yaml"
    policy.model_name=Qwen/Qwen2.5-0.5B
    value.model_name=Qwen/Qwen2.5-0.5B
    ppo.num_prompts_per_step=2
    ppo.num_generations_per_prompt=4
    ppo.ppo_epochs=2
    ppo.max_num_epochs=-1
    ppo.policy_training_start_step=1
    ppo.val_at_start=false
    ppo.val_period=0
    ppo.val_at_end=true
    ppo.max_val_samples=8
    ppo.val_batch_size=8
    ppo.reward_scaling.enabled=false
    ppo.reward_shaping.enabled=false
    ppo.seq_logprob_error_threshold=1000
    ppo.async_ppo.enabled=true
    ppo.async_ppo.max_trajectory_age_steps=1
    ppo.async_ppo.warmup_generation_lead_steps=2
    policy.train_global_batch_size=4
    policy.logprob_batch_size=4
    policy.train_micro_batch_size=1
    +policy.megatron_cfg.scheduler.override_opt_param_scheduler=true
    policy.generation.colocated.enabled=false
    policy.generation.colocated.resources.gpus_per_node=1
    policy.generation.colocated.resources.num_nodes=1
    policy.generation.vllm_cfg.async_engine=true
    loss_fn.use_importance_sampling_correction=true
    value.train_global_batch_size=4
    value.train_micro_batch_size=1
    data.use_multiple_dataloader=false
    +value.megatron_cfg.scheduler.override_opt_param_scheduler=true
    cluster.gpus_per_node=2
    logger.tensorboard_enabled=true
    logger.wandb_enabled=false
    logger.monitor_gpus=true
    checkpointing.enabled=true
    checkpointing.checkpoint_dir="${CKPT_DIR}"
    checkpointing.metric_name=null
    checkpointing.save_period=1
)

cd "${PROJECT_ROOT}"

"${TRAIN_CMD[@]}" \
    ppo.max_num_steps=2 \
    logger.log_dir="${EXP_DIR}/logs_run1" \
    "$@" \
    2>&1 | tee "${EXP_DIR}/run1.log"

grep -q "Separate PPO clusters initialized" "${EXP_DIR}/run1.log"
grep -q "Updated generation window: version=0, lead=2, max_age=2" "${EXP_DIR}/run1.log"
grep -q "Updated generation window: version=1, lead=1, max_age=2" "${EXP_DIR}/run1.log"
test "$(grep -c "Value epoch 2/2" "${EXP_DIR}/run1.log")" -eq 2
test -f "${CKPT_DIR}/step_1/replay_buffer.pt"
test -f "${CKPT_DIR}/step_2/replay_buffer.pt"

"${TRAIN_CMD[@]}" \
    ppo.max_num_steps=4 \
    logger.log_dir="${EXP_DIR}/logs_run2" \
    "$@" \
    2>&1 | tee "${EXP_DIR}/run2.log"

grep -q "Restoring replay buffer from checkpoint" "${EXP_DIR}/run2.log"
grep -q "ReplayBuffer restored:" "${EXP_DIR}/run2.log"
grep -q "Updated generation window: version=3, lead=1, max_age=1" "${EXP_DIR}/run2.log"
test "$(grep -c "Value epoch 2/2" "${EXP_DIR}/run2.log")" -eq 2
test -d "${CKPT_DIR}/step_4/policy/weights"
test -d "${CKPT_DIR}/step_4/value/weights"

for run_spec in "run1 1 1" "run2 2 2"; do
    read -r run expected_policy_steps expected_max_age <<< "${run_spec}"
    metrics="${EXP_DIR}/metrics_${run}.json"
    uv run tests/json_dump_tb_logs.py "${EXP_DIR}/logs_${run}" \
        --output_path "${metrics}"
    uv run tests/check_metrics.py "${metrics}" \
        'len(data["train/reward"]) == 2' \
        "len(data[\"train/loss\"]) == ${expected_policy_steps}" \
        'len(data["train/critic/loss"]) == 2' \
        'min(data["train/probs_ratio_clamped_min"]) > 0.79' \
        'max(data["train/probs_ratio_clamped_min"]) < 1.21' \
        'min(data["train/probs_ratio_clamped_max"]) > 0.79' \
        'max(data["train/probs_ratio_clamped_max"]) < 1.29' \
        'max(data["train/token_mult_prob_error"]) < 1.05' \
        'max(data["train/critic/loss"]) < 6.0' \
        'min(data["train/critic/loss"]) >= 0' \
        "max(data[\"train/avg_trajectory_age\"]) <= ${expected_max_age}" \
        'len(data["validation/accuracy"]) == 1'
done
