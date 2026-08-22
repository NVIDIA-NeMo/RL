#!/bin/bash
# SingleController counterpart of tests/functional/ppo_async_megatron.sh:
# same model, batch shape, critic warmup and checkpoint/restore coverage, run
# through examples/run_grpo_single_controller.py instead of run_ppo.py.
#
# Three settings could not carry over, and each is forced by what SC is:
#   - ppo.ppo_epochs=2 has no SC equivalent: SC does exactly one optimizer step
#     per RL step for both the policy and the critic.
#   - train_global_batch_size is 8, not 4: SC requires
#     num_prompts_per_step * num_generations_per_prompt == the global batch, so
#     one RL step maps to one optimizer step.
#   - No validation: SC has no validation loop yet.

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
    "${PROJECT_ROOT}/examples/run_grpo_single_controller.py"
    --config "${PROJECT_ROOT}/examples/configs/ppo_math_1B_megatron_single_controller.yaml"
    policy.model_name=Qwen/Qwen2.5-0.5B
    value.model_name=Qwen/Qwen2.5-0.5B
    grpo.num_prompts_per_step=2
    grpo.num_generations_per_prompt=4
    grpo.max_num_epochs=1000
    grpo.val_at_start=false
    grpo.val_period=0
    grpo.val_at_end=false
    grpo.reward_scaling.enabled=false
    grpo.reward_shaping.enabled=false
    grpo.seq_logprob_error_threshold=1000
    ppo.policy_training_start_step=1
    policy.train_global_batch_size=8
    policy.logprob_batch_size=4
    policy.train_micro_batch_size=1
    +policy.megatron_cfg.scheduler.override_opt_param_scheduler=true
    policy.generation.colocated.enabled=false
    policy.generation.colocated.resources.gpus_per_node=1
    policy.generation.colocated.resources.num_nodes=1
    policy.generation.vllm_cfg.async_engine=true
    loss_fn.use_importance_sampling_correction=true
    value.train_global_batch_size=8
    value.train_micro_batch_size=1
    +value.megatron_cfg.scheduler.override_opt_param_scheduler=true
    data.use_multiple_dataloader=false
    data_plane.enabled=true
    data_plane.impl=transfer_queue
    data_plane.backend=simple
    async_rl.sampler.name=in_order
    async_rl.sampler.max_lookahead_versions=1
    async_rl.min_groups_for_streaming_train=2
    async_rl.max_inflight_prompts=4
    async_rl.max_buffered_rollouts=4
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
    grpo.max_num_steps=2 \
    logger.log_dir="${EXP_DIR}/logs_run1" \
    "$@" \
    2>&1 | tee "${EXP_DIR}/run1.log"

grep -q "Initializing value model for GAE" "${EXP_DIR}/run1.log"
grep -q "Using GAE advantage estimator" "${EXP_DIR}/run1.log"
grep -q "weight_sync=CollectiveWeightSynchronizer" "${EXP_DIR}/run1.log"
# policy_training_start_step=1, so step 0 trains the critic alone and must not
# transfer weights; step 1 must.
test "$(grep -c "Critic warmup: skipping policy weight transfer" "${EXP_DIR}/run1.log")" -eq 1
test -f "${CKPT_DIR}/step_1/replay_buffer.pt"
test -f "${CKPT_DIR}/step_2/replay_buffer.pt"
test -d "${CKPT_DIR}/step_1/value/weights"

"${TRAIN_CMD[@]}" \
    grpo.max_num_steps=4 \
    logger.log_dir="${EXP_DIR}/logs_run2" \
    "$@" \
    2>&1 | tee "${EXP_DIR}/run2.log"

grep -q "Restoring replay buffer from checkpoint" "${EXP_DIR}/run2.log"
grep -qF "replay group(s) from checkpoint" "${EXP_DIR}/run2.log"
# Warmup is behind us on the resumed run, so the policy trains every step.
assert_no_warmup=$(grep -c "Critic warmup: skipping policy weight transfer" "${EXP_DIR}/run2.log" || true)
test "${assert_no_warmup}" -eq 0
test -d "${CKPT_DIR}/step_4/policy/weights"
test -d "${CKPT_DIR}/step_4/value/weights"

# run1 trains the policy on 1 of its 2 steps (step 0 is critic warmup); run2
# resumes past the warmup and trains it on both.
for run_spec in "run1 1" "run2 2"; do
    read -r run expected_policy_steps <<< "${run_spec}"
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
        'min(data["train/critic/loss"]) >= 0'
done
