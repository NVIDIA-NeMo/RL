#!/bin/bash
# SingleController counterpart of tests/functional/distillation_megatron.sh:
# same model pair and step count, run through
# examples/run_grpo_single_controller.py instead of run_distillation.py.
#
# Three settings could not carry over, and each is forced by what SC is:
#   - No validation: SC has no validation loop yet, so val_period=0.
#   - num_prompts_per_step * num_generations_per_prompt must equal the global
#     batch, so one RL step maps to one optimizer step.
#   - Generation is non-colocated, so it takes one of the two GPUs. The teacher
#     does not take a third: it is a second worker group on the *training*
#     GPUs, resident only for its own forward.

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
    --config "${PROJECT_ROOT}/examples/configs/distillation_math_1B_megatron_single_controller.yaml"
    policy.model_name=Qwen/Qwen3-0.6B-Base
    teacher.model_name=Qwen/Qwen3-0.6B
    distillation.num_prompts_per_step=4
    distillation.num_generations_per_prompt=2
    distillation.max_num_epochs=1000
    distillation.topk_logits_k=16
    policy.train_global_batch_size=8
    policy.logprob_batch_size=4
    policy.train_micro_batch_size=1
    policy.max_total_sequence_length=2048
    policy.megatron_cfg.tensor_model_parallel_size=1
    policy.megatron_cfg.pipeline_model_parallel_size=1
    policy.megatron_cfg.context_parallel_size=1
    +policy.megatron_cfg.scheduler.override_opt_param_scheduler=true
    teacher.megatron_cfg.tensor_model_parallel_size=1
    teacher.megatron_cfg.pipeline_model_parallel_size=1
    teacher.megatron_cfg.context_parallel_size=1
    policy.generation.colocated.enabled=false
    policy.generation.colocated.resources.gpus_per_node=1
    policy.generation.colocated.resources.num_nodes=1
    policy.generation.vllm_cfg.async_engine=true
    data.train.dataset_name=OpenMathInstruct-2
    ++data.train.split_validation_size=0.05
    data.validation=null
    data.use_multiple_dataloader=false
    loss_fn.zero_outside_topk=false
    async_rl.sampler.name=in_order
    async_rl.sampler.max_lookahead_versions=1
    async_rl.min_groups_for_streaming_train=4
    async_rl.max_inflight_prompts=6
    async_rl.max_buffered_rollouts=6
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
    distillation.max_num_steps=2 \
    logger.log_dir="${EXP_DIR}/logs_run1" \
    "$@" \
    2>&1 | tee "${EXP_DIR}/run1.log"

# The teacher is built, so setup reports it -- and it is built as a second
# worker group on the training GPUs, not a third GPU.
grep -q "Teacher init:" "${EXP_DIR}/run1.log"
# No critic on this path: distillation and ppo blocks are mutually exclusive.
test "$(grep -c "Value init:" "${EXP_DIR}/run1.log")" -eq 0
test -f "${CKPT_DIR}/step_1/replay_buffer.pt"
test -f "${CKPT_DIR}/step_2/replay_buffer.pt"

"${TRAIN_CMD[@]}" \
    distillation.max_num_steps=4 \
    logger.log_dir="${EXP_DIR}/logs_run2" \
    "$@" \
    2>&1 | tee "${EXP_DIR}/run2.log"

grep -q "Restoring replay buffer from checkpoint" "${EXP_DIR}/run2.log"
test -d "${CKPT_DIR}/step_4/policy/weights"

for run_spec in "run1 2" "run2 4"; do
    read -r run expected_steps <<< "${run_spec}"
    metrics="${EXP_DIR}/metrics_${run}.json"
    uv run tests/json_dump_tb_logs.py "${EXP_DIR}/logs_${run}" \
        --output_path "${metrics}"
    uv run tests/check_metrics.py "${metrics}" \
        "len(data[\"train/loss\"]) == ${expected_steps}" \
        'min(data["train/loss"]) >= 0' \
        'max(data["train/loss"]) < 20.0' \
        'len(data["timing/train/teacher_logprob_inference"]) > 0'
done
