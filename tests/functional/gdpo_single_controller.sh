#!/bin/bash
# Lightweight end-to-end coverage for multi-reward GDPO through the Single
# Controller rollout, DataPlane, advantage, and Megatron training path.

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")
git config --global --add safe.directory "$PROJECT_ROOT"

set -eou pipefail

EXP_NAME=$(basename "$0" .sh)
EXP_DIR="$SCRIPT_DIR/$EXP_NAME"
LOG_DIR="$EXP_DIR/logs"
JSON_METRICS="$EXP_DIR/metrics.json"
RUN_LOG="$EXP_DIR/run.log"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

rm -rf "$EXP_DIR" "$LOG_DIR"
mkdir -p "$EXP_DIR" "$LOG_DIR"

cd "$PROJECT_ROOT"
uv run --group test coverage run -a \
    --data-file="$PROJECT_ROOT/tests/.coverage" \
    --source="$PROJECT_ROOT/nemo_rl" \
    "$PROJECT_ROOT/examples/run_grpo_single_controller.py" \
    --config "$PROJECT_ROOT/examples/configs/gdpo_math_1B_megatron_single_controller.yaml" \
    policy.model_name=Qwen/Qwen3-0.6B \
    grpo.num_prompts_per_step=2 \
    grpo.num_generations_per_prompt=4 \
    grpo.seq_logprob_error_threshold=1000 \
    policy.train_global_batch_size=8 \
    policy.train_micro_batch_size=1 \
    cluster.gpus_per_node=2 \
    cluster.num_nodes=1 \
    grpo.max_num_steps=2 \
    logger.tensorboard_enabled=true \
    logger.log_dir="$LOG_DIR" \
    logger.wandb_enabled=false \
    logger.monitor_gpus=true \
    checkpointing.enabled=false \
    async_rl.sampler.max_lookahead_versions=0 \
    async_rl.min_groups_for_streaming_train=2 \
    async_rl.max_inflight_prompts=2 \
    async_rl.max_buffered_rollouts=2 \
    "$@" \
    2>&1 | tee "$RUN_LOG"

uv run tests/json_dump_tb_logs.py "$LOG_DIR" --output_path "$JSON_METRICS"

uv run tests/check_metrics.py "$JSON_METRICS" \
    'max(data["train/num_masked_seqs_by_logprob_error"]) == 0' \
    'max(data["train/max_seq_mult_prob_error"]) < 1000' \
    'max(data["train/gen_kl_error"]) < 0.002' \
    'min(data["train/probs_ratio_clamped_min"]) > 0.79' \
    'max(data["train/probs_ratio_clamped_min"]) < 1.21' \
    'min(data["train/probs_ratio_clamped_max"]) > 0.79' \
    'max(data["train/probs_ratio_clamped_max"]) < 1.21'
