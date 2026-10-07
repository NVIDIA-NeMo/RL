#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath $SCRIPT_DIR/../..)
# Mark the current repo as safe, since wandb fetches metadata about the repo
git config --global --add safe.directory $PROJECT_ROOT

set -eou pipefail

# EXPECT=survival requires replacement, trainer weight transfer, and completion.
# EXPECT=bounded_failure requires the specific restart-budget error at refit.
EXPECT=${EXPECT:-survival}
case "$EXPECT" in
    survival) MAX_RESTARTS=1 ;;
    bounded_failure) MAX_RESTARTS=0 ;;
    *) echo "[chaos] EXPECT must be survival or bounded_failure; got '$EXPECT'"; exit 1 ;;
esac
MAX_STEPS=12

EXP_NAME=$(basename $0 .sh)-$EXPECT
EXP_DIR=$SCRIPT_DIR/$EXP_NAME
LOG_DIR=$EXP_DIR/logs
JSON_METRICS=$EXP_DIR/metrics.json
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

rm -rf $EXP_DIR $LOG_DIR
mkdir -p $EXP_DIR $LOG_DIR

cd "$PROJECT_ROOT"
uv run --no-sync python tests/functional/_sglang_grpo_chaos.py \
    --expect "$EXPECT" --exp-dir "$EXP_DIR" --max-steps "$MAX_STEPS" \
    uv run --no-sync python "$PROJECT_ROOT/examples/run_grpo.py" \
    --config "$PROJECT_ROOT/examples/configs/grpo_math_1B_sglang.yaml" \
    policy.model_name=Qwen/Qwen3-0.6B \
    data.train.dataset_name=GSM8K \
    +data.train.subset=main \
    +data.train.split=train \
    +data.train.extract_answer=true \
    data.train.split_validation_size=0 \
    '~data.train.seed' \
    'policy.tokenizer.chat_template_kwargs={enable_thinking:false}' \
    grpo.num_prompts_per_step=4 \
    grpo.num_generations_per_prompt=4 \
    policy.train_global_batch_size=16 \
    policy.train_micro_batch_size=1 \
    cluster.num_nodes=1 \
    cluster.gpus_per_node=2 \
    policy.generation.colocated.enabled=true \
    policy.generation.use_async_rollouts=false \
    grpo.async_grpo.enabled=false \
    policy.generation.sglang_cfg.tp_size=1 \
    policy.generation.sglang_cfg.sglang_server_config.num_gpus=2 \
    policy.generation.sglang_cfg.sglang_server_config.num_gpus_per_engine=1 \
    policy.generation.sglang_cfg.disable_cuda_graph=true \
    policy.generation.sglang_cfg.mem_fraction_static=0.3 \
    policy.generation.sglang_cfg.sglang_fault_tolerance_config.use_fault_tolerance=true \
    policy.generation.sglang_cfg.sglang_fault_tolerance_config.rollout_health_check_interval=2 \
    policy.generation.sglang_cfg.sglang_fault_tolerance_config.rollout_health_check_timeout=10 \
    policy.generation.sglang_cfg.sglang_fault_tolerance_config.rollout_health_check_first_wait=0 \
    "policy.generation.sglang_cfg.sglang_fault_tolerance_config.rollout_max_restart_attempts=$MAX_RESTARTS" \
    "grpo.max_num_steps=$MAX_STEPS" \
    grpo.val_period=0 \
    grpo.val_at_start=false \
    grpo.val_at_end=false \
    logger.tensorboard_enabled=true \
    "logger.log_dir=$LOG_DIR" \
    logger.wandb_enabled=false \
    logger.monitor_gpus=false \
    checkpointing.enabled=false \
    "$@" \
    2>&1 | tee "$EXP_DIR/harness.log"

if [[ "$EXPECT" == survival ]]; then
    REPLACED_AT=$(jq .completed_step "$EXP_DIR/replacement.json")
    uv run tests/json_dump_tb_logs.py "$LOG_DIR" --output_path "$JSON_METRICS"
    uv run tests/check_metrics.py "$JSON_METRICS" \
        'max(data["train/token_mult_prob_error"]) < 1.05' \
        "mean(data['train/grad_norm'], range_start=$((REPLACED_AT + 1))) > 0" \
        "mean(data['train/global_valid_toks'], range_start=$((REPLACED_AT + 1))) > 0"
fi
