#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source "$SCRIPT_DIR/common.env"

# Parameterized because the Super Omni checkpoint and image data are external.
# ===== BEGIN CONFIG =====
NUM_NODES="${NUM_NODES:-10}"
GPUS_PER_NODE=8
STEPS_PER_RUN=3
MAX_STEPS=3
NUM_RUNS=1
NUM_MINUTES="${NUM_MINUTES:-840}"
# ===== END CONFIG =====

exit_if_max_steps_reached

: "${MODEL_PATH:?MODEL_PATH must point at the Super Omni HF checkpoint}"
: "${TRAIN_PATH:?TRAIN_PATH must point at circle-count Gym JSONL data}"
VAL_PATH="${VAL_PATH:-$TRAIN_PATH}"

teacher_args=()
if [[ -n "${TEACHER_MODEL_PATH:-}" ]]; then
    teacher_args+=(
        "on_policy_distillation.teacher_model_by_agent_name.circle_count_simple_agent=${TEACHER_MODEL_PATH}"
    )
fi

cd "$PROJECT_ROOT"
uv run examples/nemo_gym/run_grpo_nemo_gym.py \
    --config "$CONFIG_PATH" \
    grpo.max_num_steps=$MAX_STEPS \
    policy.model_name="$MODEL_PATH" \
    policy.tokenizer.chat_template="$MODEL_PATH/chat_template.jinja" \
    policy.generation.vllm_cfg.http_server_serving_chat_kwargs.chat_template="$MODEL_PATH/chat_template.jinja" \
    data.train.data_path="$TRAIN_PATH" \
    data.validation.data_path="$VAL_PATH" \
    logger.log_dir="$LOG_DIR" \
    logger.wandb_enabled=True \
    logger.wandb.project=nemo-rl \
    logger.wandb.name="$EXP_NAME" \
    logger.monitor_gpus=True \
    logger.tensorboard_enabled=True \
    checkpointing.enabled=False \
    "${teacher_args[@]}" \
    "$@" \
    2>&1 | tee "$RUN_LOG"

uv run tests/json_dump_tb_logs.py "$LOG_DIR" --output_path "$JSON_METRICS"

if [[ $(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | max' "$JSON_METRICS") -ge $MAX_STEPS ]]; then
    uv run tests/check_metrics.py "$JSON_METRICS" \
        'max(data["train/loss"]) < 1000000.0' \
        'min(data["train/loss"]) > -1000000.0' \
        'max(data["train/grad_norm"]) < 1000000.0' \
        'min(data["train/grad_norm"]) >= 0.0'
fi
