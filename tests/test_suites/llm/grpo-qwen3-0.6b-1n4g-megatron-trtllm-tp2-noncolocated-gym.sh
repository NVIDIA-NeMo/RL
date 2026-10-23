#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source "$SCRIPT_DIR/common.env"

# ===== BEGIN CONFIG =====
NUM_NODES=1
GPUS_PER_NODE=4
STEPS_PER_RUN=6
MAX_STEPS=6
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))
NUM_MINUTES=90
# ===== END CONFIG =====

exit_if_max_steps_reached

cd "$PROJECT_ROOT"

DATA_DIR="$EXP_DIR/data"
mkdir -p "$DATA_DIR"
cd 3rdparty/Gym-workspace/Gym
if [[ ! -f env.yaml ]]; then
    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo "[ERROR] HF_TOKEN is not set"
        exit 1
    fi
    echo "hf_token: $HF_TOKEN" >> env.yaml
fi
uv run ng_prepare_data "+config_paths=[resources_servers/workplace_assistant/configs/workplace_assistant.yaml]" \
    +output_dirpath=data/workplace_assistant \
    +mode=train_preparation \
    +should_download=true \
    +data_source=huggingface
cd -

TRAIN_PATH="$DATA_DIR/workplace_assistant_train.jsonl"
VALIDATION_PATH="$DATA_DIR/workplace_assistant_validation.jsonl"
jq -c '.responses_create_params.tools |= (.[0:1])' \
    3rdparty/Gym-workspace/Gym/data/workplace_assistant/train.jsonl > "$TRAIN_PATH"
jq -c '.responses_create_params.tools |= (.[0:1])' \
    3rdparty/Gym-workspace/Gym/data/workplace_assistant/validation.jsonl > "$VALIDATION_PATH"

uv run examples/nemo_gym/run_grpo_nemo_gym.py \
    --config "$CONFIG_PATH" \
    grpo.max_num_steps="$MAX_STEPS" \
    logger.log_dir="$LOG_DIR" \
    logger.wandb_enabled=true \
    logger.wandb.project=nemo-rl \
    logger.wandb.name="$EXP_NAME" \
    logger.monitor_gpus=true \
    logger.tensorboard_enabled=true \
    checkpointing.enabled=true \
    checkpointing.checkpoint_dir="$CKPT_DIR" \
    data.train.data_path="$TRAIN_PATH" \
    data.validation.data_path="$VALIDATION_PATH" \
    "$@" \
    2>&1 | tee "$RUN_LOG"

uv run tests/json_dump_tb_logs.py "$LOG_DIR" --output_path "$JSON_METRICS"

LAST_STEP=$(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | max' "$JSON_METRICS")
if [[ "$LAST_STEP" -lt "$MAX_STEPS" ]]; then
    echo "[ERROR] Expected step $MAX_STEPS, reached $LAST_STEP"
    exit 1
fi

uv run tests/check_metrics.py "$JSON_METRICS" \
    'mean(data["train/reward"]) > 0.05' \
    'median(data["train/token_mult_prob_error"]) < 1.1' \
    'mean(data["train/gen_kl_error"]) < 0.02' \
    'mean(data["train/grad_norm"], 2, 0) > 0.1' \
    'mean(data["train/grad_norm"], 2, 0) < 30.0'

rm -rf "$CKPT_DIR"
