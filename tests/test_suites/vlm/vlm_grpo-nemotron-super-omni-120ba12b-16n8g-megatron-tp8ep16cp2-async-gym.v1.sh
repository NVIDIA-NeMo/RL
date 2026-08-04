#!/bin/bash
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "$SCRIPT_DIR/common.env"

# ===== BEGIN CONFIG =====
NUM_NODES=16
GPUS_PER_NODE=8
STEPS_PER_RUN=3
MAX_STEPS=3
NUM_RUNS=$(((MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN))
NUM_MINUTES=180
# ===== END CONFIG =====

exit_if_max_steps_reached

MODEL_PATH="${NEMOTRON_SUPER_OMNI_MODEL_PATH:-/path/to/nemotron-super-omni-hf-checkpoint}"
TRAIN_PATH="${NEMOTRON_SUPER_OMNI_GYM_DATA_PATH:-/path/to/super-omni-gym-train.jsonl}"
CHAT_TEMPLATE="${NEMOTRON_SUPER_OMNI_CHAT_TEMPLATE:-${MODEL_PATH}/chat_template.jinja}"

export NRL_REFIT_BUFFER_MEMORY_RATIO=0.006
export NRL_REFIT_NUM_BUFFERS=1
export NEMO_RL_VLLM_PRECOMPUTED_IMG_SIZES=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export NRL_FORCE_REBUILD_VENVS="${NRL_FORCE_REBUILD_VENVS:-true}"

cd "$PROJECT_ROOT"

MEGATRON_BRIDGE_SRC="$PROJECT_ROOT/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src"
MEGATRON_LM_SRC="$PROJECT_ROOT/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM"
export PYTHONPATH="$PROJECT_ROOT:$MEGATRON_BRIDGE_SRC:$MEGATRON_LM_SRC${PYTHONPATH:+:$PYTHONPATH}"

GYM_ROOT="$PROJECT_ROOT/3rdparty/Gym-workspace/Gym"
EXTENSION_ROOT="$PROJECT_ROOT/examples/nemo_gym/nemotron-3-super/gym_extensions/resources_servers"
CREATED_EXTENSION_LINKS=()
cleanup_extension_links() {
    for link in "${CREATED_EXTENSION_LINKS[@]}"; do
        if [[ -L "$link" ]]; then
            unlink "$link"
        fi
    done
}
trap cleanup_extension_links EXIT

for extension in gui_coordinate string_match; do
    destination="$GYM_ROOT/resources_servers/$extension"
    if [[ ! -e "$destination" ]]; then
        ln -s "$EXTENSION_ROOT/$extension" "$destination"
        CREATED_EXTENSION_LINKS+=("$destination")
    fi
done

uv run --no-sync examples/nemo_gym/run_multimodal_grpo_nemo_gym.py \
    --config "$CONFIG_PATH" \
    grpo.max_num_steps=$MAX_STEPS \
    policy.model_name="$MODEL_PATH" \
    policy.tokenizer.chat_template="$CHAT_TEMPLATE" \
    policy.generation.vllm_cfg.http_server_serving_chat_kwargs.chat_template="$CHAT_TEMPLATE" \
    data.train.data_path="$TRAIN_PATH" \
    data.validation.data_path="$TRAIN_PATH" \
    logger.log_dir="$LOG_DIR" \
    logger.wandb_enabled=True \
    logger.wandb.project=nemo-rl \
    logger.wandb.name="$EXP_NAME" \
    logger.monitor_gpus=True \
    logger.tensorboard_enabled=True \
    checkpointing.enabled=True \
    checkpointing.checkpoint_dir="$CKPT_DIR" \
    "$@" \
    2>&1 | tee "$RUN_LOG"

uv run tests/json_dump_tb_logs.py "$LOG_DIR" --output_path "$JSON_METRICS"

if [[ $(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | max' "$JSON_METRICS") -ge $MAX_STEPS ]]; then
    uv run tests/check_metrics.py "$JSON_METRICS" \
        'median(data["train/token_mult_prob_error"]) < 1.1' \
        "data['train/token_mult_prob_error']['$MAX_STEPS'] < 1.1"

    rm -rf "$CKPT_DIR"
fi
