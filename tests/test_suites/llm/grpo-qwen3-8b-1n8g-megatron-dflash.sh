#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source "$SCRIPT_DIR/common.env"

# ===== BEGIN CONFIG =====
NUM_NODES=1
GPUS_PER_NODE=8
STEPS_PER_RUN=1
MAX_STEPS=1
NUM_RUNS=1
NUM_MINUTES=30
# ===== END CONFIG =====

exit_if_max_steps_reached

cd "$PROJECT_ROOT"
uv run examples/run_grpo.py \
    --config "$CONFIG_PATH" \
    data_plane.enabled=true \
    policy.megatron_cfg.tensor_model_parallel_size=2 \
    policy.megatron_cfg.pipeline_model_parallel_size=1 \
    policy.megatron_cfg.context_parallel_size=4 \
    policy.megatron_cfg.sequence_parallel=true \
    policy.sequence_packing.enabled=true \
    policy.make_sequence_length_divisible_by=16 \
    +policy.draft.update_probe_enabled=true \
    grpo.max_num_steps=$MAX_STEPS \
    grpo.val_period=0 \
    logger.log_dir="$LOG_DIR" \
    logger.wandb_enabled=false \
    logger.monitor_gpus=true \
    logger.tensorboard_enabled=true \
    checkpointing.enabled=false \
    checkpointing.checkpoint_dir="$CKPT_DIR" \
    "$@" \
    2>&1 | tee "$RUN_LOG"

grep -q "Draft Loss:" "$RUN_LOG"
grep -q "draft_update_probe=complete" "$RUN_LOG"

uv run tests/json_dump_tb_logs.py "$LOG_DIR" --output_path "$JSON_METRICS"

if [[ $(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | max' "$JSON_METRICS") -ge $MAX_STEPS ]]; then
    uv run tests/check_metrics.py "$JSON_METRICS" \
        'min(data["train/draft_loss"]) > 0'
fi
