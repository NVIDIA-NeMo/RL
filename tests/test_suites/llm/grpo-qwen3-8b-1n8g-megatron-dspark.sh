#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source "$SCRIPT_DIR/common.env"

# ===== BEGIN CONFIG =====
NUM_NODES=1
GPUS_PER_NODE=4
STEPS_PER_RUN=2
MAX_STEPS=2
NUM_RUNS=1
NUM_MINUTES=45
# ===== END CONFIG =====

cd "$PROJECT_ROOT"
uv run examples/run_grpo.py \
    --config "$CONFIG_PATH" \
    data_plane.enabled=true \
    cluster.gpus_per_node=$GPUS_PER_NODE \
    policy.megatron_cfg.tensor_model_parallel_size=2 \
    policy.megatron_cfg.pipeline_model_parallel_size=1 \
    policy.megatron_cfg.context_parallel_size=2 \
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
    checkpointing.enabled=true \
    checkpointing.save_period=1 \
    checkpointing.save_optimizer=false \
    checkpointing.checkpoint_dir="$CKPT_DIR" \
    "$@" \
    2>&1 | tee "$RUN_LOG"

grep -q "Draft Loss:" "$RUN_LOG"
grep -q "draft_update_probe=complete" "$RUN_LOG"
grep -q "draft_refit_manifest=draft_count=" "$RUN_LOG"
awk '
    /draft_update_probe=complete/ { updated = 1; next }
    updated && /draft_refit_manifest=draft_count=/ { refitted = 1 }
    END { exit !refitted }
' "$RUN_LOG"
grep -q "Saving checkpoint for step 1..." "$RUN_LOG"
grep -q "Saving checkpoint for step 2..." "$RUN_LOG"
test -f "$CKPT_DIR/step_1/training_info.json"
test -f "$CKPT_DIR/step_1/config.yaml"
test -d "$CKPT_DIR/step_1/policy/weights"
test -f "$CKPT_DIR/step_2/training_info.json"
test -f "$CKPT_DIR/step_2/config.yaml"
test -d "$CKPT_DIR/step_2/policy/weights"

uv run tests/json_dump_tb_logs.py "$LOG_DIR" --output_path "$JSON_METRICS"
uv run tests/check_metrics.py "$JSON_METRICS" \
    'min(data["train/draft_loss"]) > 0'

rm -rf "$CKPT_DIR"
