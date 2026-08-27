#!/bin/bash
# Twenty-step GB200 smoke test for TE NVFP4 training and per-token vLLM refits.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source "$SCRIPT_DIR/common.env"

export NRL_ROUTER_REPLAY_VALIDATE=1

# ===== BEGIN CONFIG =====
NUM_NODES=4
GPUS_PER_NODE=4
STEPS_PER_RUN=20
MAX_STEPS=20
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))
NUM_MINUTES=90
SNAPSHOT_MEGATRON_BRIDGE=1
# ===== END CONFIG =====

exit_if_max_steps_reached

cd "$PROJECT_ROOT"
uv run --no-sync examples/run_grpo.py \
    --config "$CONFIG_PATH" \
    grpo.max_num_steps=$MAX_STEPS \
    logger.log_dir="$LOG_DIR" \
    logger.wandb_enabled=True \
    logger.wandb.project=nemo-rl \
    logger.wandb.name="$EXP_NAME" \
    logger.monitor_gpus=True \
    logger.tensorboard_enabled=True \
    checkpointing.enabled=False \
    checkpointing.checkpoint_dir="$CKPT_DIR" \
    "$@" \
    2>&1 | tee "$RUN_LOG"

uv run --no-sync tests/json_dump_tb_logs.py "$LOG_DIR" --output_path "$JSON_METRICS"

grep -F -q \
    "[fp4_cfg] Megatron FP4 training enabled: fp4=e2m1 recipe=nvfp4 fp4_param=False" \
    "$RUN_LOG"
grep -q "\[fp4_cfg\] TE per-module precision recipe loaded" "$RUN_LOG"
grep -q "\[nvfp4_pertoken\] per-token NVFP4 activation scaling active" "$RUN_LOG"
REFIT_COUNT=$(grep -F -c "[nvfp4_pertoken] refit: quantized" "$RUN_LOG" || true)
if [[ $REFIT_COUNT -lt $MAX_STEPS ]]; then
    echo "[ERROR] Expected at least $MAX_STEPS quantized refits, found $REFIT_COUNT"
    exit 1
fi

mapfile -t QUANTIZED_COUNTS < <(
    grep -F "[nvfp4_pertoken] refit: quantized" "$RUN_LOG" \
        | sed -E 's/.*refit: quantized ([0-9]+) expert.*/\1/'
)
UNIQUE_QUANTIZED_COUNTS=$(printf '%s\n' "${QUANTIZED_COUNTS[@]}" | sort -nu | wc -l)
if [[ $UNIQUE_QUANTIZED_COUNTS -ne 1 ]]; then
    echo "[ERROR] Quantized expert-group count changed across refits: ${QUANTIZED_COUNTS[*]}"
    exit 1
fi

MAX_RECORDED_STEP=$(jq -r 'if has("train/loss") then (."train/loss" | keys | map(tonumber) | max // 0) else 0 end' "$JSON_METRICS")
if [[ $MAX_RECORDED_STEP -lt $MAX_STEPS ]]; then
    echo "[ERROR] Expected train/loss through step $MAX_STEPS, found step $MAX_RECORDED_STEP"
    exit 1
fi

uv run --no-sync tests/check_metrics.py "$JSON_METRICS" \
    'min(data["train/num_valid_samples"]) > 0' \
    'mean(data["train/gen_kl_error"]) < 0.05' \
    'median(data["train/token_mult_prob_error"]) < 2.0' \
    'max(data["train/reward"]) > -1.1' \
    'mean(data["train/grad_norm"], 2, 0) > 0.0' \
    "abs(float(data[\"train/loss\"][\"$MAX_RECORDED_STEP\"])) < 1000000"

mapfile -t TRAIN_DATA_FILES < <(
    find "$LOG_DIR" -type f -name 'train_data_step*.jsonl' -print | sort -V
)
if [[ ${#TRAIN_DATA_FILES[@]} -ne $MAX_STEPS ]]; then
    echo "[ERROR] Expected $MAX_STEPS rollout JSONL files, found ${#TRAIN_DATA_FILES[@]}"
    exit 1
fi
