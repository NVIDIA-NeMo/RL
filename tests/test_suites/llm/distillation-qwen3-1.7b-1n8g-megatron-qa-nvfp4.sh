#!/bin/bash
# Nightly low-cost signal for the on-policy Quantization-Aware Distillation
# (QAD) integration: NVFP4-quantized Megatron student + NVFP4-quantized vLLM
# generation worker, self-distilled from Qwen3-1.7B-Base.
#
# Calibration runs (24 / 25-step traces) showed:
#   start loss   ~0.068, end loss ~0.045-0.047, last-5 mean ~0.044-0.048
#   relative reduction ~33-36%, total wall time ~20-22 min on 1n8g H100
# Thresholds below leave ~2x headroom on each direction so the test fires on
# real regressions (loss flat / blowing up, step time doubling) without
# false-positives from run-to-run noise.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=1
STEPS_PER_RUN=25
MAX_STEPS=25
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=60
# ===== END CONFIG =====

exit_if_max_steps_reached

# Run the experiment
cd $PROJECT_ROOT
uv run examples/run_distillation.py \
    --config $CONFIG_PATH \
    distillation.max_num_steps=$MAX_STEPS \
    distillation.val_period=30 \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=True \
    logger.wandb.project=nemo-rl \
    logger.wandb.name=$EXP_NAME \
    logger.monitor_gpus=True \
    logger.tensorboard_enabled=True \
    checkpointing.enabled=True \
    checkpointing.checkpoint_dir=$CKPT_DIR \
    $@ \
    2>&1 | tee $RUN_LOG

# Convert tensorboard logs to json
uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# Only run metrics if the target step is reached
if [[ $(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | max' $JSON_METRICS) -ge $MAX_STEPS ]]; then
    uv run tests/check_metrics.py $JSON_METRICS \
        'data["train/loss"]["1"] > 0' \
        'data["train/loss"]["1"] < 0.10' \
        'data["train/loss"]["25"] < 0.07' \
        'mean(data["train/loss"], -6, -1) < 0.07' \
        '(data["train/loss"]["1"] - mean(data["train/loss"], -6, -1)) / data["train/loss"]["1"] > 0.20' \
        'max(data["ray/node.0.gpu.0.mem_gb"]) < 70' \
        'mean(data["timing/train/total_step_time"], -6, -1) < 30'

    # Clean up checkpoint directory after successful run to save space.
    rm -rf "$CKPT_DIR"
fi
