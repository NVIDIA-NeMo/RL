#!/bin/bash
# Nightly signal for real-quant Quantization-Aware Distillation (QAD): the same
# NVFP4 Megatron student as distillation-qwen3-1.7b-1n8g-megatron-qa-nvfp4, with
# vLLM serving the policy-exported NVFP4 checkpoint through its native ModelOpt
# loader. Loss gates follow the fake-quant sibling; NVFP4 rollout needs GB200.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=1
GPUS_PER_NODE=4
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
    if ! grep -q "quantization=modelopt" "$RUN_LOG"; then
        echo "FAIL: vLLM did not load a native ModelOpt quantization method"
        exit 1
    fi
    if grep -q "FakeQuantWorker" "$RUN_LOG"; then
        echo "FAIL: real-quant run unexpectedly used FakeQuantWorker"
        exit 1
    fi

    uv run tests/check_metrics.py $JSON_METRICS \
        'data["train/loss"]["1"] > 0' \
        'data["train/loss"]["1"] < 0.10' \
        'data["train/loss"]["25"] < 0.07' \
        'mean(data["train/loss"], -6, -1) < 0.07' \
        '(data["train/loss"]["1"] - mean(data["train/loss"], -6, -1)) / data["train/loss"]["1"] > 0.20' \
        'mean(data["timing/train/total_step_time"], -6, -1) < 30'

    # Clean up checkpoint directory after successful run to save space.
    rm -rf "$CKPT_DIR"
fi
