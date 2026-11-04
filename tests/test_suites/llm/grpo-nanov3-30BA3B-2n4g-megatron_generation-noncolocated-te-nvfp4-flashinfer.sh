#!/bin/bash
# Five-step GB200 test for Nemotron Nano v3 30B-A3B with TE per-token NVFP4 routed
# experts in training and FlashInfer per-token NVFP4 routed experts in Megatron
# generation, without batch-invariant mode. Refit stays BF16.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=2
GPUS_PER_NODE=4
STEPS_PER_RUN=5
MAX_STEPS=5
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=120
SNAPSHOT_MEGATRON_BRIDGE=1
# ===== END CONFIG =====

exit_if_max_steps_reached

cd $PROJECT_ROOT
uv run --no-sync examples/run_grpo.py \
    --config $CONFIG_PATH \
    grpo.max_num_steps=$MAX_STEPS \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=True \
    logger.wandb.project=nemo-rl \
    logger.wandb.name=$EXP_NAME \
    logger.monitor_gpus=True \
    logger.tensorboard_enabled=True \
    checkpointing.enabled=False \
    checkpointing.checkpoint_dir=$CKPT_DIR \
    $@ \
    2>&1 | tee $RUN_LOG

uv run --no-sync tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# Training must have run TE FP4.
grep -F -q "[fp4_cfg] Megatron FP4 training enabled: fp4=e2m1 recipe=nvfp4 fp4_param=False" $RUN_LOG

MAX_RECORDED_STEP=$(jq -r 'if has("train/loss") then (."train/loss" | keys | map(tonumber) | max // 0) else 0 end' $JSON_METRICS)
if [[ $MAX_RECORDED_STEP -lt $MAX_STEPS ]]; then
    echo "[ERROR] Expected train/loss through step $MAX_STEPS, found step $MAX_RECORDED_STEP"
    exit 1
fi

uv run --no-sync tests/check_metrics.py $JSON_METRICS \
    'min(data["train/num_valid_samples"]) > 0' \
    'all_finite(data["train/loss"])' \
    'all_finite(data["train/grad_norm"])' \
    'all_finite(data["train/gen_kl_error"])' \
    'max(data["train/gen_kl_error"]) < 0.05'

# Clean up checkpoint directory after successful run to save space.
rm -rf "$CKPT_DIR"
