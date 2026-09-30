#!/bin/bash
# Five-step GB200 test for zero train/generation KL on Qwen3-30B-A3B: Megatron
# training (TP1, EP4) against colocated Megatron inference on the same 4 GPUs
# (inference_optimized generation, in-place reshard), with the batch-invariant
# kernel stack (te_native GEMMs, ordered collectives, FA4).
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=1
GPUS_PER_NODE=4
STEPS_PER_RUN=5
MAX_STEPS=5
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=90
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

# The preset must have switched the kernels on, and must not have had to
# override anything the recipe sets (an override means the recipe drifted from
# the preset and would reintroduce train/generation mismatch).
grep -F -q "[zero_train_gen_mismatch] batch-invariant kernels enabled: backend=te_native collective=ordered flash_attention_version=4" $RUN_LOG
assert_not_grep "zero_train_gen_mismatch=true overrides" $RUN_LOG \
    "Recipe values conflict with the zero_train_gen_mismatch preset"

MAX_RECORDED_STEP=$(jq -r 'if has("train/loss") then (."train/loss" | keys | map(tonumber) | max // 0) else 0 end' $JSON_METRICS)
if [[ $MAX_RECORDED_STEP -lt $MAX_STEPS ]]; then
    echo "[ERROR] Expected train/loss through step $MAX_STEPS, found step $MAX_RECORDED_STEP"
    exit 1
fi

# Zero train/generation KL means bitwise agreement: gen_kl_error is exactly 0 and
# token_mult_prob_error (max multiplicative probability error) is exactly 1 on every
# step. Both min and max are checked so a negative KL estimate cannot slip through.
uv run --no-sync tests/check_metrics.py $JSON_METRICS \
    'min(data["train/num_valid_samples"]) > 0' \
    'all_finite(data["train/loss"])' \
    'all_finite(data["train/grad_norm"])' \
    'min(data["train/gen_kl_error"]) == 0' \
    'max(data["train/gen_kl_error"]) == 0' \
    'min(data["train/token_mult_prob_error"]) == 1' \
    'max(data["train/token_mult_prob_error"]) == 1'

# Clean up checkpoint directory after successful run to save space.
rm -rf "$CKPT_DIR"
