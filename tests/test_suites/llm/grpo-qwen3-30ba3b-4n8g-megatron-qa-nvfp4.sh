#!/bin/bash
# Nightly low-cost signal for QARL on a non-Nano3 MoE model. This is a
# one-step health test, not a convergence benchmark: it exercises ModelOpt
# NVFP4 calibration, Megatron Qwen3 MoE expert weights, quantized vLLM
# generation, logprobs, and the quantized policy train step.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=4
GPUS_PER_NODE=8
STEPS_PER_RUN=1
MAX_STEPS=1
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=30
# ===== END CONFIG =====

exit_if_max_steps_reached

# Run the experiment
cd $PROJECT_ROOT
uv run examples/run_grpo.py \
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

# Convert tensorboard logs to json
uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# This one-step smoke is intentionally a QARL control-flow check, not a
# logprob-consistency benchmark. The matching BF16 Qwen3 one-step control can
# also produce very large token_mult_prob_error from a single long outlier, so
# the multi-step Qwen3 performance tests keep ownership of that metric.
grep -q "720 TensorQuantizers found in model" "$RUN_LOG"
grep -q "MegatronQuantPolicyWorker.*723 TensorQuantizers found in model" "$RUN_LOG"

MAX_RECORDED_STEP=$(jq -r 'if has("train/loss") then (."train/loss" | keys | map(tonumber) | max // 0) else 0 end' $JSON_METRICS)
if [[ $MAX_RECORDED_STEP -lt $MAX_STEPS ]]; then
    echo "[ERROR] Expected train/loss through step $MAX_STEPS, found step $MAX_RECORDED_STEP"
    exit 1
fi

uv run tests/check_metrics.py $JSON_METRICS \
    'data["train/loss"]["1"] > 0.0' \
    'data["train/loss"]["1"] < 0.2' \
    'data["train/num_valid_samples"]["1"] == 16' \
    'data["train/global_valid_seqs"]["1"] == 16' \
    'data["train/global_valid_toks"]["1"] > 30000' \
    'data["train/mean_gen_tokens_per_sample"]["1"] > 2000' \
    'data["train/mean_gen_tokens_per_sample"]["1"] < 4096' \
    'data["train/reward"]["1"] > 0.2' \
    'data["train/reward"]["1"] < 0.6' \
    'data["train/gen_kl_error"]["1"] > 0.0' \
    'data["train/gen_kl_error"]["1"] < 1.0' \
    'data["train/probs_ratio"]["1"] > 0.99' \
    'data["train/probs_ratio"]["1"] < 1.01' \
    'data["train/sampling_importance_ratio"]["1"] > 0.95' \
    'data["train/sampling_importance_ratio"]["1"] < 1.02'
