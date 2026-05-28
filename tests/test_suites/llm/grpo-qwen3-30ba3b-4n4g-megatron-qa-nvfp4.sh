#!/bin/bash
# Nightly low-cost signal for QARL on a non-Nano3 MoE model. This is a
# one-step health test, not a convergence benchmark: it exercises ModelOpt
# NVFP4 calibration, Megatron Qwen3 MoE expert weights, quantized vLLM
# generation, logprobs, and the quantized policy train step.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
set -eou pipefail

PROJECT_ROOT=$(realpath $SCRIPT_DIR/../../..)
EXP_NAME=$(basename $0 .sh)
EXP_DIR=$SCRIPT_DIR/$EXP_NAME
LOG_DIR=$EXP_DIR/logs
CKPT_DIR=$EXP_DIR/ckpts
JSON_METRICS=$EXP_DIR/metrics.json
RUN_LOG=$EXP_DIR/run.log
CONFIG_PATH=$PROJECT_ROOT/examples/modelopt/qa_grpo_qwen3_30ba3b_megatron.yaml

export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

exit_if_max_steps_reached() {
  STEPS_SO_FAR=$(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | max' $JSON_METRICS || echo 0)
  if [[ $STEPS_SO_FAR -ge $MAX_STEPS ]]; then
      echo "[INFO] Target step $MAX_STEPS reached, skipping run"
      exit 0
  fi
  echo "[INFO] Steps so far: $STEPS_SO_FAR, running till $MAX_STEPS steps"
}

# ===== BEGIN CONFIG =====
NUM_NODES=4
GPUS_PER_NODE=4
STEPS_PER_RUN=1
MAX_STEPS=1
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=45
# ===== END CONFIG =====

if [[ -n "${TEST_DRYRUN:-}" ]]; then
  echo "[INFO] TEST_DRYRUN mode: used for testing"
  exit
fi

mkdir -p $EXP_DIR $LOG_DIR $CKPT_DIR

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
grep -q "VllmQuantGenerationWorker.*720 TensorQuantizers found in model" "$RUN_LOG"
grep -q "MegatronQuantPolicyWorker.*2739 TensorQuantizers found in model" "$RUN_LOG"

# Only run metrics if the target step is reached
if [[ $(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | max' $JSON_METRICS) -ge $MAX_STEPS ]]; then
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
        'data["train/gen_kl_error"]["1"] > 0.05' \
        'data["train/gen_kl_error"]["1"] < 1.0' \
        'data["train/probs_ratio"]["1"] > 0.99' \
        'data["train/probs_ratio"]["1"] < 1.01' \
        'data["train/sampling_importance_ratio"]["1"] > 0.95' \
        'data["train/sampling_importance_ratio"]["1"] < 1.02'
fi
