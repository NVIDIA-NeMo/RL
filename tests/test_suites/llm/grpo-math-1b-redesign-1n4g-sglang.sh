#!/bin/bash
set -eou pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath $SCRIPT_DIR/../../..)

EXP_NAME=$(basename $0 .sh)
EXP_DIR=$SCRIPT_DIR/$EXP_NAME
LOG_DIR=$EXP_DIR/logs
CKPT_DIR=$EXP_DIR/ckpts
JSON_METRICS=$EXP_DIR/metrics.json
RUN_LOG=$EXP_DIR/run.log

# This test targets the redesigned sglang backend config which lives outside
# examples/configs/recipes/llm, so we set CONFIG_PATH explicitly rather than
# sourcing common.env.
CONFIG_PATH=$PROJECT_ROOT/examples/configs/grpo_math_1B_redesign_sglang.yaml
if [[ ! -f $CONFIG_PATH ]]; then
  echo "[ERROR] Config file $CONFIG_PATH not found"
  exit 1
fi

export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}

exit_if_max_steps_reached() {
  STEPS_SO_FAR=$(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | max' $JSON_METRICS || echo 0)
  if [[ $STEPS_SO_FAR -ge $MAX_STEPS ]]; then
      echo "[INFO] Target step $MAX_STEPS reached, skipping run"
      exit 0
  fi
  echo "[INFO] Steps so far: $STEPS_SO_FAR, running till $MAX_STEPS steps"
}

if [[ -n "${TEST_DRYRUN:-}" ]]; then
  echo "[INFO] TEST_DRYRUN mode: used for testing"
  exit
fi

mkdir -p $EXP_DIR $LOG_DIR $CKPT_DIR

# ===== BEGIN CONFIG =====
NUM_NODES=1
STEPS_PER_RUN=450
MAX_STEPS=450
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=150  # ~13.7s/step without piecewise CUDA graphs
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
    checkpointing.enabled=True \
    checkpointing.checkpoint_dir=$CKPT_DIR \
    $@ \
    2>&1 | tee $RUN_LOG

# Convert tensorboard logs to json
uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# Only run metrics if the target step is reached
if [[ $(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | max' $JSON_METRICS) -ge $MAX_STEPS ]]; then
    uv run tests/check_metrics.py $JSON_METRICS \
        'median(data["train/token_mult_prob_error"]) < 1.1' \
        'mean(data["timing/train/total_step_time"], 2) < 25'

    # Clean up checkpoint directory after successful run to save space.
    rm -rf "$CKPT_DIR"
fi
