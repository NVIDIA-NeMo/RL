#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=4
GPUS_PER_NODE=4
STEPS_PER_RUN=3
MAX_STEPS=3
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=90
# ===== END CONFIG =====

exit_if_max_steps_reached

cd $PROJECT_ROOT

uv run examples/nemo_gym/run_grpo_nemo_gym.py \
    --config $CONFIG_PATH \
    grpo.max_num_steps=$MAX_STEPS \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=True \
    logger.wandb.project=nemo-rl \
    logger.wandb.name=$EXP_NAME \
    logger.monitor_gpus=True \
    logger.tensorboard_enabled=True \
    $@ \
    2>&1 | tee $RUN_LOG

uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

LAST_STEP=$(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | max' $JSON_METRICS)
if [[ "$LAST_STEP" -lt "$MAX_STEPS" ]]; then
    echo "[ERROR] Expected step $MAX_STEPS, reached $LAST_STEP"
    exit 1
fi

# token_mult_prob_error and gen_kl_error are what this job exists to watch: both
# compare the text TRT-LLM generated against the logprobs Megatron recomputes for
# it, so a chat-template or tool-parser regression on the TRT-LLM server shows up
# here. The bound matches the vLLM SWE1 nightly's.
uv run tests/check_metrics.py $JSON_METRICS \
    'median(data["train/token_mult_prob_error"]) < 1.05' \
    'mean(data["train/gen_kl_error"]) < 0.02' \
    'mean(data["train/total_reward/mean"]) > 0.05' \
    'all_finite(data["train/grad_norm"])'
