#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# CISPO-vs-GRPO A/B: control arm (vanilla GRPO with hard PPO clip).
# See examples/configs/recipes/llm/cispo-ab-qwen2.5-math-1.5b-instruct-1n8g-grpo.yaml.
# NOT in the CISPO PR - local research-validation only.

# ===== BEGIN CONFIG =====
NUM_NODES=1
STEPS_PER_RUN=100
MAX_STEPS=100
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))
NUM_MINUTES=90
# ===== END CONFIG =====

exit_if_max_steps_reached

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
    $@ \
    2>&1 | tee $RUN_LOG

# Dump TB to JSON so the A/B runs can be compared offline (e.g. with
# `python tests/json_dump_tb_logs.py --diff`).
uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# NOTE: no `tests/check_metrics.py` thresholds here. This is a research
# A/B - the goal is to *compare* the two arms, not gate either of them
# against absolute numbers. Inspect train/reward, validation/reward,
# train/token_mult_prob_error, and train/probs_ratio_clamped_frac side by
# side (wandb group=cispo-ab, or feed both metrics.json files to your
# preferred diff tool).
