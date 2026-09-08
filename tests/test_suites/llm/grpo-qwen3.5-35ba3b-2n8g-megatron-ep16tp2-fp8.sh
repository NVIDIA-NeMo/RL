#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=2
STEPS_PER_RUN=20
MAX_STEPS=20
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=240
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
    checkpointing.enabled=True \
    checkpointing.checkpoint_dir=$CKPT_DIR \
    $@ \
    2>&1 | tee $RUN_LOG

uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

if [[ $(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | max' $JSON_METRICS) -ge $MAX_STEPS ]]; then
    uv run tests/check_metrics.py $JSON_METRICS \
        'mean(data["train/gen_kl_error"]) < 0.004' \
        'max(data["train/reward"]) > 0.5'
    # gen_kl_error measured on 2n8g EP16 fp8 verification runs: first-20-step
    # mean 0.0024 (dynamic batching and packed), 126-step mean 0.0031 / max
    # 0.0050 => 0.004 bounds the 20-step mean with margin (bf16 parent: 0.002).

    rm -rf "$CKPT_DIR"
fi
