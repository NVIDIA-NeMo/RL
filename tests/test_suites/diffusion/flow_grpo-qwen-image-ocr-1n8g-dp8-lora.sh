#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=1
STEPS_PER_RUN=60
MAX_STEPS=60
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=360
# ===== END CONFIG =====

exit_if_max_steps_reached

cd $PROJECT_ROOT

# The recipe inherits data.{train,val}.prompt_file from the exemplar config;
# generate the OCR prompt files on first use.
if [[ ! -f examples/data/diffusion/ocr/train_prompts.jsonl ]]; then
  uv run python tools/export_ocr_prompts.py --out-dir examples/data/diffusion/ocr
fi

uv run --extra diffusion examples/run_flow_grpo.py \
    --config $CONFIG_PATH \
    flow_grpo.max_num_steps=$MAX_STEPS \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=True \
    logger.wandb.project=nemo-rl \
    logger.wandb.name=$EXP_NAME \
    logger.tensorboard_enabled=True \
    checkpointing.enabled=True \
    checkpointing.checkpoint_dir=$CKPT_DIR \
    $@ \
    2>&1 | tee $RUN_LOG

# Convert tensorboard logs to json
uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# Flow-GRPO logs 0-based steps, so the last step key is MAX_STEPS - 1.
LAST_STEP=$((MAX_STEPS - 1))
if [[ $(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | max' $JSON_METRICS) -ge $LAST_STEP ]]; then
    # Nightly uses the CPU PaddleOCR (ocr) reward, not the exemplar's genrm_ocr
    # judge. Measured 60-step val/reward_mean on 1x8 B200 (aligned config):
    # 0.670 -> 0.897 (+0.227); gate conservatively at +0.03 (random SDE rollouts
    # and judge noise leave a wide margin against flakiness).
    uv run tests/check_metrics.py $JSON_METRICS \
        "median(data['train/mean_ratio']) > 0.5" \
        "median(data['train/mean_ratio']) < 1.5" \
        "data['val/reward_mean']['$LAST_STEP'] > data['val/reward_mean']['0'] + 0.03" \
        "max(data['train/grad_norm']) < 100"

    # Clean up checkpoint directory after successful run to save space.
    rm -rf "$CKPT_DIR"
fi
