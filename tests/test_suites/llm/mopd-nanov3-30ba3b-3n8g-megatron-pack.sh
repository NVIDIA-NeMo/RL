#!/bin/bash
# MOPD nightly smoke: nano-v3 30B-A3B student distilled from a nano-v3 teacher
# (student == teacher -> OPD loss ~0), sequence packing ON, 3 nodes
# (1 policy + 1 vLLM + 1 teacher). Uses the nemo_gym entrypoint, so this driver
# is self-contained rather than sourcing common.env (which requires the config
# under examples/configs/recipes/llm/; gym recipes live in examples/nemo_gym/).
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath $SCRIPT_DIR/../../..)
EXP_NAME=$(basename $0 .sh)
EXP_DIR=$SCRIPT_DIR/$EXP_NAME
LOG_DIR=$EXP_DIR/logs
CKPT_DIR=$EXP_DIR/ckpts
JSON_METRICS=$EXP_DIR/metrics.json
RUN_LOG=$EXP_DIR/run.log
CONFIG_PATH=$PROJECT_ROOT/examples/nemo_gym/mopd_nanov3_3n8g_smoke.yaml
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

# ===== BEGIN CONFIG =====
NUM_NODES=3
GPUS_PER_NODE=8
MAX_STEPS=5
NUM_MINUTES=60
# nemo_gym train/validation jsonl — set by the nightly environment.
NRL_TRAIN_PATH="${NRL_TRAIN_PATH:?set NRL_TRAIN_PATH to the nemo_gym train jsonl}"
NRL_VAL_PATH="${NRL_VAL_PATH:?set NRL_VAL_PATH to the nemo_gym validation jsonl}"
# ===== END CONFIG =====

mkdir -p $EXP_DIR $LOG_DIR $CKPT_DIR

cd $PROJECT_ROOT
uv run examples/nemo_gym/run_grpo_nemo_gym.py \
    --config $CONFIG_PATH \
    grpo.max_num_steps=$MAX_STEPS \
    data.train.data_path=$NRL_TRAIN_PATH \
    data.validation.data_path=$NRL_VAL_PATH \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=True \
    logger.wandb.project=nemo-rl \
    logger.wandb.name=$EXP_NAME \
    logger.monitor_gpus=True \
    logger.tensorboard_enabled=True \
    checkpointing.enabled=True \
    checkpointing.checkpoint_dir=$CKPT_DIR \
    "$@" \
    2>&1 | tee $RUN_LOG

# Convert tensorboard logs to json
uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# Student == teacher, so the distillation signal should be ~0 and the
# train-to-inference probability error should stay near 1.0.
if [[ $(jq 'to_entries | .[] | select(.key == "train/token_mult_prob_error") | .value | keys | map(tonumber) | max' $JSON_METRICS) -ge $MAX_STEPS ]]; then
    uv run tests/check_metrics.py $JSON_METRICS \
        'median(data["train/token_mult_prob_error"]) < 1.1'

    # Clean up checkpoint directory after a successful run to save space.
    rm -rf "$CKPT_DIR"
fi
