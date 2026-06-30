#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

export NRL_ROUTER_REPLAY_VALIDATE=1

# ===== BEGIN CONFIG =====
NUM_NODES=16
GPUS_PER_NODE=8
STEPS_PER_RUN=10
MAX_STEPS=10
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=240
# ===== END CONFIG =====

exit_if_max_steps_reached

cd $PROJECT_ROOT

DATA_DIR=$EXP_DIR/data
RAW_DATA_DIR=$DATA_DIR/raw
TRAIN_PATH=$DATA_DIR/swe1_train.jsonl
VALIDATION_PATH=$DATA_DIR/swe1_validation.jsonl
mkdir -p $RAW_DATA_DIR

if [[ ! -f $RAW_DATA_DIR/swe1.jsonl ]]; then
    uv run hf download nvidia/Nemotron-RL-Super-Training-Blends swe1.jsonl \
        --repo-type dataset \
        --local-dir $RAW_DATA_DIR
fi

if [[ ! -f $TRAIN_PATH ]]; then
    head -n 512 $RAW_DATA_DIR/swe1.jsonl > $TRAIN_PATH
fi
if [[ ! -f $VALIDATION_PATH ]]; then
    tail -n 32 $RAW_DATA_DIR/swe1.jsonl > $VALIDATION_PATH
fi

uv run examples/nemo_gym/run_grpo_nemo_gym.py \
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
    data.train.data_path=$TRAIN_PATH \
    data.validation.data_path=$VALIDATION_PATH \
    $@ \
    2>&1 | tee $RUN_LOG

uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

if [[ $(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | max' $JSON_METRICS) -ge $MAX_STEPS ]]; then
    uv run tests/check_metrics.py $JSON_METRICS \
        'median(data["train/token_mult_prob_error"]) < 1.02'

    rm -rf "$CKPT_DIR"
fi
