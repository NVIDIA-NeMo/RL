#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=2
GPUS_PER_NODE=8
STEPS_PER_RUN=10
MAX_STEPS=10
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
# 30B MoE across 2 nodes, plus nemo_gym head-server startup and vLLM warmup on top
# of the 10 steps; 120 min leaves margin for teardown + metric dump.
NUM_MINUTES=120
# ===== END CONFIG =====

exit_if_max_steps_reached

cd $PROJECT_ROOT

# circle_click generates its own data (no HF download). Regenerate rather than reuse the
# 5-row example.jsonl so the run never trains on stale committed data, and so train/eval
# are disjoint (distinct --seed-offset).
DATA_DIR=$EXP_DIR/data
mkdir -p $DATA_DIR
GYM_DIR=3rdparty/Gym-workspace/Gym
RAW_TRAIN=$DATA_DIR/circle_click_train_raw.jsonl
RAW_VALIDATION=$DATA_DIR/circle_click_validation_raw.jsonl
( cd $GYM_DIR && uv run python resources_servers/circle_click/generate_data.py \
    --n 512 --seed-offset 0 --out $PROJECT_ROOT/$RAW_TRAIN )
( cd $GYM_DIR && uv run python resources_servers/circle_click/generate_data.py \
    --n 32 --seed-offset 100000 --out $PROJECT_ROOT/$RAW_VALIDATION )

# Attach `agent_ref` so rollouts are routed to the env's agent. The name must match the
# group registered in resources_servers/circle_click/configs/circle_click.yaml.
TRAIN_PATH=$DATA_DIR/circle_click_train.jsonl
VALIDATION_PATH=$DATA_DIR/circle_click_validation.jsonl
jq -c '. + {agent_ref: {name: "circle_click_simple_agent"}}' $RAW_TRAIN > $TRAIN_PATH
jq -c '. + {agent_ref: {name: "circle_click_simple_agent"}}' $RAW_VALIDATION > $VALIDATION_PATH

# Run the experiment via the gym entrypoint (circle_click is a NeMo-Gym env, so this
# recipe runs through run_grpo_nemo_gym.py rather than run_vlm_grpo.py).
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

# Convert tensorboard logs to json
uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# Only run metrics if the target step is reached
if [[ $(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | max' $JSON_METRICS) -ge $MAX_STEPS ]]; then
    # Smoke-level threshold: this recipe has not been run end to end yet, so assert only
    # that the multimodal gym path produces non-zero reward. Tighten once real runs land.
    uv run tests/check_metrics.py $JSON_METRICS \
        'max(data["train/reward"]) > 0.0'

    # Clean up checkpoint directory after successful run to save space.
    rm -rf "$CKPT_DIR"
fi
