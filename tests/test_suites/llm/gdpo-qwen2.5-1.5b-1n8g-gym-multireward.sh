#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=1
GPUS_PER_NODE=8
STEPS_PER_RUN=10
MAX_STEPS=10
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
# Small model, but nemo_gym server startup + vLLM warmup adds overhead on top of the
# 10 steps; 60 min leaves margin for teardown + metric dump.
NUM_MINUTES=60
# ===== END CONFIG =====

exit_if_max_steps_reached

cd $PROJECT_ROOT

# example_tool_call_multireward ships its data (data/example.jsonl) — no HF download or
# ng_prepare_data needed. Regenerate it so the run never trains on stale example data.
( cd 3rdparty/Gym-workspace/Gym && uv run python resources_servers/example_tool_call_multireward/create_examples.py )

# Prepare the training data:
#  - attach `agent_ref` to each row (points rollouts at the env's agent),
#  - replicate so there are enough prompts to run MAX_STEPS (the shipped example set is
#    just a handful of rows; one epoch would otherwise end before a single step).
DATA_DIR=$EXP_DIR/data
mkdir -p $DATA_DIR
ENV_DATA=3rdparty/Gym-workspace/Gym/resources_servers/example_tool_call_multireward/data/example.jsonl
VALIDATION_PATH=$DATA_DIR/example_tool_call_multireward_validation.jsonl
TRAIN_PATH=$DATA_DIR/example_tool_call_multireward_train.jsonl
jq -c '. + {agent_ref: {name: "example_tool_call_multireward_simple_agent"}}' $ENV_DATA > $VALIDATION_PATH
for _ in $(seq 1 60); do cat $VALIDATION_PATH; done > $TRAIN_PATH

# Run the experiment via the gym entrypoint
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

# Smoke-level threshold; tighten after observing real runs.
if [[ $(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | max' $JSON_METRICS) -ge $MAX_STEPS ]]; then
    uv run tests/check_metrics.py $JSON_METRICS \
        'max(data["train/reward"]) > 0.0'

    # Clean up checkpoint directory after successful run to save space.
    rm -rf "$CKPT_DIR"
fi
