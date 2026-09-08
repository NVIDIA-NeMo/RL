#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=16
GPUS_PER_NODE=8
STEPS_PER_RUN=10
MAX_STEPS=50
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
# Async 16-node run: Gym server startup, vLLM warmup and the 120B HF->Megatron
# conversion dominate the first steps, so the budget is well above the
# steady-state step time.
NUM_MINUTES=840
# ===== END CONFIG =====

exit_if_max_steps_reached

# The Super Omni checkpoint and the multimodal Gym blend are far too large to
# ship with the repo, so this driver is parameterized rather than
# self-contained, and is listed in disabled.txt rather than a nightly suite.
# examples/nemo_gym/nemotron-3-super-omni/super_omni_launch.sh is the Slurm
# wrapper that exports these and submits the job.
: "${MODEL_PATH:?MODEL_PATH must point at the Super Omni HF checkpoint}"
: "${TRAIN_PATH:?TRAIN_PATH must point at the Gym-format training jsonl}"
VAL_PATH="${VAL_PATH:-$TRAIN_PATH}"

cd $PROJECT_ROOT
uv run examples/nemo_gym/run_grpo_nemo_gym.py \
    --config $CONFIG_PATH \
    grpo.max_num_steps=$MAX_STEPS \
    policy.model_name=$MODEL_PATH \
    policy.tokenizer.chat_template=$MODEL_PATH/chat_template.jinja \
    policy.generation.vllm_cfg.http_server_serving_chat_kwargs.chat_template=$MODEL_PATH/chat_template.jinja \
    data.train.data_path=$TRAIN_PATH \
    data.validation.data_path=$VAL_PATH \
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
        'max(data["train/reward"]) > 0.0'

    # Clean up checkpoint directory after successful run to save space.
    rm -rf "$CKPT_DIR"
fi
