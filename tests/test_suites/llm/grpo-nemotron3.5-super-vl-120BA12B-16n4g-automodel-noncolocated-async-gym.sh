#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=16
GPUS_PER_NODE=4
STEPS_PER_RUN=5
MAX_STEPS=5
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
# ~25 min startup (120B load, vLLM CUDA-graph warmup, NeMo Gym servers and
# venvs) plus 5 async steps at ~6 min/step on 8 training + 8 generation nodes.
NUM_MINUTES=120
# ===== END CONFIG =====

exit_if_max_steps_reached

cd $PROJECT_ROOT

# Prepare the NeMo Gym workplace_assistant dataset (mirrors the nanov3 async-gym tests;
# no tool trimming is needed at 16K context).
cd 3rdparty/Gym-workspace/Gym
if [[ ! -f env.yaml ]]; then
    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo "[ERROR] HF_TOKEN is not set"
        exit 1
    fi
    echo "hf_token: $HF_TOKEN" >> env.yaml
fi
uv run ng_prepare_data "+config_paths=[resources_servers/workplace_assistant/configs/workplace_assistant.yaml]" \
    +output_dirpath=data/workplace_assistant \
    +mode=train_preparation \
    +should_download=true \
    +data_source=huggingface
cd -
TRAIN_PATH=3rdparty/Gym-workspace/Gym/data/workplace_assistant/train.jsonl
VALIDATION_PATH=3rdparty/Gym-workspace/Gym/data/workplace_assistant/validation.jsonl

# Release smoke: 5 async GRPO steps with non-colocated generation on 16 x 4-GPU
# GB200 nodes. Validation is skipped (val_at_start=false, val_period > MAX_STEPS):
# one pass over the 545 tasks costs ~3 min and 5 steps is too few to move
# accuracy; the refit-health metrics below are what this test guards, including
# the multi-node NCCL collective refit of the AutoModel trainer.
uv run examples/nemo_gym/run_grpo_nemo_gym.py \
    --config $CONFIG_PATH \
    grpo.max_num_steps=$MAX_STEPS \
    grpo.val_at_start=False \
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
    "$@" \
    2>&1 | tee $RUN_LOG

# Convert tensorboard logs to json
uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# Only run metrics if the target step is reached
if [[ $(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | max' $JSON_METRICS) -ge $MAX_STEPS ]]; then
    uv run tests/check_metrics.py $JSON_METRICS \
        'median(data["train/token_mult_prob_error"]) < 1.1' \
        "data['train/token_mult_prob_error']['$MAX_STEPS'] < 1.1" \
        'mean(data["train/gen_kl_error"]) < 0.005' \
        'max(data["train/reward"]) > 0.5' \
        "data['train/reward']['$MAX_STEPS'] > 0.4"

    # Clean up checkpoint directory after successful run to save space.
    rm -rf "$CKPT_DIR"
fi
