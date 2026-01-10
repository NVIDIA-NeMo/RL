#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env
# disable NVLS to avoid OOM issue
export NCCL_NVLS_ENABLE=0
export NRT_REBUILD_VENVS=true
# allow user to pass an existing HF checkpoint path based on instruction in https://github.com/NVIDIA-NeMo/RL/blob/main/docs/guides/deepseek.md
export MODEL_NAME=${NRL_DEEPSEEK_V3_HF_CKPT:-"unsloth/DeepSeek-V3-0324-BF16"}

# ===== BEGIN CONFIG =====
NUM_NODES=64
STEPS_PER_RUN=10
MAX_STEPS=10
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=240
# ===== END CONFIG =====

exit_if_max_steps_reached

# Run the experiment
cd $PROJECT_ROOT
uv run examples/run_grpo_math.py \
    --config $CONFIG_PATH \
    grpo.num_prompts_per_step=64 \
    grpo.num_generations_per_prompt=8 \
    grpo.max_num_steps=$MAX_STEPS \
    policy.model_name=$MODEL_NAME \
    policy.tokenizer.name=$MODEL_NAME \
    cluster.num_nodes=$NUM_NODES \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=True \
    logger.wandb.project=nemo-rl \
    logger.wandb.name=$EXP_NAME \
    logger.monitor_gpus=True \
    logger.tensorboard_enabled=True \
    $@ \
    2>&1 | tee $RUN_LOG

# Convert tensorboard logs to json
uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# Only run metrics if the target step is reached
if [[ $(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | max' $JSON_METRICS) -ge $MAX_STEPS ]]; then
    uv run tests/check_metrics.py $JSON_METRICS \
        'median(data["train/token_mult_prob_error"]) < 1.1' \
        'data["train/token_mult_prob_error"]["10"] < 1.1'

    # Clean up checkpoint directory after successful run to save space.
    rm -rf "$CKPT_DIR"
fi
