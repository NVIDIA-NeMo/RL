#!/bin/bash
# clean up checkpoint directory on exit
trap "rm -rf /tmp/grpo_megatron_lora_checkpoints" EXIT

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath $SCRIPT_DIR/../..)
# Mark the current repo as safe, since wandb fetches metadata about the repo
git config --global --add safe.directory $PROJECT_ROOT

set -eou pipefail

EXP_NAME=$(basename $0 .sh)
EXP_DIR=$SCRIPT_DIR/$EXP_NAME
LOG_DIR=$EXP_DIR/logs
JSON_METRICS=$EXP_DIR/metrics.json
RUN_LOG=$EXP_DIR/run.log
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

rm -rf $EXP_DIR $LOG_DIR
mkdir -p $EXP_DIR $LOG_DIR

# Using Qwen2.5-0.5B instead of Qwen3-0.6B because the latter is not supported by Megatron yet
cd $PROJECT_ROOT
uv run coverage run -a --data-file=$PROJECT_ROOT/tests/.coverage --source=$PROJECT_ROOT/nemo_rl \
    $PROJECT_ROOT/examples/run_grpo.py \
    --config $PROJECT_ROOT/examples/configs/grpo_math_1B_megatron.yaml \
    policy.model_name=Qwen/Qwen2.5-0.5B \
    grpo.max_num_steps=3 \
    grpo.num_prompts_per_step=8 \
    grpo.num_generations_per_prompt=4 \
    data.shuffle=false \
    policy.megatron_cfg.peft.enabled=true \
    policy.megatron_cfg.peft.dim=32 \
    policy.train_global_batch_size=32 \
    policy.train_micro_batch_size=1 \
    policy.logprob_batch_size=32 \
    cluster.gpus_per_node=2 \
    logger.tensorboard_enabled=true \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=false \
    logger.monitor_gpus=true \
    checkpointing.enabled=true \
    checkpointing.save_period=3 \
    checkpointing.checkpoint_dir=/tmp/grpo_megatron_lora_checkpoints \
    $@ \
    2>&1 | tee $RUN_LOG

uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

uv run tests/check_metrics.py $JSON_METRICS \
    'max(data["train/reward"]) > 0.03' \
    'max(data["train/gen_kl_error"]) < 0.001'

