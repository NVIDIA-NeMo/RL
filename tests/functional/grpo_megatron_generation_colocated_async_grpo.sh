#!/bin/bash

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
CKPT_DIR=$EXP_DIR/ckpts
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

rm -rf $EXP_DIR $LOG_DIR
mkdir -p $EXP_DIR $LOG_DIR $CKPT_DIR

# clean up checkpoint directory on exit
trap "rm -rf $CKPT_DIR" EXIT

# async colocated
cd $PROJECT_ROOT
uv run coverage run -a --data-file=$PROJECT_ROOT/tests/.coverage --source=$PROJECT_ROOT/nemo_rl \
    $PROJECT_ROOT/examples/run_grpo.py \
    --config $PROJECT_ROOT/examples/configs/grpo_math_1B_megatron.yaml \
    policy.model_name=Qwen/Qwen2.5-0.5B \
    grpo.num_prompts_per_step=2 \
    grpo.num_generations_per_prompt=4 \
    policy.train_global_batch_size=4 \
    policy.logprob_batch_size=4 \
    policy.train_micro_batch_size=1 \
    policy.generation.backend=megatron \
    grpo.async_grpo.enabled=true \
    grpo.async_grpo.max_trajectory_age_steps=1 \
    grpo.async_grpo.in_flight_weight_updates=true \
    loss_fn.use_importance_sampling_correction=true \
    grpo.max_num_steps=3 \
    grpo.val_period=3 \
    grpo.max_val_samples=8 \
    grpo.val_batch_size=8 \
    cluster.gpus_per_node=2 \
    logger.tensorboard_enabled=true \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=false \
    logger.monitor_gpus=true \
    checkpointing.enabled=true \
    checkpointing.checkpoint_dir=$CKPT_DIR \
    checkpointing.save_period=2 \
    $@ \
    2>&1 | tee $RUN_LOG

uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# Smoke-level threshold (matches grpo_megatron_generation_async_gym.sh); tighten after CI runs.
uv run tests/check_metrics.py $JSON_METRICS \
    'median(data["train/gen_kl_error"]) < 1.3' \
    '"3" in data["train/loss"]' \
    '"3" in data["validation/accuracy"]'

# The save-bound step must defer the engine wake past the checkpoint save.
# `val_period=3` gives us a step 2 that does not wake/sleep cycle the engine before save.
if ! grep -q "Keeping colocated engine asleep for checkpointing" $RUN_LOG; then
    echo "FAIL: deferred-wake log line not found (colocated checkpoint path not exercised)"
    exit 1
fi

if [[ ! -f $CKPT_DIR/step_2/replay_buffer.pt ]]; then
    echo "FAIL: replay_buffer.pt not found in step_2 checkpoint"
    exit 1
fi
