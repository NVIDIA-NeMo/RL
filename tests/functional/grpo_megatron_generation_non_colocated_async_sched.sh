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
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

rm -rf $EXP_DIR $LOG_DIR
mkdir -p $EXP_DIR $LOG_DIR

# Non-colocated Megatron generation (1 generation GPU + 1 training GPU) with
# mcore's async dynamic-batching scheduler. The dedicated engine is suspended and
# refit on every step, so this also covers async scheduling across weight updates.
# Using Qwen2.5-0.5B instead of Qwen3-0.6B because the latter is not supported by Megatron yet
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
    policy.generation.colocated.enabled=false \
    policy.generation.colocated.resources.gpus_per_node=1 \
    policy.generation.mcore_generation_config.refit_backend=nccl \
    policy.generation.mcore_generation_config.async_sched_mode=async \
    cluster.gpus_per_node=2 \
    grpo.max_num_steps=10 \
    grpo.val_period=10 \
    grpo.max_val_samples=8 \
    grpo.val_batch_size=8 \
    logger.tensorboard_enabled=true \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=false \
    logger.monitor_gpus=true \
    checkpointing.enabled=false \
    $@ \
    2>&1 | tee $RUN_LOG

uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

uv run tests/check_metrics.py $JSON_METRICS \
    'max(data["train/token_mult_prob_error"]) < 1.05' \
    '"10" in data["train/loss"]'

# mcore quietly falls back to legacy ordering for any step it cannot overlap, so
# a nonzero overlap-step count is the only signal that async_sched_mode=async
# actually took effect rather than being accepted and ignored.
ASYNC_SCHED_STEPS=$(grep -o 'mcore async scheduling steps (cumul): [0-9]*' $RUN_LOG | grep -o '[0-9]*$' | sort -n | tail -1 || true)
if [[ -z "${ASYNC_SCHED_STEPS:-}" ]]; then
    echo "FAIL: async scheduling counter not found (async_sched_mode did not reach the engine)"
    exit 1
fi
if [[ "$ASYNC_SCHED_STEPS" -eq 0 ]]; then
    echo "FAIL: async scheduling reported 0 overlap steps"
    exit 1
fi
echo "async scheduling overlap steps: $ASYNC_SCHED_STEPS"
