#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath $SCRIPT_DIR/../..)
# Mark the current repo as safe, since wandb fetches metadata about the repo
git config --global --add safe.directory $PROJECT_ROOT

set -eou pipefail

EXP_NAME=$(basename $0 .sh)
EXP_DIR=$SCRIPT_DIR/$EXP_NAME
LOG_DIR=$EXP_DIR/logs
RUN_LOG=$EXP_DIR/run.log
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

rm -rf $EXP_DIR $LOG_DIR
mkdir -p $EXP_DIR $LOG_DIR

cd $PROJECT_ROOT
# Smoke test: drive run_grpo.py through the random-dataset path for a couple
# of steps and rely on `set -eou pipefail` to catch any failure. The random
# dataset has no ground-truth signal, so we don't validate metrics here.
uv run coverage run -a --data-file=$PROJECT_ROOT/tests/.coverage --source=$PROJECT_ROOT/nemo_rl \
    $PROJECT_ROOT/examples/run_grpo.py \
    policy.model_name=Qwen/Qwen3-0.6B \
    policy.tokenizer.name=Qwen/Qwen3-0.6B \
    policy.max_total_sequence_length=64 \
    policy.generation.vllm_cfg.max_model_len=64 \
    policy.generation.vllm_cfg.gpu_memory_utilization=0.45 \
    policy.generation.ignore_eos=true \
    ++policy.generation.output_len_or_output_len_generator=8 \
    grpo.num_prompts_per_step=2 \
    grpo.num_generations_per_prompt=2 \
    grpo.max_num_steps=2 \
    grpo.val_at_start=false \
    grpo.val_period=0 \
    policy.train_global_batch_size=4 \
    policy.train_micro_batch_size=1 \
    cluster.gpus_per_node=1 \
    +data.dataset_name=random \
    data.max_input_seq_length=32 \
    +data.input_len_or_input_len_generator=16 \
    +data.num_samples=4 \
    logger.tensorboard_enabled=true \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=false \
    logger.monitor_gpus=false \
    checkpointing.enabled=false \
    $@ \
    2>&1 | tee $RUN_LOG
