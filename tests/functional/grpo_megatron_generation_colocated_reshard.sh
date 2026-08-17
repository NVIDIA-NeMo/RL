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

# Colocated reshard: training runs TP2 while inference runs TP1 x DP2, so
# prepare_for_generation must build a dedicated inference model and swap
# weights across differing layouts. A wrong-weights reshard shows up as
# generation/training logprob disagreement, hence the mult_prob_error gate.
cd $PROJECT_ROOT
uv run coverage run -a --data-file=$PROJECT_ROOT/tests/.coverage --source=$PROJECT_ROOT/nemo_rl \
    $PROJECT_ROOT/examples/run_grpo.py \
    --config $PROJECT_ROOT/examples/configs/grpo_math_1B_megatron.yaml \
    policy.model_name=Qwen/Qwen3-0.6B \
    grpo.num_prompts_per_step=2 \
    grpo.num_generations_per_prompt=4 \
    policy.train_global_batch_size=4 \
    policy.logprob_batch_size=4 \
    policy.train_micro_batch_size=1 \
    policy.megatron_cfg.tensor_model_parallel_size=2 \
    policy.generation.backend=megatron \
    ++policy.generation.mcore_generation_config.transformer_impl=inference_optimized \
    ++policy.generation.mcore_generation_config.tensor_model_parallel_size=1 \
    policy.generation.mcore_generation_config.refit_backend=nccl \
    cluster.gpus_per_node=2 \
    grpo.max_num_steps=2 \
    logger.tensorboard_enabled=true \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=false \
    logger.monitor_gpus=true \
    checkpointing.enabled=false \
    $@ \
    2>&1 | tee $RUN_LOG

uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

uv run tests/check_metrics.py $JSON_METRICS \
    'max(data["train/token_mult_prob_error"]) < 1.05'

# Check that colocated reshard actually took place.
# A matched layout that requires no reshard would pass all tests.
# While the configs we are passing guarantee reshard, this is still a useful sanity check.
if ! grep -q "\[colocated-reshard\] building dedicated inference model" $RUN_LOG; then
    echo "FAIL: colocated-reshard marker not found; dedicated inference model was never built"
    exit 1
fi
