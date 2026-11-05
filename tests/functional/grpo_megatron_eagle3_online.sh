#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath $SCRIPT_DIR/../..)

set -eou pipefail

EXP_NAME=$(basename $0 .sh)
EXP_DIR=$SCRIPT_DIR/$EXP_NAME
LOG_DIR=$EXP_DIR/logs
JSON_METRICS=$EXP_DIR/metrics.json
RUN_LOG=$EXP_DIR/run.log
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

# Mark the current repo as safe, since wandb fetches metadata about the repo
git config --global --add safe.directory $PROJECT_ROOT

POLICY_MODEL=${NRL_EAGLE3_POLICY_MODEL:-Qwen/Qwen3-1.7B}
DRAFT_MODEL=${NRL_EAGLE3_DRAFT_MODEL:-AngelSlim/Qwen3-1.7B_eagle3}
CONFIG_PATH=$PROJECT_ROOT/examples/configs/recipes/llm/grpo-qwen3-1.7b-1n4g-megatron-eagle3.yaml

rm -rf $EXP_DIR $LOG_DIR
mkdir -p $EXP_DIR $LOG_DIR

cd $PROJECT_ROOT
uv run coverage run -a --data-file=$PROJECT_ROOT/tests/.coverage --source=$PROJECT_ROOT/nemo_rl \
    $PROJECT_ROOT/examples/run_grpo.py \
    --config $CONFIG_PATH \
    policy.model_name="$POLICY_MODEL" \
    policy.tokenizer.name="$POLICY_MODEL" \
    policy.draft.model_name="$DRAFT_MODEL" \
    policy.generation.vllm_kwargs.speculative_config.model="$DRAFT_MODEL" \
    grpo.max_num_steps=2 \
    logger.tensorboard_enabled=true \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=false \
    logger.monitor_gpus=true \
    checkpointing.enabled=false \
    cluster.gpus_per_node=2 \
    $@ \
    2>&1 | tee $RUN_LOG

if grep -q "Speculative decoding is enabled without draft refit sync" "$RUN_LOG"; then
    echo "Unexpected startup-weight warning for refit-backed Eagle3 path"
    exit 1
fi

grep -q "Draft Loss:" "$RUN_LOG"

uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

uv run tests/check_metrics.py $JSON_METRICS \
    'min(data["train/draft_loss"]) > 0'
