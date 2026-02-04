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
TIMING_FILE=$LOG_DIR/worker_init_timing.json
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

rm -rf $EXP_DIR $LOG_DIR
mkdir -p $EXP_DIR $LOG_DIR

cd $PROJECT_ROOT
uv run coverage run -a --data-file=$PROJECT_ROOT/tests/.coverage --source=$PROJECT_ROOT/nemo_rl \
    $PROJECT_ROOT/examples/run_grpo.py \
    policy.model_name=Qwen/Qwen3-0.6B \
    grpo.num_prompts_per_step=2 \
    grpo.num_generations_per_prompt=4 \
    policy.train_global_batch_size=4 \
    policy.train_micro_batch_size=1 \
    cluster.gpus_per_node=2 \
    grpo.max_num_steps=1 \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=false \
    logger.tensorboard_enabled=false \
    logger.collect_worker_init_timing=true \
    checkpointing.enabled=false \
    $@ \
    2>&1 | tee $RUN_LOG

# Check that worker_init_timing.json was created
if [ ! -f "$TIMING_FILE" ]; then
    echo "ERROR: Worker init timing file not found at $TIMING_FILE"
    exit 1
fi

# Verify the JSON file has expected structure
uv run python -c "
import json
import sys

with open('$TIMING_FILE') as f:
    data = json.load(f)

assert 'timings' in data, 'Missing timings key'
assert 'metadata' in data, 'Missing metadata key'
assert 'num_workers' in data['metadata'], 'Missing num_workers in metadata'
assert len(data['timings']) > 0, 'No timing data found'

print('✅ Worker init timing file validated successfully')
print(f'  - Number of timing labels: {len(data[\"timings\"])}')
print(f'  - Number of workers: {data[\"metadata\"][\"num_workers\"]}')
for label, value in data['timings'].items():
    print(f'  - {label}: {value:.4f}s')
"
