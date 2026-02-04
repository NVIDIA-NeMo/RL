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
    cluster.gpus_per_node=2 \
    grpo.max_num_steps=1 \
    logger.tensorboard_enabled=true \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=false \
    logger.collect_worker_init_timing=true \
    checkpointing.enabled=false \
    $@ \
    2>&1 | tee $RUN_LOG

# Find the timing file in the exp_* subdirectory (log_dir gets exp_XXX appended by run_grpo.py)
TIMING_FILE=$(find $LOG_DIR -name "worker_init_timing.json" -type f 2>/dev/null | head -1)

# Check that worker_init_timing.json was created
if [ -z "$TIMING_FILE" ] || [ ! -f "$TIMING_FILE" ]; then
    echo "ERROR: Worker init timing file not found in $LOG_DIR (searched exp_* subdirs)"
    exit 1
fi

# Verify the JSON file has expected structure and timing metrics
uv run python -c "
import json
import sys

with open('$TIMING_FILE') as f:
    data = json.load(f)

# Check top-level structure
assert 'timings' in data, 'Missing timings key'
assert 'metadata' in data, 'Missing metadata key'
assert 'num_policy_workers' in data['metadata'], 'Missing num_policy_workers in metadata'
assert 'num_vllm_workers' in data['metadata'], 'Missing num_vllm_workers in metadata'
assert len(data['timings']) > 0, 'No timing data found'

# Check for expected Megatron policy worker timing labels (prefixed with 'policy/')
expected_policy_labels = [
    'policy/module_import',
    'policy/total_init',
    'policy/setup_distributed_nccl',
    'policy/model_and_optimizer_setup',
]

# Check for expected vLLM generation worker timing labels (prefixed with 'vllm/')
expected_vllm_labels = [
    'vllm/module_import',
    'vllm/total_init',
    'vllm/create_engine',
]

expected_labels = expected_policy_labels + expected_vllm_labels
missing_labels = [label for label in expected_labels if label not in data['timings']]
if missing_labels:
    print(f'ERROR: Missing expected timing labels: {missing_labels}', file=sys.stderr)
    print(f'Available labels: {list(data[\"timings\"].keys())}', file=sys.stderr)
    sys.exit(1)

# Validate that timing values are reasonable (positive and less than 1000s)
for label, value in data['timings'].items():
    assert isinstance(value, (int, float)), f'Timing value for {label} is not a number: {value}'
    assert value >= 0, f'Timing value for {label} is negative: {value}'
    assert value < 1000, f'Timing value for {label} is unreasonably large (>1000s): {value}'

print('✅ Worker init timing file validated successfully')
print(f'  - Number of timing labels: {len(data[\"timings\"])}')
print(f'  - Number of policy workers: {data[\"metadata\"][\"num_policy_workers\"]}')
print(f'  - Number of vLLM workers: {data[\"metadata\"][\"num_vllm_workers\"]}')
print('  - Timing breakdown:')
for label, value in sorted(data['timings'].items()):
    print(f'    • {label}: {value:.4f}s')
"
