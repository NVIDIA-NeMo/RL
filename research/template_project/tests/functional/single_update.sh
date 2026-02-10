#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath $SCRIPT_DIR/../..)

set -eou pipefail

EXP_NAME=$(basename $0 .sh)
EXP_DIR=$SCRIPT_DIR/$EXP_NAME
LOG_DIR=$EXP_DIR/logs
RUN_LOG=$EXP_DIR/run.log
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

rm -rf $EXP_DIR $LOG_DIR
mkdir -p $EXP_DIR $LOG_DIR

cd $PROJECT_ROOT

# Run single_update.py for just 1 iteration
SINGLE_UPDATE_ITERS=1 uv run coverage run -a --data-file=$PROJECT_ROOT/tests/.coverage --source=$PROJECT_ROOT/nemo_rl \
    single_update.py \
    --config $PROJECT_ROOT/configs/grpo_math_1B.yaml \
    cluster.gpus_per_node=1 \
    cluster.num_nodes=1 \
    policy.train_global_batch_size=1 \
    policy.train_micro_batch_size=1 \
    $@ \
    2>&1 | tee $RUN_LOG

echo "Functional test passed: single_update.py completed 1 step successfully"

