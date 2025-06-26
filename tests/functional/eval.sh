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

cd $PROJECT_ROOT
uv run $PROJECT_ROOT/examples/run_eval.py \
    cluster.gpus_per_node=2 \
    $@ \
    2>&1 | tee $RUN_LOG

cat $RUN_LOG | grep "score=" | sed 's/.*score=\([^ ]*\).*/{"score": \1}/' > $JSON_METRICS

uv run tests/check_metrics.py $JSON_METRICS \
  'data["score"] == 0.1' \
