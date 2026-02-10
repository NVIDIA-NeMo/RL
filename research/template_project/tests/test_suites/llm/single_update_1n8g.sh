#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
NEMO_RL_ROOT=$(realpath $SCRIPT_DIR/../../../../..)

# Source common.env from local test suite
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=1
STEPS_PER_RUN=10
MAX_STEPS=10
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=30
# ===== END CONFIG =====

# This test does not convert tensorboard logs to metrics.json. We typically use this
# to check if the test is already completed to not run any further. This is relevant
# when MAX_STPES > STEPS_PER_RUN since when launching multiple times, we want to
# exit early if we've already completed the MAX_STEPS. See tests in nemo-rl for
# concrete examples.
exit_if_max_steps_reached

# Run the experiment
cd $PROJECT_ROOT

# Set the number of iterations via environment variable
NRL_FORCE_REBUILD_VENVS=true \
SINGLE_UPDATE_ITERS=$MAX_STEPS \
uv run single_update.py \
    --config configs/grpo_math_1B.yaml \
    cluster.gpus_per_node=8 \
    cluster.num_nodes=$NUM_NODES \
    $@ \
    2>&1 | tee $RUN_LOG

# We create a simple metrics.json to check if the script ran normally (usually based on tensorboard logs)
if grep -q "All done." "$RUN_LOG"; then
  echo '{"succeed": "yes"}' > $JSON_METRICS
else
  echo '{"succeed": "no"}' > $JSON_METRICS
fi

# This is standard for nemo-rl tests to always run this script. We have some automation
# that checks the output of this script for success or failure.
uv run $NEMO_RL_ROOT/tests/check_metrics.py $JSON_METRICS \
  'data["succeed"] == "yes"'
