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
CHECKPOINT_DIR=$EXP_DIR/checkpoints
DATA_DIR=$EXP_DIR/data
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

rm -rf $EXP_DIR $LOG_DIR
mkdir -p $EXP_DIR $LOG_DIR $CHECKPOINT_DIR $DATA_DIR

# clean up checkpoint directory on exit
trap "rm -rf $CHECKPOINT_DIR" EXIT

cd $PROJECT_ROOT

# Reuse the checked-in NeMo-Gym sanity fixture so this functional test does not
# depend on external dataset download. The fixture rows use
# `example_multi_step_simple_agent`, which is provided by Gym's
# resources_servers/example_multi_step config. The model overrides use the same
# small non-identical Qwen3 pair as other distillation functional tests so this
# stays cheap while still catching a degenerate zero-KL path.
TRAIN_PATH=$DATA_DIR/example_multi_step_train.jsonl
VALIDATION_PATH=$DATA_DIR/example_multi_step_validation.jsonl
python - <<'PY' "$PROJECT_ROOT/tests/unit/environments/nemo_gym_test_data/test_nemo_gym_sanity.json" "$TRAIN_PATH" "$VALIDATION_PATH"
import json
import sys

fixture_path, train_path, validation_path = sys.argv[1:]
with open(fixture_path) as f:
    rows = json.load(f)["input"]

for output_path in (train_path, validation_path):
    with open(output_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
PY

uv run coverage run -a --data-file=$PROJECT_ROOT/tests/.coverage --source=$PROJECT_ROOT/nemo_rl \
    $PROJECT_ROOT/examples/nemo_gym/run_distillation_nemo_gym.py \
    --config $PROJECT_ROOT/examples/nemo_gym/distillation_qwen3_0_6b.yaml \
    policy.model_name=Qwen/Qwen3-0.6B-Base \
    policy.tokenizer.name=Qwen/Qwen3-0.6B-Base \
    teacher.model_name=Qwen/Qwen3-0.6B \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=false \
    checkpointing.enabled=false \
    checkpointing.checkpoint_dir=$CHECKPOINT_DIR \
    cluster.gpus_per_node=2 \
    policy.dtensor_cfg.tensor_parallel_size=1 \
    policy.dtensor_cfg.context_parallel_size=2 \
    policy.make_sequence_length_divisible_by=2 \
    teacher.dtensor_cfg.tensor_parallel_size=2 \
    teacher.dtensor_cfg.context_parallel_size=1 \
    data.train.data_path=$TRAIN_PATH \
    data.validation.data_path=$VALIDATION_PATH \
    $@ \
    2>&1 | tee $RUN_LOG

uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

uv run tests/check_metrics.py $JSON_METRICS \
    'data["train/loss"]["1"] > 0' \
    'data["train/loss"]["1"] < 2.0' \
    'data["timing/train/generation"]["1"] > 0' \
    'data["train/mean_gen_tokens_per_sample"]["1"] > 0'
