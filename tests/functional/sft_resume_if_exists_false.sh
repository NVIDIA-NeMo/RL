#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath $SCRIPT_DIR/../..)
# Mark the current repo as safe, since wandb fetches metadata about the repo
git config --global --add safe.directory $PROJECT_ROOT

set -eou pipefail

EXP_NAME=$(basename $0 .sh)
EXP_DIR=$SCRIPT_DIR/$EXP_NAME
LOG_DIR=$EXP_DIR/logs
CKPT_DIR=$EXP_DIR/ckpts
RUN_LOG=$EXP_DIR/run.log
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

rm -rf $EXP_DIR
mkdir -p $EXP_DIR

prefix_output() {
  sed "s/^/$1/"
}

train_cmd() {
uv run coverage run -a --data-file=$PROJECT_ROOT/tests/.coverage --source=$PROJECT_ROOT/nemo_rl \
    $PROJECT_ROOT/examples/run_sft.py \
    policy.model_name=Qwen/Qwen3-0.6B \
    data.shuffle=false \
    sft.val_period=0 \
    sft.val_at_start=false \
    sft.val_at_end=false \
    policy.train_global_batch_size=4 \
    policy.train_micro_batch_size=1 \
    cluster.gpus_per_node=1 \
    logger.tensorboard_enabled=true \
    logger.wandb_enabled=false \
    logger.monitor_gpus=false \
    checkpointing.enabled=false \
    checkpointing.save_period=1 \
    $@
}

cd $PROJECT_ROOT

# One-step fresh baseline.
train_cmd logger.log_dir=$LOG_DIR/baseline sft.max_num_steps=1 checkpointing.enabled=false policy.dtensor_cfg.enabled=true policy.megatron_cfg.enabled=false $@ 2>&1 | prefix_output "[baseline] " | tee ${RUN_LOG}.baseline
uv run tests/json_dump_tb_logs.py $LOG_DIR/baseline --output_path $EXP_DIR/baseline.json

# Create an initial checkpoint in the shared checkpoint directory.
train_cmd logger.log_dir=$LOG_DIR/original sft.max_num_steps=1 checkpointing.enabled=true checkpointing.resume_if_exists=true checkpointing.checkpoint_dir=$CKPT_DIR/dtensor policy.dtensor_cfg.enabled=true policy.megatron_cfg.enabled=false $@ 2>&1 | prefix_output "[original] " | tee ${RUN_LOG}.original
test -d $CKPT_DIR/dtensor/step_1

# Reuse the same checkpoint directory, but force a cold start via resume_if_exists=false.
train_cmd logger.log_dir=$LOG_DIR/cold_start sft.max_num_steps=1 checkpointing.enabled=true checkpointing.resume_if_exists=false checkpointing.checkpoint_dir=$CKPT_DIR/dtensor policy.dtensor_cfg.enabled=true policy.megatron_cfg.enabled=false $@ 2>&1 | prefix_output "[cold_start] " | tee ${RUN_LOG}.cold_start
uv run tests/json_dump_tb_logs.py $LOG_DIR/cold_start --output_path $EXP_DIR/cold_start.json

test -d $CKPT_DIR/dtensor/run_0/step_1
test -d $CKPT_DIR/dtensor/step_1

uv run python - <<EOF $EXP_DIR/baseline.json $EXP_DIR/cold_start.json
import json
import sys

import numpy as np

baseline_json, cold_start_json = sys.argv[1:3]

with open(baseline_json) as f:
    baseline = json.load(f)
with open(cold_start_json) as f:
    cold_start = json.load(f)

def assert_all_close(name, *, rtol=1e-6):
    baseline_value = baseline[name]["1"]
    cold_start_value = cold_start[name]["1"]
    assert np.isclose(cold_start_value, baseline_value, rtol=rtol), (
        f"cold_start[{name!r}]['1'] ({cold_start_value}) != "
        f"baseline[{name!r}]['1'] ({baseline_value})"
    )
    print(
        f"cold_start[{name!r}]['1'] ({cold_start_value}) == "
        f"baseline[{name!r}]['1'] ({baseline_value})"
    )

assert_all_close("train/lr")
assert_all_close("train/global_valid_seqs")
assert_all_close("train/global_valid_toks")
assert_all_close("train/num_unmasked_tokens")
assert_all_close("train/num_valid_samples")
assert_all_close("train/grad_norm", rtol=0.05)
assert_all_close("train/loss", rtol=0.05)
EOF
