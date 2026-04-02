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

# This test will run for 2 steps and make sure that 1+1 steps w/ resume leads to the same result
# Because mcore does not support setting different max steps, to control this behavior we run for just 2 steps,
# but use the checkpointing.checkpoint_must_save_by=0:0:0:1 feature to exit after 1 step.

prefix_output() {
  sed "s/^/$1/"
}

train_cmd() {
uv run coverage run -a --data-file=$PROJECT_ROOT/tests/.coverage --source=$PROJECT_ROOT/nemo_rl \
    $PROJECT_ROOT/examples/run_sft.py \
    policy.model_name=Qwen/Qwen3-0.6B \
    sft.val_period=1 \
    sft.val_batches=2 \
    sft.val_global_batch_size=2 \
    policy.train_global_batch_size=4 \
    policy.train_micro_batch_size=1 \
    cluster.gpus_per_node=1 \
    logger.tensorboard_enabled=true \
    logger.wandb_enabled=false \
    logger.monitor_gpus=true \
    checkpointing.enabled=false \
    checkpointing.save_period=1 \
    $@ 
}

cd $PROJECT_ROOT

# Dtensor 2 step baseline
train_cmd logger.log_dir=$LOG_DIR/baseline sft.max_num_steps=2 policy.dtensor_cfg.enabled=true policy.megatron_cfg.enabled=false $@ 2>&1 | prefix_output "[baseline 2step] " | tee ${RUN_LOG}.2step_baseline
uv run tests/json_dump_tb_logs.py $LOG_DIR/baseline --output_path $EXP_DIR/baseline.json
# Dtensor 1+1 step
train_cmd logger.log_dir=$LOG_DIR/dtensor sft.max_num_steps=2 checkpointing.checkpoint_must_save_by=0:0:0:1 checkpointing.enabled=true checkpointing.checkpoint_dir=$CKPT_DIR/dtensor policy.dtensor_cfg.enabled=true policy.megatron_cfg.enabled=false $@ 2>&1 | prefix_output "[dtensor 1step] " | tee ${RUN_LOG}.dtensor_1step
uv run tests/json_dump_tb_logs.py $LOG_DIR/dtensor --output_path $EXP_DIR/dtensor_1step.json
train_cmd logger.log_dir=$LOG_DIR/dtensor sft.max_num_steps=2 checkpointing.enabled=true checkpointing.checkpoint_dir=$CKPT_DIR/dtensor policy.dtensor_cfg.enabled=true policy.megatron_cfg.enabled=false $@ 2>&1 | prefix_output "[dtensor 2step] " | tee ${RUN_LOG}.dtensor_2step
uv run tests/json_dump_tb_logs.py $LOG_DIR/dtensor --output_path $EXP_DIR/dtensor_2step.json
# Mcore 2+2 step
train_cmd logger.log_dir=$LOG_DIR/mcore sft.max_num_steps=2 checkpointing.checkpoint_must_save_by=0:0:0:1 checkpointing.enabled=true checkpointing.checkpoint_dir=$CKPT_DIR/mcore policy.dtensor_cfg.enabled=false policy.megatron_cfg.enabled=true $@ 2>&1 | prefix_output "[mcore 1step] " | tee ${RUN_LOG}.mcore_1step
uv run tests/json_dump_tb_logs.py $LOG_DIR/mcore --output_path $EXP_DIR/mcore_1step.json
train_cmd logger.log_dir=$LOG_DIR/mcore sft.max_num_steps=2 checkpointing.enabled=true checkpointing.checkpoint_dir=$CKPT_DIR/mcore policy.dtensor_cfg.enabled=false policy.megatron_cfg.enabled=true $@ 2>&1 | prefix_output "[mcore 2step] " | tee ${RUN_LOG}.mcore_2step
uv run tests/json_dump_tb_logs.py $LOG_DIR/mcore --output_path $EXP_DIR/mcore_2step.json

uv run python - <<EOF $EXP_DIR/baseline.json $EXP_DIR/dtensor_1step.json $EXP_DIR/dtensor_2step.json $EXP_DIR/mcore_1step.json $EXP_DIR/mcore_2step.json
import sys
import json
import numpy as np

baseline_json, dtensor_1step_json, dtensor_2step_json, mcore_1step_json, mcore_2step_json = sys.argv[1:6]

with open(baseline_json) as f:
    base = json.load(f)
with open(dtensor_1step_json) as f:
    dtensor_1 = json.load(f)
with open(dtensor_2step_json) as f:
    dtensor_2 = json.load(f)
with open(mcore_1step_json) as f:
    mcore_1 = json.load(f)
with open(mcore_2step_json) as f:
    mcore_2 = json.load(f)

def assert_all_close(i, name, close_args={}, **kwargs):
    baseline = kwargs["baseline"][name][str(i)]
    for test_name, test_data in kwargs.items():
        val = test_data[name][str(i)]
        assert np.isclose(val, baseline, **close_args), f"{test_name}[{repr(name)}][{repr(i)}] ({val}) != baseline[{repr(name)}][{repr(i)}] ({baseline})"
        print(f"{test_name}[{repr(name)}][{repr(i)}] ({val}) == baseline[{repr(name)}][{repr(i)}] ({baseline})")
    print(f"✓ {name} {i} is equal")

assert_all_close(1, "train/lr", baseline=base, dtensor_1=dtensor_1, dtensor_2=dtensor_2, mcore_1=mcore_1, mcore_2=mcore_2)
assert_all_close(2, "train/lr", baseline=base, dtensor_2=dtensor_2, mcore_2=mcore_2)
assert_all_close(1, "train/global_valid_seqs", baseline=base, dtensor_1=dtensor_1, dtensor_2=dtensor_2, mcore_1=mcore_1, mcore_2=mcore_2)
assert_all_close(2, "train/global_valid_seqs", baseline=base, dtensor_2=dtensor_2, mcore_2=mcore_2)
assert_all_close(1, "train/global_valid_toks", baseline=base, dtensor_1=dtensor_1, dtensor_2=dtensor_2, mcore_1=mcore_1, mcore_2=mcore_2)
assert_all_close(2, "train/global_valid_toks", baseline=base, dtensor_2=dtensor_2, mcore_2=mcore_2)
assert_all_close(1, "train/num_unmasked_tokens", baseline=base, dtensor_1=dtensor_1, dtensor_2=dtensor_2, mcore_1=mcore_1, mcore_2=mcore_2)
assert_all_close(2, "train/num_unmasked_tokens", baseline=base, dtensor_2=dtensor_2, mcore_2=mcore_2)
assert_all_close(1, "train/num_valid_samples", baseline=base, dtensor_1=dtensor_1, dtensor_2=dtensor_2, mcore_1=mcore_1, mcore_2=mcore_2)
assert_all_close(2, "train/num_valid_samples", baseline=base, dtensor_2=dtensor_2, mcore_2=mcore_2)

assert_all_close(1, "train/grad_norm", close_args={"rtol": 0.05}, baseline=base, dtensor_1=dtensor_1, dtensor_2=dtensor_2, mcore_1=mcore_1, mcore_2=mcore_2)
assert_all_close(2, "train/grad_norm", close_args={"rtol": 0.05}, baseline=base, dtensor_2=dtensor_2, mcore_2=mcore_2)

assert_all_close(1, "train/loss", close_args={"rtol": 0.05}, baseline=base, dtensor_1=dtensor_1, dtensor_2=dtensor_2, mcore_1=mcore_1, mcore_2=mcore_2)
assert_all_close(2, "train/loss", close_args={"rtol": 0.05}, baseline=base, dtensor_2=dtensor_2, mcore_2=mcore_2)
EOF