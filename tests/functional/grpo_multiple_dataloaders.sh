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

rm -rf $EXP_DIR $LOG_DIR
mkdir -p $EXP_DIR $LOG_DIR

# This test will run for 2 steps and make sure that 1+1 steps w/ resume leads to the same result.
# We use the checkpointing.checkpoint_must_save_by=0:0:0:1 feature to exit after 1 step.

prefix_output() {
  sed "s/^/$1/"
}

train_cmd() {
uv run coverage run -a --data-file=$PROJECT_ROOT/tests/.coverage --source=$PROJECT_ROOT/nemo_rl \
    $PROJECT_ROOT/examples/run_grpo.py \
    --config $PROJECT_ROOT/examples/configs/grpo_multiple_datasets.yaml \
    policy.model_name=Qwen/Qwen3-0.6B \
    data.use_multiple_dataloader=true \
    data.num_prompts_per_dataloader=1 \
    data.custom_dataloader=examples.custom_dataloader.custom_dataloader.example_custom_dataloader \
    grpo.val_at_start=true \
    grpo.max_val_samples=4 \
    grpo.val_batch_size=4 \
    grpo.num_prompts_per_step=2 \
    grpo.num_generations_per_prompt=4 \
    policy.train_global_batch_size=4 \
    policy.train_micro_batch_size=1 \
    cluster.gpus_per_node=2 \
    grpo.max_num_steps=2 \
    logger.tensorboard_enabled=true \
    logger.wandb_enabled=false \
    logger.monitor_gpus=true \
    checkpointing.enabled=false \
    checkpointing.save_period=1 \
    $@
}

cd $PROJECT_ROOT

# 2 step baseline
train_cmd logger.log_dir=$LOG_DIR/baseline $@ 2>&1 | prefix_output "[baseline 2step] " | tee ${RUN_LOG}.2step_baseline
uv run tests/json_dump_tb_logs.py $LOG_DIR/baseline --output_path $EXP_DIR/baseline.json
# 1+1 step
train_cmd logger.log_dir=$LOG_DIR/resume checkpointing.checkpoint_must_save_by=0:0:0:1 checkpointing.enabled=true checkpointing.checkpoint_dir=$CKPT_DIR/resume $@ 2>&1 | prefix_output "[resume 1step] " | tee ${RUN_LOG}.resume_1step
uv run tests/json_dump_tb_logs.py $LOG_DIR/resume --output_path $EXP_DIR/resume_1step.json
train_cmd logger.log_dir=$LOG_DIR/resume checkpointing.enabled=true checkpointing.checkpoint_dir=$CKPT_DIR/resume $@ 2>&1 | prefix_output "[resume 2step] " | tee ${RUN_LOG}.resume_2step
uv run tests/json_dump_tb_logs.py $LOG_DIR/resume --output_path $EXP_DIR/resume_2step.json

uv run python - <<EOF $EXP_DIR/baseline.json $EXP_DIR/resume_1step.json $EXP_DIR/resume_2step.json
import sys
import json
import numpy as np

baseline_json, resume_1step_json, resume_2step_json = sys.argv[1:4]

with open(baseline_json) as f:
    base = json.load(f)
with open(resume_1step_json) as f:
    resume_1 = json.load(f)
with open(resume_2step_json) as f:
    resume_2 = json.load(f)

def assert_all_close(i, name, close_args={}, **kwargs):
    baseline = kwargs["baseline"][name][str(i)]
    for test_name, test_data in kwargs.items():
        val = test_data[name][str(i)]
        assert np.isclose(val, baseline, **close_args), f"{test_name}[{repr(name)}][{repr(i)}] ({val}) != baseline[{repr(name)}][{repr(i)}] ({baseline})"
        print(f"{test_name}[{repr(name)}][{repr(i)}] ({val}) == baseline[{repr(name)}][{repr(i)}] ({baseline})")
    print(f"✓ {name} {i} is equal")

assert_all_close(1, "train/lr", baseline=base, resume_1=resume_1, resume_2=resume_2)
assert_all_close(2, "train/lr", baseline=base, resume_2=resume_2)
assert_all_close(1, "train/mean_prompt_length", baseline=base, resume_1=resume_1, resume_2=resume_2)
assert_all_close(2, "train/mean_prompt_length", baseline=base, resume_2=resume_2)

assert max(base["train/token_mult_prob_error"].values()) < 1.05
assert max(resume_1["train/token_mult_prob_error"].values()) < 1.05
assert max(resume_2["train/token_mult_prob_error"].values()) < 1.05
EOF
