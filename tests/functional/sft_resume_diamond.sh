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

# This test will run for 4 steps and make sure that 2+2 steps w/ resume leads to the same result

prefix_output() {
  sed "s/^/$1/"
}

train_cmd() {
uv run coverage run -a --data-file=$PROJECT_ROOT/tests/.coverage --source=$PROJECT_ROOT/nemo_rl \
    $PROJECT_ROOT/examples/run_sft.py \
    policy.model_name=Qwen/Qwen3-0.6B \
    sft.val_period=2 \
    sft.val_batches=2 \
    sft.val_global_batch_size=2 \
    policy.train_global_batch_size=4 \
    policy.train_micro_batch_size=1 \
    cluster.gpus_per_node=1 \
    logger.tensorboard_enabled=true \
    logger.wandb_enabled=false \
    logger.monitor_gpus=true \
    checkpointing.enabled=false \
    checkpointing.save_period=2 \
    $@ 
}

cd $PROJECT_ROOT

# Dtensor 4 step baseline
train_cmd logger.log_dir=$LOG_DIR/baseline sft.max_num_steps=4 policy.dtensor_cfg.enabled=true policy.megatron_cfg.enabled=false $@ 2>&1 | prefix_output "[baseline 4step] " | tee ${RUN_LOG}.4step_baseline
uv run tests/json_dump_tb_logs.py $LOG_DIR/baseline --output_path $EXP_DIR/baseline.json
# Dtensor 2+2 step
train_cmd logger.log_dir=$LOG_DIR/dtensor sft.max_num_steps=2 checkpointing.enabled=true checkpointing.checkpoint_dir=$CKPT_DIR/dtensor policy.dtensor_cfg.enabled=true policy.megatron_cfg.enabled=false $@ 2>&1 | prefix_output "[dtensor 2step] " | tee ${RUN_LOG}.dtensor_2step
uv run tests/json_dump_tb_logs.py $LOG_DIR/dtensor --output_path $EXP_DIR/dtensor_2step.json
train_cmd logger.log_dir=$LOG_DIR/dtensor sft.max_num_steps=4 checkpointing.enabled=true checkpointing.checkpoint_dir=$CKPT_DIR/dtensor policy.dtensor_cfg.enabled=true policy.megatron_cfg.enabled=false $@ 2>&1 | prefix_output "[dtensor 4step] " | tee ${RUN_LOG}.dtensor_4step
uv run tests/json_dump_tb_logs.py $LOG_DIR/dtensor --output_path $EXP_DIR/dtensor_4step.json
# Mcore 2+2 step
train_cmd logger.log_dir=$LOG_DIR/mcore sft.max_num_steps=2 checkpointing.enabled=true checkpointing.checkpoint_dir=$CKPT_DIR/mcore policy.dtensor_cfg.enabled=false policy.megatron_cfg.enabled=true $@ 2>&1 | prefix_output "[mcore 2step] " | tee ${RUN_LOG}.mcore_2step
uv run tests/json_dump_tb_logs.py $LOG_DIR/mcore --output_path $EXP_DIR/mcore_2step.json
train_cmd logger.log_dir=$LOG_DIR/mcore sft.max_num_steps=4 checkpointing.enabled=true checkpointing.checkpoint_dir=$CKPT_DIR/mcore policy.dtensor_cfg.enabled=false policy.megatron_cfg.enabled=true $@ 2>&1 | prefix_output "[mcore 4step] " | tee ${RUN_LOG}.mcore_4step
uv run tests/json_dump_tb_logs.py $LOG_DIR/mcore --output_path $EXP_DIR/mcore_4step.json


#uv run tests/check_metrics.py $JSON_METRICS \
#    'max(data["train/token_mult_prob_error"]) < 1.05'

