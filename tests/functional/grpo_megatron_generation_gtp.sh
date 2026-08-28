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

# GTP refit: training runs TP1 x GTP2 (tensor_parallel_num_weight_shards=2 over
# TP=1), so every training weight is split along dim 0 across the 2 ranks and
# rematerialized on demand. Inference runs TP1 with whole weights, so refit must
# reassemble each weight from its GTP shards -- including the alignment padding
# that carries no logical data. Getting that wrong yields subtly wrong inference
# weights, which shows up as generation/training logprob disagreement, hence the
# token_mult_prob_error gate.
#
# Requires a Megatron-LM with GTP support in megatron/core/resharding
# (NVIDIA/Megatron-LM#6133). Older revisions reject the plan outright.
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
    policy.megatron_cfg.tensor_model_parallel_size=1 \
    policy.megatron_cfg.tensor_parallel_num_weight_shards=2 \
    policy.generation.backend=megatron \
    policy.generation.mcore_generation_config.refit_backend=nccl \
    cluster.gpus_per_node=2 \
    grpo.max_num_steps=2 \
    logger.tensorboard_enabled=true \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=false \
    logger.monitor_gpus=true \
    checkpointing.enabled=false \
    $@ \
    2>&1 | tee $RUN_LOG

uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

uv run tests/check_metrics.py $JSON_METRICS \
    'max(data["train/token_mult_prob_error"]) < 1.05'

# Guard against a vacuous pass. GTP is only active when MCore actually creates
# the gtp_remat process group; if that silently degrades to size 1 the run is an
# ordinary TP1 job that would pass every metric above without testing anything.
if ! grep -q "\[gtp\] enabling GTP weight rematerialization (gtp_remat_size=2" $RUN_LOG; then
    echo "FAIL: GTP marker not found; the training model was not GTP-sharded"
    exit 1
fi

# GTP-sharded training weights must be reassembled into a dedicated inference
# model. Without this, generation would run on the training model directly and
# the refit path under test would never execute.
if ! grep -q "\[colocated-reshard\] building dedicated inference model" $RUN_LOG; then
    echo "FAIL: colocated-reshard marker not found; dedicated inference model was never built"
    exit 1
fi
