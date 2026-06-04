#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath $SCRIPT_DIR/../..)
# Mark the current repo as safe, since wandb fetches metadata about the repo
git config --global --add safe.directory $PROJECT_ROOT

set -euo pipefail

EXP_NAME=$(basename $0 .sh)
EXP_DIR=$SCRIPT_DIR/$EXP_NAME
LOG_DIR=$EXP_DIR/logs
PROJ_DIR=$EXP_DIR/projection
PROJ_PATH=$PROJ_DIR/xtoken_l1_smoke_special.pt
JSON_METRICS=$EXP_DIR/metrics.json
RUN_LOG=$EXP_DIR/run.log
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

rm -rf $EXP_DIR $LOG_DIR
mkdir -p $EXP_DIR $LOG_DIR $PROJ_DIR

cd $PROJECT_ROOT

# Step 1: build a runtime projection matrix for the (student, teacher) pair.
# Student is the YAML default (Llama-3.2-1B); teacher is scaled down from the
# YAML default (Qwen3-4B) to Qwen3-1.7B so both replicas fit on 2 GPUs.
# We invoke minimal_projection_via_multitoken directly (mirroring the PT
# reference workflow) instead of the build_projection_matrix.sh orchestrator
# so we skip the ~15-min Qwen3-Embedding-4B encoding pass that Step 1 of the
# orchestrator does — the embedding seed is optional, and --enable-exact-match
# constructs a usable matrix from tokenizer surface forms directly.
STUDENT_MODEL=meta-llama/Llama-3.2-1B
TEACHER_MODEL=Qwen/Qwen3-1.7B
uv run python -m tools.x_token.minimal_projection_via_multitoken \
    --student-model $STUDENT_MODEL \
    --teacher-model $TEACHER_MODEL \
    --top-k 4 \
    --enable-special-token-mapping \
    --enable-exact-match \
    --disable-reverse-pass \
    --disable-scale-trick \
    --output-filename xtoken_l1_smoke \
    --output-dir $PROJ_DIR

# Step 2: run the xtoken off-policy distillation trainer for 3 steps. Only
# overrides that DIFFER from examples/configs/xtoken_off_policy_distillation.yaml
# are set here.
uv run coverage run -a --data-file=$PROJECT_ROOT/tests/.coverage --source=$PROJECT_ROOT/nemo_rl \
    $PROJECT_ROOT/examples/run_xtoken_off_policy_distillation.py \
    teacher.model_name=$TEACHER_MODEL \
    teacher.tokenizer.name=$TEACHER_MODEL \
    cluster.gpus_per_node=2 \
    policy.train_global_batch_size=8 \
    policy.max_total_sequence_length=256 \
    teacher.train_global_batch_size=8 \
    teacher.max_total_sequence_length=256 \
    distillation.num_prompts_per_step=8 \
    distillation.max_num_steps=3 \
    loss_fn.projection_matrix_path=$PROJ_PATH \
    data.train.characters_per_sample=256 \
    logger.tensorboard_enabled=true \
    logger.log_dir=$LOG_DIR \
    $@ \
    2>&1 | tee $RUN_LOG

uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

uv run tests/check_metrics.py $JSON_METRICS \
  'data["train/loss"]["3"] < 5'
