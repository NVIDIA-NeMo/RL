#!/bin/bash

# Functional test for QARL: 1 step of QA-Distillation followed by exporting
# the resulting quantized Megatron checkpoint to HuggingFace format. Covers
# the brittle export path (modelopt -> Megatron-Bridge -> HF safetensors)
# referenced in docs/guides/quantization-aware-rl.md.

# clean up checkpoint and export directories on exit
trap "rm -rf /tmp/qa_distillation_checkpoints" EXIT

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath $SCRIPT_DIR/../..)
# Mark the current repo as safe, since wandb fetches metadata about the repo
git config --global --add safe.directory $PROJECT_ROOT

set -euo pipefail

EXP_NAME=$(basename $0 .sh)
EXP_DIR=$SCRIPT_DIR/$EXP_NAME
LOG_DIR=$EXP_DIR/logs
JSON_METRICS=$EXP_DIR/metrics.json
RUN_LOG=$EXP_DIR/run.log
CKPT_DIR=/tmp/qa_distillation_checkpoints
EXPORT_DIR=$EXP_DIR/hf_export
HF_MODEL=Qwen/Qwen3-0.6B
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

rm -rf $EXP_DIR $LOG_DIR $CKPT_DIR
mkdir -p $EXP_DIR $LOG_DIR

cd $PROJECT_ROOT

# ============================================================================
# Stage 1: 1 step of QA-Distillation (W4A4 / NVFP4_DEFAULT_CFG).
# QAD is "self-distillation" — student is the quantized variant of the
# teacher, so policy.model_name == teacher.model_name (cf. recipe header in
# examples/modelopt/qa_distillation_math_megatron.yaml).
# ============================================================================
uv run coverage run -a --data-file=$PROJECT_ROOT/tests/.coverage --source=$PROJECT_ROOT/nemo_rl \
    $PROJECT_ROOT/examples/run_distillation.py \
    --config $PROJECT_ROOT/examples/modelopt/qa_distillation_math_megatron.yaml \
    policy.model_name=$HF_MODEL \
    teacher.model_name=$HF_MODEL \
    cluster.gpus_per_node=2 \
    policy.train_global_batch_size=16 \
    policy.megatron_cfg.tensor_model_parallel_size=2 \
    policy.megatron_cfg.pipeline_model_parallel_size=1 \
    policy.megatron_cfg.context_parallel_size=1 \
    policy.max_total_sequence_length=2048 \
    policy.quant_calib_size=16 \
    policy.quant_sequence_length=512 \
    teacher.megatron_cfg.tensor_model_parallel_size=2 \
    teacher.megatron_cfg.pipeline_model_parallel_size=1 \
    teacher.megatron_cfg.context_parallel_size=1 \
    distillation.max_num_steps=1 \
    distillation.num_prompts_per_step=16 \
    distillation.max_val_samples=16 \
    distillation.val_batch_size=8 \
    distillation.val_period=1 \
    data.train.dataset_name=OpenMathInstruct-2 \
    ++data.train.split_validation_size=0.05 \
    data.validation=null \
    loss_fn.zero_outside_topk=false \
    logger.tensorboard_enabled=true \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=false \
    logger.monitor_gpus=true \
    checkpointing.enabled=true \
    checkpointing.save_period=1 \
    checkpointing.checkpoint_dir=$CKPT_DIR \
    $@ \
    2>&1 | tee $RUN_LOG

uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

uv run tests/check_metrics.py $JSON_METRICS \
  'data["train/loss"]["1"] > 0'

# ============================================================================
# Stage 2: export the quantized checkpoint to HuggingFace format.
# ============================================================================
WEIGHTS_DIR=$CKPT_DIR/step_1/policy/weights
test -d $WEIGHTS_DIR || { echo "[FAIL] expected checkpoint at $WEIGHTS_DIR"; exit 1; }

rm -rf $EXPORT_DIR
mkdir -p $EXPORT_DIR

uv run --extra mcore --extra modelopt \
    torchrun --nproc_per_node 2 \
    $PROJECT_ROOT/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/examples/quantization/export.py \
    --hf-model-id $HF_MODEL \
    --megatron-load-path $WEIGHTS_DIR \
    --export-dir $EXPORT_DIR \
    --tp 2 --pp 1 \
    2>&1 | tee -a $RUN_LOG

# Sanity-check the export produced expected artifacts
[ -f $EXPORT_DIR/config.json ] || { echo "[FAIL] export missing config.json"; exit 1; }
ls $EXPORT_DIR/*.safetensors > /dev/null 2>&1 || { echo "[FAIL] export missing safetensors"; exit 1; }
echo "[PASS] qa-distillation + export functional test"
