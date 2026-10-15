#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=1
STEPS_PER_RUN=5
MAX_STEPS=5
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=30
# ===== END CONFIG =====

exit_if_max_steps_reached

cd $PROJECT_ROOT

# This reproducer consumes a persisted student-vLLM rollout rather than a raw
# text corpus. The JSONL messages must retain token_ids and routed_experts.
: "${CSD_STUDENT_TO_TEACHER_TABLE:?Set CSD_STUDENT_TO_TEACHER_TABLE to the Moonlight-to-Qwen3 sub-token table}"
: "${CSD_TEACHER_TO_STUDENT_TABLE:?Set CSD_TEACHER_TO_STUDENT_TABLE to the Qwen3-to-Moonlight sub-token table}"
: "${CSD_ROUTED_ROLLOUTS:?Set CSD_ROUTED_ROLLOUTS to the routed-rollout JSONL}"

uv run examples/run_csd.py \
    --config $CONFIG_PATH \
    distillation.max_num_steps=$MAX_STEPS \
    teachers.0.aligner.student_to_teacher_table_path="$CSD_STUDENT_TO_TEACHER_TABLE" \
    teachers.0.aligner.teacher_to_student_table_path="$CSD_TEACHER_TO_STUDENT_TABLE" \
    data.train.data_path="$CSD_ROUTED_ROLLOUTS" \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=True \
    logger.wandb.project=nemo-rl \
    logger.wandb.name=$EXP_NAME \
    logger.monitor_gpus=True \
    logger.tensorboard_enabled=True \
    checkpointing.enabled=False \
    $@ \
    2>&1 | tee $RUN_LOG
