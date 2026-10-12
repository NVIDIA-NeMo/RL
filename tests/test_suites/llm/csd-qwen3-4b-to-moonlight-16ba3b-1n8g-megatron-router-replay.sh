#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# This reproducer consumes a persisted student-vLLM rollout rather than a raw
# text corpus. The JSONL messages must retain token_ids and routed_experts.
: "${CSD_STUDENT_TO_TEACHER_TABLE:?Set CSD_STUDENT_TO_TEACHER_TABLE to the Moonlight-to-Qwen3 sub-token table}"
: "${CSD_TEACHER_TO_STUDENT_TABLE:?Set CSD_TEACHER_TO_STUDENT_TABLE to the Qwen3-to-Moonlight sub-token table}"
: "${CSD_ROUTED_ROLLOUTS:?Set CSD_ROUTED_ROLLOUTS to the routed-rollout JSONL}"

MAX_STEPS=${MAX_STEPS:-5}
exit_if_max_steps_reached

cd $PROJECT_ROOT

uv run examples/run_csd.py \
    --config $CONFIG_PATH \
    distillation.max_num_steps=$MAX_STEPS \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=True \
    logger.wandb.project=nemo-rl \
    logger.wandb.name=$EXP_NAME \
    logger.monitor_gpus=True \
    logger.tensorboard_enabled=True \
    checkpointing.enabled=False \
    $@ \
    2>&1 | tee $RUN_LOG
