#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=1
MAX_STEPS=100
NUM_MINUTES=30
STUDENT_MODEL=meta-llama/Llama-3.2-1B
TEACHER0_MODEL=Qwen/Qwen3-4B            # cross-tokenizer (P-KL via projection)
TEACHER1_MODEL=meta-llama/Llama-3.2-3B  # same tokenizer/vocab as student
# ===== END CONFIG =====

exit_if_max_steps_reached

cd $PROJECT_ROOT

# Prepare both complete token-chain directions for the cross-tokenizer teacher.
: "${CSD_TRAIN_DATA:?Set CSD_TRAIN_DATA to a text JSONL corpus fitting both tokenizers}"
TABLE_DIR="$EXP_DIR/csd_tables"
mkdir -p "$TABLE_DIR"
export CSD_STUDENT_TO_TEACHER_TABLE="$TABLE_DIR/student_to_teacher.pt"
export CSD_TEACHER_TO_STUDENT_TABLE="$TABLE_DIR/teacher_to_student.pt"
uv run python -m tools.csd.build_subtoks_table \
    --source-tokenizer "$STUDENT_MODEL" --destination-tokenizer "$TEACHER0_MODEL" \
    --output "$CSD_STUDENT_TO_TEACHER_TABLE" --overwrite
uv run python -m tools.csd.build_subtoks_table \
    --source-tokenizer "$TEACHER0_MODEL" --destination-tokenizer "$STUDENT_MODEL" \
    --output "$CSD_TEACHER_TO_STUDENT_TABLE" --overwrite

uv run examples/run_csd.py \
    --config $CONFIG_PATH \
    distillation.max_num_steps=$MAX_STEPS \
    data.train.data_files="$CSD_TRAIN_DATA" \
    data.train.characters_per_sample=null \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=True \
    logger.wandb.project=nemo-rl \
    logger.wandb.name=$EXP_NAME \
    logger.monitor_gpus=True \
    logger.tensorboard_enabled=True \
    checkpointing.enabled=False \
    $@ \
    2>&1 | tee $RUN_LOG

# Convert tensorboard logs to json
uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# Smoke qualification checks finiteness; it does not assert benchmark convergence.
uv run tests/check_metrics.py "$JSON_METRICS" 'all_finite(data["train/loss"])'
