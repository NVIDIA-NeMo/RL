#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
MODEL_NAME=${NRL_NEMOTRON_SUPER_MODEL_PATH}
GEMRM_MODEL_PATH=${NRL_GEMRM_MODEL_PATH}
TRAIN_PATH=${NRL_TRAIN_PATH}
VAL_PATH=${NRL_VAL_PATH}
NUM_NODES=8
GPUS_PER_NODE=8
STEPS_PER_RUN=10
MAX_STEPS=10
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=45
# ===== END CONFIG =====

exit_if_max_steps_reached

# Run the experiment
cd $PROJECT_ROOT
uv run ./examples/nemo_gym/run_grpo_nemo_gym.py \
    --config $CONFIG_PATH \
    policy.model_name=$MODEL_NAME \
    data.train_jsonl_fpath=$TRAIN_PATH \
    data.validation_jsonl_fpath=$VAL_PATH \
    env.nemo_gym.nl2bash_judge_model.responses_api_models.vllm_model.model=$GEMRM_MODEL_PATH \
    grpo.max_num_steps=$MAX_STEPS \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=True \
    logger.wandb.project=nemo-rl \
    logger.wandb.name=$EXP_NAME \
    logger.monitor_gpus=True \
    logger.tensorboard_enabled=True \
    checkpointing.enabled=True \
    checkpointing.checkpoint_dir=$CKPT_DIR \
    $@ \
    2>&1 | tee $RUN_LOG

# Convert tensorboard logs to json
uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS
