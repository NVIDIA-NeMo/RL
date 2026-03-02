#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
MODEL_NAME=${NRL_NEMOTRON_SUPER_MODEL_PATH}
GENRM_MODEL_PATH=${NRL_GENRM_MODEL_PATH}
NL2BASH_JUDGE_MODEL_PATH=${NRL_NL2BASH_JUDGE_MODEL_PATH}
TRAIN_PATH=${NRL_TRAIN_PATH}
VAL_PATH=${NRL_VAL_PATH}
NUM_NODES=20
GPUS_PER_NODE=8
STEPS_PER_RUN=15
MAX_STEPS=15
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=60
# ===== END CONFIG =====

exit_if_max_steps_reached

# Run the experiment
cd $PROJECT_ROOT

export NRL_VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

uv run ./examples/nemo_gym/run_grpo_nemo_gym.py \
    --config $CONFIG_PATH \
    policy.model_name=$MODEL_NAME \
    data.train_jsonl_fpath=$TRAIN_PATH \
    data.validation_jsonl_fpath=$VAL_PATH \
    env.nemo_gym.nl2bash_judge_model.responses_api_models.vllm_model.model=$NL2BASH_JUDGE_MODEL_PATH \
    env.nemo_gym.genrm_model.responses_api_models.vllm_model.model=$GENRM_MODEL_PATH \
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
