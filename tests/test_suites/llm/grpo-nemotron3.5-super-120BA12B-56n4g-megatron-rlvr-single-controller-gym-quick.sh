#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
# 48 NeMo RL nodes (16 training, 32 generation) plus 8 nodes in a second hetgroup for the
# GenRM, NL2Bash and safety judge pools in the .pools file next to this script.
NUM_NODES=48
NUM_EXTERNAL_SERVICE_NODES=8
GPUS_PER_NODE=4
SEGMENT_SIZE=16
EXTERNAL_VLLM_SEGMENT_SIZE=2
STEPS_PER_RUN=3
MAX_STEPS=3
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
# On HSG this shape had Ray and the judge pools up in ~7 min and finished 3 steps in 47 min.
NUM_MINUTES=120
USE_GYM_CONTAINER=true
USES_SANDBOX=1
JOB_REAPER_COMMENT='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"90","reason":"other","description":"Super 3.5 GRPO with external judge pools idles GPUs during model loading"}}'
# ===== END CONFIG =====

exit_if_max_steps_reached

cd $PROJECT_ROOT

ASSETS_DIR=$HF_HOME/nemotron3.5-super-rlvr

# Competitive coding compiles in a directory that the sandbox container also mounts
# (tools/launch mounts the code snapshot, which holds EXP_DIR, into the sandbox).
export CCC_TEST_FILE=$ASSETS_DIR/ccc/ccc_v2.1_RL_all_competitions_full_metadata.jsonl
export CCC_SHARED_TEMP_DIR=$EXP_DIR/ccc_shared_tmp
mkdir -p $CCC_SHARED_TEMP_DIR

# Runtime settings of the journey launcher this recipe was validated with.
export NRL_VLLM_ASYNC_TIMEOUT_SECONDS=3600
export NRL_WG_USE_RAY_REF=1
export NRL_USE_FASTOKENS=1
export VLLM_USE_FLASHINFER_MOE_FP8=1
export VLLM_FLASHINFER_MOE_BACKEND=latency
export WANDB_INIT_TIMEOUT=300

uv run examples/run_grpo_single_controller.py \
    --config $CONFIG_PATH \
    grpo.max_num_steps=$MAX_STEPS \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=True \
    logger.wandb.project=nemo-rl \
    logger.wandb.name=$EXP_NAME \
    logger.monitor_gpus=True \
    logger.tensorboard_enabled=True \
    checkpointing.enabled=False \
    checkpointing.checkpoint_dir=$CKPT_DIR \
    $@ \
    2>&1 | tee $RUN_LOG

uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# On HSG this shape gave token_mult_prob_error 1.021-1.027 and gen_kl_error 0.0010-0.0014.
# The GenRM check fails if the external GenRM pool never scores a rollout.
uv run tests/check_metrics.py $JSON_METRICS \
    'data["train/token_mult_prob_error"]["3"] < 1.1' \
    'max(data["train/gen_kl_error"]) < 0.005' \
    'max(data["train/reward"]) > 0.0' \
    'max(data["train/genrm_simple_agent/reward/mean"]) > 0.0'
