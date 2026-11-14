#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=1
GPUS_PER_NODE=8
STEPS_PER_RUN=100
MAX_STEPS=100
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
# Small model; nemo_gym server startup + vLLM warmup add to the 100 steps.
NUM_MINUTES=45
# ===== END CONFIG =====

exit_if_max_steps_reached

cd $PROJECT_ROOT

# Prepare the nemo-gym workplace_assistant dataset (mirrors the nanov3 Gym nightly).
DATA_DIR=$EXP_DIR/data
mkdir -p $DATA_DIR
cd 3rdparty/Gym-workspace/Gym
if [[ ! -f env.yaml ]]; then
    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo "[ERROR] HF_TOKEN is not set"
        exit 1
    fi
    echo "hf_token: $HF_TOKEN" >> env.yaml
fi
uv run ng_prepare_data "+config_paths=[resources_servers/workplace_assistant/configs/workplace_assistant.yaml]" \
    +output_dirpath=data/workplace_assistant \
    +mode=train_preparation \
    +should_download=true \
    +data_source=huggingface
cd -

# Trim to a single tool so the first prompt fits the sequence budget.
TRAIN_PATH=$DATA_DIR/workplace_assistant_train.jsonl
VALIDATION_PATH=$DATA_DIR/workplace_assistant_validation.jsonl
jq -c '.responses_create_params.tools |= (.[0:1])' 3rdparty/Gym-workspace/Gym/data/workplace_assistant/train.jsonl > $TRAIN_PATH
jq -c '.responses_create_params.tools |= (.[0:1])' 3rdparty/Gym-workspace/Gym/data/workplace_assistant/validation.jsonl > $VALIDATION_PATH

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
    data.train.data_path=$TRAIN_PATH \
    data.validation.data_path=$VALIDATION_PATH \
    $@ \
    2>&1 | tee $RUN_LOG

uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# length_penalty/* is the signal that the penalty ran on the no-capture rollout
# path. Thresholds are set from a measured 100-step run (reward mean 0.15,
# adjusted_frac peaked at 0.375, reward_delta_mean at machine epsilon, vLLM decode
# length fell 35 -> 3):
#  - adjusted_frac >= 0.125: a step rewrote >=1/8 of its rollouts' rewards.
#  - skipped_non_binary / wiped_correct == 0: the verifier reward is binary and
#    the zero-centred penalty never drove a correct rollout to 0.
#  - |reward_delta_mean| < 1e-6: the penalty is zero-centred, so the group-mean
#    reward it publishes is unchanged -- the correctness check.
#  - last-quartile vLLM decode length < first-quartile: the penalty shortened
#    generations over the run (the point of the feature).
uv run tests/check_metrics.py $JSON_METRICS \
    'median(data["train/gen_kl_error"]) < 0.002' \
    'mean(data["train/reward"]) > 0.14' \
    'max(data["train/length_penalty/adjusted_frac"]) >= 0.125' \
    'max(data["train/length_penalty/skipped_non_binary_frac"]) == 0' \
    'max(data["train/length_penalty/wiped_correct_frac"]) == 0' \
    'max(data["train/length_penalty/reward_delta_mean"]) < 1e-6' \
    'min(data["train/length_penalty/reward_delta_mean"]) > -1e-6' \
    'mean(data["train/vllm/generation_length_mean"], -25) < mean(data["train/vllm/generation_length_mean"], 1, 26)'
