#!/bin/bash
# SingleController + NeMo-Gym + Megatron generation e2e smoke.
# Mirrors grpo_async_gym_single_controller.sh with the generation backend
# swapped to non-colocated Megatron Inference. The config is loaded from the
# exemplar YAML, which resolves value-equal to that test's CLI monolith
# everywhere but the generation backend.

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
CHECKPOINT_DIR=$EXP_DIR/checkpoints
DATA_DIR=$EXP_DIR/data
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

rm -rf $EXP_DIR $LOG_DIR
mkdir -p $EXP_DIR $LOG_DIR $CHECKPOINT_DIR $DATA_DIR

# clean up checkpoint directory on exit
trap "rm -rf $CHECKPOINT_DIR" EXIT

cd $PROJECT_ROOT

# Follow nemo-gym instructions here to get this data:
# https://docs.nvidia.com/nemo/gym/0.1.0/tutorials/nemo-rl-grpo/setup.html#training-nemo-rl-grpo-setup
cd 3rdparty/Gym-workspace/Gym

# We need HF_TOKEN to download the data from huggingface
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

# This trimming of the workplace assistant dataset is necessary b/c with all the tools the first prompt is >4000 tokens
# which will cause the generation engine to return nothing on the first prompt and crash RL. Since we want to keep this test short to
# smoke test, we trim all but the first tool
TRAIN_PATH=$DATA_DIR/workplace_assistant_train.jsonl
VALIDATION_PATH=$DATA_DIR/workplace_assistant_validation.jsonl
jq -c '.responses_create_params.tools |= (.[0:1])' 3rdparty/Gym-workspace/Gym/data/workplace_assistant/train.jsonl > $TRAIN_PATH
jq -c '.responses_create_params.tools |= (.[0:1])' 3rdparty/Gym-workspace/Gym/data/workplace_assistant/validation.jsonl > $VALIDATION_PATH

uv run coverage run -a --data-file=$PROJECT_ROOT/tests/.coverage --source=$PROJECT_ROOT/nemo_rl \
    $PROJECT_ROOT/examples/run_grpo_single_controller.py \
    --config $PROJECT_ROOT/examples/nemo_gym/grpo_qwen3_0_6b_megatron_generation_single_controller.yaml \
    policy.generation.max_new_tokens=128 \
    policy.max_total_sequence_length=512 \
    logger.tensorboard_enabled=true \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=false \
    logger.monitor_gpus=true \
    data.train.data_path=$TRAIN_PATH \
    data.validation.data_path=$VALIDATION_PATH \
    $@ \
    2>&1 | tee $RUN_LOG

uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# Lag-0 run: strict engine/trainer token parity on top of the standard gym gates
uv run tests/check_metrics.py $JSON_METRICS \
    'max(data["train/token_mult_prob_error"]) < 1.05' \
    'median(data["train/gen_kl_error"]) < 1.3' \
    'max(data["train/reward"]) > 0'
