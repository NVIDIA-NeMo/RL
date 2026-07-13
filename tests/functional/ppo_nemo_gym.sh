#!/bin/bash
# PPO + NeMo-Gym (Math / RLVR) smoke test. Colocated + synchronous PPO, which
# exercises the three new gym paths: setup() gym spinup (vLLM deferred load),
# ppo_train() gym rollout, and validate() gym dispatch. See grpo_async_gym.sh
# for the GRPO analog and the NeMo-Gym data-prep prerequisites.

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

# Follow the NeMo-Gym instructions to obtain the Gym workspace + data:
# https://docs.nvidia.com/nemo/gym/0.1.0/tutorials/nemo-rl-grpo/setup.html
cd 3rdparty/Gym-workspace/Gym

# We need HF_TOKEN to download the data from huggingface
if [[ ! -f env.yaml ]]; then
    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo "[ERROR] HF_TOKEN is not set"
        exit 1
    fi
    echo "hf_token: $HF_TOKEN" >> env.yaml
fi

uv run ng_prepare_data "+config_paths=[resources_servers/math_with_judge/configs/math_with_judge.yaml]" \
    +output_dirpath=data/math_with_judge \
    +mode=train_preparation \
    +should_download=true \
    +data_source=huggingface
cd -

TRAIN_PATH=$DATA_DIR/math_with_judge_train.jsonl
VALIDATION_PATH=$DATA_DIR/math_with_judge_validation.jsonl
cp 3rdparty/Gym-workspace/Gym/data/math_with_judge/train.jsonl $TRAIN_PATH
cp 3rdparty/Gym-workspace/Gym/data/math_with_judge/validation.jsonl $VALIDATION_PATH

uv run coverage run -a --data-file=$PROJECT_ROOT/tests/.coverage --source=$PROJECT_ROOT/nemo_rl \
    $PROJECT_ROOT/examples/nemo_gym/run_ppo_nemo_gym.py \
    --config $PROJECT_ROOT/examples/nemo_gym/ppo_math_rlvr_nemo_gym.yaml \
    policy.model_name=Qwen/Qwen3-0.6B \
    policy.dtensor_cfg.enabled=true \
    policy.megatron_cfg.enabled=false \
    value.model_name=Qwen/Qwen3-0.6B \
    value.dtensor_cfg.enabled=true \
    value.megatron_cfg.enabled=false \
    policy.generation.vllm_cfg.tensor_parallel_size=1 \
    policy.generation.vllm_cfg.async_engine=true \
    policy.generation.vllm_cfg.expose_http_server=true \
    policy.max_total_sequence_length=512 \
    policy.generation.colocated.enabled=true \
    ppo.num_prompts_per_step=4 \
    ppo.num_generations_per_prompt=2 \
    ppo.max_num_steps=10 \
    ppo.val_period=5 \
    ppo.policy_training_start_step=0 \
    ppo.reward_scaling.enabled=false \
    ppo.reward_shaping.enabled=false \
    policy.train_global_batch_size=4 \
    policy.train_micro_batch_size=1 \
    cluster.gpus_per_node=2 \
    loss_fn.use_importance_sampling_correction=true \
    logger.tensorboard_enabled=true \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=false \
    logger.monitor_gpus=true \
    checkpointing.enabled=true \
    checkpointing.save_period=5 \
    checkpointing.checkpoint_dir=$CHECKPOINT_DIR \
    data.train.data_path=$TRAIN_PATH \
    data.validation.data_path=$VALIDATION_PATH \
    $@ \
    2>&1 | tee $RUN_LOG

uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# Smoke thresholds: mismatch diagnostic stays bounded and the critic learns a
# finite value loss (explained_variance is logged by PPO's value stage).
uv run tests/check_metrics.py $JSON_METRICS \
    'median(data["train/gen_kl_error"]) < 1.3' \
    'data["validation/accuracy"]["10"] >= 0.0'
