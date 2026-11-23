#!/bin/bash
# Zero train/generation KL on the SingleController token-capture path, with per-token
# NVFP4 routed experts on FlashInfer, on GB200.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=2
GPUS_PER_NODE=4
STEPS_PER_RUN=3
MAX_STEPS=3
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=150
SNAPSHOT_MEGATRON_BRIDGE=1
# ===== END CONFIG =====

exit_if_max_steps_reached

cd $PROJECT_ROOT

# Prepare nemo-gym workplace_assistant dataset (mirrors tests/functional/grpo_async_gym.sh).
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

# Trim to a single tool so first prompt fits.
TRAIN_PATH=$DATA_DIR/workplace_assistant_train.jsonl
VALIDATION_PATH=$DATA_DIR/workplace_assistant_validation.jsonl
jq -c '.responses_create_params.tools |= (.[0:1])' 3rdparty/Gym-workspace/Gym/data/workplace_assistant/train.jsonl > $TRAIN_PATH
jq -c '.responses_create_params.tools |= (.[0:1])' 3rdparty/Gym-workspace/Gym/data/workplace_assistant/validation.jsonl > $VALIDATION_PATH

# Token capture is only wired up on the SingleController path.
uv run --no-sync examples/run_grpo_single_controller.py \
    --config $CONFIG_PATH \
    grpo.max_num_steps=$MAX_STEPS \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=True \
    logger.wandb.project=nemo-rl \
    logger.wandb.name=$EXP_NAME \
    logger.monitor_gpus=True \
    logger.tensorboard_enabled=True \
    checkpointing.enabled=True \
    checkpointing.checkpoint_dir=$CKPT_DIR \
    checkpointing.save_period=3 \
    data.train.data_path=$TRAIN_PATH \
    data.validation.data_path=$VALIDATION_PATH \
    $@ \
    2>&1 | tee $RUN_LOG

uv run --no-sync tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# The recipe must have switched the batch-invariant kernels on.
grep -F -q "[zero_train_gen_mismatch] batch-invariant kernels enabled: backend=te_native collective=ordered flash_attention_version=4" $RUN_LOG

MAX_RECORDED_STEP=$(jq -r 'if has("train/loss") then (."train/loss" | keys | map(tonumber) | max // 0) else 0 end' $JSON_METRICS)
if [[ $MAX_RECORDED_STEP -lt $MAX_STEPS ]]; then
    echo "[ERROR] Expected train/loss through step $MAX_STEPS, found step $MAX_RECORDED_STEP"
    exit 1
fi

# Zero train/generation KL means bitwise agreement: gen_kl_error is exactly 0 and
# token_mult_prob_error is exactly 1 on every step. The finalize/* gates are the
# capture health signal: a rejected or poisoned rollout means capture or recovery lost
# a rollout's tokens.
uv run --no-sync tests/check_metrics.py $JSON_METRICS \
    'all_finite(data["train/loss"])' \
    'all_finite(data["train/grad_norm"])' \
    'min(data["train/gen_kl_error"]) == 0' \
    'max(data["train/gen_kl_error"]) == 0' \
    'min(data["train/token_mult_prob_error"]) == 1' \
    'max(data["train/token_mult_prob_error"]) == 1' \
    'max(data["train/finalize/invalid_row_rate"]) == 0' \
    'max(data["train/finalize/capture_poisoned_rollouts"]) == 0'

# Clean up checkpoint directory after successful run to save space.
rm -rf "$CKPT_DIR"
