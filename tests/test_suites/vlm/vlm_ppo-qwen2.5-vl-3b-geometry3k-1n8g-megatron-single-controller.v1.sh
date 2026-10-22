#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=1
STEPS_PER_RUN=50
MAX_STEPS=50
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=60
# ===== END CONFIG =====

exit_if_max_steps_reached

cd $PROJECT_ROOT
uv run examples/run_grpo_single_controller.py \
    --config $CONFIG_PATH \
    ppo.max_num_steps=$MAX_STEPS \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=True \
    logger.wandb.project=nemo-rl \
    logger.wandb.name=$EXP_NAME \
    logger.monitor_gpus=True \
    logger.tensorboard_enabled=True \
    checkpointing.enabled=True \
    checkpointing.checkpoint_dir=$CKPT_DIR \
    data_plane.observability.verify_tensor_hash=True \
    "$@" \
    2>&1 | tee $RUN_LOG

# The value worker emits this marker only after packed media reaches its input.
grep -q "MegatronValueWorker multimodal input fields:.*pixel_values" $RUN_LOG

uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

if [[ $(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | max' $JSON_METRICS) -ge $MAX_STEPS ]]; then
    uv run tests/check_metrics.py $JSON_METRICS \
        'all_finite(data["train/loss"])' \
        'all_finite(data["train/grad_norm"])' \
        'all_finite(data["train/critic/loss"])' \
        'all_finite(data["train/critic/grad_norm"])' \
        'all_finite(data["train/reward"])' \
        'median(data["train/token_mult_prob_error"]) < 1.1' \
        'max(data["train/policy_kl_error"]) < 0.02' \
        'len(data["train/critic/loss"]) == 50' \
        'max(data["train/critic/loss"]) < 1.5' \
        'mean(data["train/critic/loss"], range_start=-10) < 0.05' \
        'mean(data["train/critic/explained_var"], range_start=-10) > -1.5' \
        'mean(data["train/reward"], range_start=-10) > 0.1' \
        'max({**data.get("data_plane/cluster/step/hash/mismatches", {}), **data.get("data_plane/driver/step/hash/mismatches", {})}) == 0' \
        'max({**data.get("data_plane/cluster/step/hash/rows_checked", {}), **data.get("data_plane/driver/step/hash/rows_checked", {})}) > 0'

    rm -rf "$CKPT_DIR"
fi
