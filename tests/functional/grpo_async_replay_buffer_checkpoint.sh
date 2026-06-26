#!/bin/bash
# Verifies that async GRPO (non-colocated) saves and restores the replay
# buffer checkpoint across a training restart.

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath $SCRIPT_DIR/../..)
git config --global --add safe.directory $PROJECT_ROOT

set -eou pipefail

EXP_NAME=$(basename $0 .sh)
EXP_DIR=$SCRIPT_DIR/$EXP_NAME
LOG_DIR=$EXP_DIR/logs
CKPT_DIR=$EXP_DIR/ckpts
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

rm -rf $EXP_DIR
mkdir -p $EXP_DIR $LOG_DIR $CKPT_DIR

TRAIN_CMD=(
    uv run coverage run -a --data-file=$PROJECT_ROOT/tests/.coverage --source=$PROJECT_ROOT/nemo_rl
    $PROJECT_ROOT/examples/run_grpo.py
    policy.model_name=Qwen/Qwen3-0.6B
    grpo.num_prompts_per_step=2
    grpo.num_generations_per_prompt=4
    policy.train_global_batch_size=4
    policy.train_micro_batch_size=1
    policy.generation.colocated.enabled=false
    policy.generation.colocated.resources.gpus_per_node=1
    policy.generation.colocated.resources.num_nodes=1
    policy.generation.vllm_cfg.async_engine=true
    grpo.async_grpo.enabled=true
    grpo.async_grpo.max_trajectory_age_steps=1
    grpo.async_grpo.in_flight_weight_updates=false
    loss_fn.use_importance_sampling_correction=true
    cluster.gpus_per_node=2
    logger.tensorboard_enabled=false
    logger.wandb_enabled=false
    checkpointing.enabled=true
    checkpointing.checkpoint_dir=$CKPT_DIR
    checkpointing.save_period=2
)

cd $PROJECT_ROOT

# --- Run 1: train 2 steps and checkpoint ---
echo "=== Run 1: steps 1-2 ==="
"${TRAIN_CMD[@]}" \
    grpo.max_num_steps=2 \
    logger.log_dir=$LOG_DIR/run1 \
    $@ \
    2>&1 | tee $EXP_DIR/run1.log

REPLAY_BUFFER_PT=$CKPT_DIR/step_2/replay_buffer.pt
if [[ ! -f "$REPLAY_BUFFER_PT" ]]; then
    echo "FAIL: replay_buffer.pt not found at $REPLAY_BUFFER_PT"
    exit 1
fi
echo "✅ replay_buffer.pt saved: $REPLAY_BUFFER_PT"

# --- Run 2: resume from checkpoint and train 2 more steps ---
echo "=== Run 2: resuming, steps 3-4 ==="
"${TRAIN_CMD[@]}" \
    grpo.max_num_steps=4 \
    logger.log_dir=$LOG_DIR/run2 \
    $@ \
    2>&1 | tee $EXP_DIR/run2.log

if ! grep -q "Restoring replay buffer from checkpoint" $EXP_DIR/run2.log; then
    echo "FAIL: replay buffer restore log line not found in run2 output"
    exit 1
fi
echo "✅ Replay buffer restored from checkpoint"

if ! grep -q "ReplayBuffer restored:" $EXP_DIR/run2.log; then
    echo "FAIL: ReplayBuffer restored summary line not found in run2 output"
    exit 1
fi

if ! grep -q "train_data_step4.jsonl" $EXP_DIR/run2.log; then
    echo "FAIL: run2 did not reach training step 4"
    exit 1
fi

echo "✅ grpo_async_replay_buffer_checkpoint passed"
