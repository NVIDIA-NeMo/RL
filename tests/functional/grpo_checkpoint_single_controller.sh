#!/bin/bash
# Verifies checkpoint save/restore MECHANICS only. Metric equivalence across
# resume is not well-defined here: prompts in flight at save time are dropped
# while the dataloader cursor has already advanced past them (not exact-once),
# so a resumed run does not consume the same prompts as an uninterrupted one.
#
# Covered: the SingleController (SC) path saves and restores a full
# checkpoint across a training restart: model weights/optimizer + training
# state, dataloader position, and the replay buffer. Two runs share one
# checkpoint_dir and one config (same max_num_steps, so Megatron train_iters
# stays consistent across the restart):
#   Run 1 sets a tiny checkpointing.checkpoint_must_save_by so the timeout
#   save fires at the first step boundary and training stops early (also
#   exercising the timeout-save path).
#   Run 2 drops the timeout and must resume from step_1 to max_num_steps.
# Same shape as grpo_dp_single_controller.sh (Qwen3-0.6B, 2 GPUs) with the
# two-run structure of grpo_async_replay_buffer_checkpoint.sh.

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath $SCRIPT_DIR/../..)
# Mark the current repo as safe, since wandb fetches metadata about the repo
git config --global --add safe.directory $PROJECT_ROOT

set -eou pipefail

EXP_NAME=$(basename $0 .sh)
EXP_DIR=$SCRIPT_DIR/$EXP_NAME
LOG_DIR=$EXP_DIR/logs
CKPT_DIR=$EXP_DIR/ckpts
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

rm -rf $EXP_DIR
mkdir -p $EXP_DIR $LOG_DIR $CKPT_DIR

# The windowed (over-sampled) sampler; its per-policy key is not in the
# exemplar's in_order sampler block, hence the '+' append prefix.
TRAIN_CMD=(
    uv run coverage run -a --data-file=$PROJECT_ROOT/tests/.coverage --source=$PROJECT_ROOT/nemo_rl
    $PROJECT_ROOT/examples/run_grpo_single_controller.py
    policy.model_name=Qwen/Qwen3-0.6B
    grpo.num_prompts_per_step=2
    grpo.num_generations_per_prompt=4
    policy.train_global_batch_size=8
    policy.train_micro_batch_size=1
    cluster.gpus_per_node=2
    grpo.max_num_steps=4
    logger.tensorboard_enabled=false
    logger.wandb_enabled=false
    logger.monitor_gpus=false
    checkpointing.enabled=true
    checkpointing.checkpoint_dir=$CKPT_DIR
    checkpointing.save_period=2
    checkpointing.metric_name=null
    checkpointing.save_data_plane=true
    data_plane.enabled=true
    data_plane.impl=transfer_queue
    data_plane.backend=simple
    async_rl.sampler.name=windowed
    +async_rl.sampler.max_staleness_versions=1
    async_rl.min_groups_for_streaming_train=2
    async_rl.max_inflight_prompts=4
    async_rl.max_buffered_rollouts=4
)

cd $PROJECT_ROOT

# --- Run 1: the timeout save fires at the first step boundary and stops
# training early, leaving a complete step_1 checkpoint. ---
echo "=== Run 1: timeout save + early stop ==="
"${TRAIN_CMD[@]}" \
    checkpointing.checkpoint_must_save_by=00:00:00:01 \
    logger.log_dir=$LOG_DIR/run1 \
    $@ \
    2>&1 | tee $EXP_DIR/run1.log

if ! grep -q "Timeout has been reached, stopping training early" $EXP_DIR/run1.log; then
    echo "FAIL: timeout early-stop log line not found in run1 output"
    exit 1
fi
echo "✅ run1 stopped early on the checkpoint_must_save_by timeout"

STEP1=$CKPT_DIR/step_1
for artifact in \
    "$STEP1/training_info.json" \
    "$STEP1/config.yaml" \
    "$STEP1/policy/weights" \
    "$STEP1/train_dataloader.pt" \
    "$STEP1/replay_buffer_metadata.pt" \
    "$STEP1/data_plane"; do
    if [[ ! -e "$artifact" ]]; then
        echo "FAIL: expected checkpoint artifact missing: $artifact"
        exit 1
    fi
done
if [[ -e "$STEP1/replay_buffer.pt" ]]; then
    echo "FAIL: legacy tensor-bearing replay_buffer.pt should not be written"
    exit 1
fi
if compgen -G "$CKPT_DIR/tmp_step_*" > /dev/null; then
    echo "FAIL: tmp_step_* leftovers — async finalization was not flushed"
    exit 1
fi
echo "✅ step_1 checkpoint complete (weights, dataloader, TQ replay), no tmp leftovers"

if ! grep -q '"current_step": 1' "$STEP1/training_info.json"; then
    echo "FAIL: training_info.json does not record current_step=1"
    cat "$STEP1/training_info.json"
    exit 1
fi
if ! grep -q '"sampler_name": "windowed"' "$STEP1/training_info.json"; then
    echo "FAIL: training_info.json does not record the saving sampler_name"
    cat "$STEP1/training_info.json"
    exit 1
fi
echo "✅ training_info.json records current_step=1 and sampler_name=windowed"

# --- Run 2: resume from step_1 and train to max_num_steps. ---
echo "=== Run 2: resuming to step 4 ==="
"${TRAIN_CMD[@]}" \
    logger.log_dir=$LOG_DIR/run2 \
    $@ \
    2>&1 | tee $EXP_DIR/run2.log

if ! grep -q "Restoring dataloader state from checkpoint" $EXP_DIR/run2.log; then
    echo "FAIL: dataloader restore log line not found in run2 output"
    exit 1
fi
if ! grep -q "Restoring replay buffer metadata" $EXP_DIR/run2.log; then
    echo "FAIL: replay buffer restore log line not found in run2 output"
    exit 1
fi
if ! grep -qF "replay group(s) from checkpoint" $EXP_DIR/run2.log; then
    echo "FAIL: replay buffer restored-count summary line not found in run2 output"
    exit 1
fi
echo "✅ dataloader and replay buffer restored from checkpoint"

STEP4=$CKPT_DIR/step_4
if [[ ! -e "$STEP4/training_info.json" ]]; then
    echo "FAIL: run2 did not produce step_4 (resume did not reach step 4)"
    exit 1
fi
if [[ ! -f "$STEP4/replay_buffer_metadata.pt" || ! -d "$STEP4/data_plane" ]]; then
    echo "FAIL: resumed run did not produce native TQ replay artifacts at step_4"
    exit 1
fi
if [[ -e "$STEP4/replay_buffer.pt" ]]; then
    echo "FAIL: resumed run wrote legacy tensor-bearing replay_buffer.pt"
    exit 1
fi
if ! grep -q '"current_step": 4' "$STEP4/training_info.json"; then
    echo "FAIL: step_4 training_info.json does not record current_step=4"
    cat "$STEP4/training_info.json"
    exit 1
fi
if compgen -G "$CKPT_DIR/tmp_step_*" > /dev/null; then
    echo "FAIL: tmp_step_* leftovers after run2"
    exit 1
fi
echo "✅ grpo_checkpoint_single_controller passed"
