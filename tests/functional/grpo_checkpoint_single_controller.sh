#!/bin/bash
# Verifies that the SingleController (SC) path saves and restores a full
# checkpoint across a training restart: model weights/optimizer + training
# state (phase 1), dataloader position (phase 2), and the replay buffer
# (phase 3, over_sampling=True). Two runs share one checkpoint_dir: run 1
# trains 2 steps and checkpoints; run 2 resumes and trains 2 more.
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

# over_sampling=True (via staleness_window) so replay_buffer.pt is exercised;
# strict_on_policy would force over_sampling=False and skip it.
TRAIN_CMD=(
    uv run coverage run -a --data-file=$PROJECT_ROOT/tests/.coverage --source=$PROJECT_ROOT/nemo_rl
    $PROJECT_ROOT/examples/run_grpo_single_controller.py
    policy.model_name=Qwen/Qwen3-0.6B
    grpo.num_prompts_per_step=2
    grpo.num_generations_per_prompt=4
    policy.train_global_batch_size=8
    policy.train_micro_batch_size=1
    cluster.gpus_per_node=2
    logger.tensorboard_enabled=false
    logger.wandb_enabled=false
    logger.monitor_gpus=false
    checkpointing.enabled=true
    checkpointing.checkpoint_dir=$CKPT_DIR
    checkpointing.save_period=2
    # The two runs use different max_num_steps (2 then 4), so the Megatron
    # backend derives a different train_iters each run (via
    # _maybe_inject_megatron_train_iters), which changes the optimizer
    # scheduler's wd_incr_steps/lr_decay_iters. Megatron-LM's
    # OptimizerParamScheduler.load_state_dict asserts the saved horizon
    # matches the freshly-built one, so resuming with a longer horizon fails.
    # override_opt_param_scheduler=true keeps the (new) class schedule on
    # restore instead of asserting — correct for a resumed RL run that
    # extends training, and unrelated to the SC checkpoint save/restore under
    # test (the SC checkpoint itself loads fine). use_checkpoint_...=false is
    # required alongside override (SchedulerConfig.finalize forbids both true).
    # These scheduler keys are not in the YAML schema (struct-locked), so
    # Hydra needs the '+' append prefix to add them.
    +policy.megatron_cfg.scheduler.override_opt_param_scheduler=true
    +policy.megatron_cfg.scheduler.use_checkpoint_opt_param_scheduler=false
    data_plane.enabled=true
    data_plane.impl=transfer_queue
    data_plane.backend=simple
    async_rl.batch_selection_strategy=staleness_window
    async_rl.max_weight_staleness_versions=1
    async_rl.min_prompt_groups_per_batch=2
    async_rl.max_inflight_prompts=4
    async_rl.max_buffered_rollouts=4
    async_rl.over_sampling=true
)

cd $PROJECT_ROOT

# --- Run 1: train 2 steps and checkpoint ---
echo "=== Run 1: steps 1-2 ==="
"${TRAIN_CMD[@]}" \
    grpo.max_num_steps=2 \
    logger.log_dir=$LOG_DIR/run1 \
    $@ \
    2>&1 | tee $EXP_DIR/run1.log

STEP2=$CKPT_DIR/step_2
for artifact in \
    "$STEP2/training_info.json" \
    "$STEP2/config.yaml" \
    "$STEP2/policy/weights" \
    "$STEP2/train_dataloader.pt" \
    "$STEP2/replay_buffer.pt"; do
    if [[ ! -e "$artifact" ]]; then
        echo "FAIL: expected checkpoint artifact missing: $artifact"
        exit 1
    fi
done
echo "✅ step_2 checkpoint artifacts present (weights, dataloader, replay buffer)"

if ! grep -q '"current_step": 2' "$STEP2/training_info.json"; then
    echo "FAIL: training_info.json does not record current_step=2"
    cat "$STEP2/training_info.json"
    exit 1
fi
echo "✅ training_info.json records current_step=2"

# --- Run 2: resume from checkpoint and train 2 more steps ---
echo "=== Run 2: resuming, steps 3-4 ==="
"${TRAIN_CMD[@]}" \
    grpo.max_num_steps=4 \
    logger.log_dir=$LOG_DIR/run2 \
    $@ \
    2>&1 | tee $EXP_DIR/run2.log

if ! grep -q "Restoring dataloader state from checkpoint" $EXP_DIR/run2.log; then
    echo "FAIL: dataloader restore log line not found in run2 output"
    exit 1
fi
echo "✅ dataloader state restored from checkpoint"

if ! grep -q "Restoring replay buffer from checkpoint" $EXP_DIR/run2.log; then
    echo "FAIL: replay buffer restore log line not found in run2 output"
    exit 1
fi
if ! grep -qF "replay group(s) from checkpoint" $EXP_DIR/run2.log; then
    echo "FAIL: replay buffer restored-count summary line not found in run2 output"
    exit 1
fi
echo "✅ replay buffer restored from checkpoint"

if [[ ! -e "$CKPT_DIR/step_4" ]]; then
    echo "FAIL: run2 did not produce step_4 (resume did not reach step 4)"
    exit 1
fi
echo "✅ grpo_sc_checkpoint passed"
