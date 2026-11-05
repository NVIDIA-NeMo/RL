#!/bin/bash
# Rollout-only simulator e2e with SGLang generation on the native math env:
# examples/run_single_controller_simulator.py on
# examples/configs/grpo_math_1B_sglang_single_controller.yaml with the trainer stubbed
# out (cpu train backend) and a no-op refit. With SGLang the pause lives inside the
# synchronizer, so each simulated refit still pauses the engine, flushes its KV cache
# and holds for --refit-time-s before resuming; the gate on timing/train/weight_sync
# checks that hold happened. Same data and overrides as grpo_dp_single_controller_sglang.sh.

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath $SCRIPT_DIR/../..)
# Mark the current repo as safe, since wandb fetches metadata about the repo
git config --global --add safe.directory $PROJECT_ROOT

set -eou pipefail

EXP_NAME=${EXP_NAME:-$(basename $0 .sh)}
EXP_DIR=$SCRIPT_DIR/$EXP_NAME
LOG_DIR=$EXP_DIR/logs
JSON_METRICS=$EXP_DIR/metrics.json
RUN_LOG=$EXP_DIR/run.log
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

rm -rf $EXP_DIR $LOG_DIR
mkdir -p $EXP_DIR $LOG_DIR

cd $PROJECT_ROOT
uv run --group test coverage run -a --data-file=$PROJECT_ROOT/tests/.coverage --source=$PROJECT_ROOT/nemo_rl \
    $PROJECT_ROOT/examples/run_single_controller_simulator.py \
    --config $PROJECT_ROOT/examples/configs/grpo_math_1B_sglang_single_controller.yaml \
    --num-steps 4 \
    --train-time-s 10 \
    --refit-time-s 3 \
    policy.model_name=Qwen/Qwen3-0.6B \
    grpo.num_prompts_per_step=2 \
    grpo.num_generations_per_prompt=4 \
    grpo.seq_logprob_error_threshold=1000 \
    policy.train_global_batch_size=8 \
    policy.train_micro_batch_size=1 \
    cluster.gpus_per_node=2 \
    logger.tensorboard_enabled=true \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=false \
    logger.monitor_gpus=false \
    checkpointing.enabled=false \
    data_plane.enabled=true \
    data_plane.impl=transfer_queue \
    data_plane.backend=simple \
    async_rl.sampler.name=in_order \
    async_rl.sampler.max_lookahead_versions=0 \
    async_rl.min_groups_for_streaming_train=2 \
    async_rl.max_inflight_prompts=2 \
    async_rl.max_buffered_rollouts=2 \
    $@ \
    2>&1 | tee $RUN_LOG

if [[ "${RUN_CONVERGENCE_CHECKS:-1}" == "1" ]]; then
    uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

    # Liveness: the math env scored at least one rollout. Fidelity: the cpu trainer
    # publishes generation_logprobs as prev_logprobs, so the ratio is exactly 1. Refit:
    # the SGLang no-op synchronizer pauses, flushes and holds --refit-time-s every step.
    uv run tests/check_metrics.py $JSON_METRICS \
        'max(data["train/reward"]) > 0' \
        'abs(max(data["train/max_seq_mult_prob_error"]) - 1.0) < 1e-6' \
        'min(data["timing/train/weight_sync"]) >= 3' \
        'min(data["train/simulated_train_time_s"]) >= 0' \
        'max(data["train/simulated_train_idle_skipped_s"]) <= 10'
fi
