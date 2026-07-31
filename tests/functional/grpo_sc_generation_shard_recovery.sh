#!/bin/bash
# SIGKILL one of two vLLM generation shards mid-run and assert the job RECOVERS:
# training continues to completion on the surviving shard, with the refit communicator
# rebuilt without the dead ranks.
#
# The inverse of grpo_dp_single_controller_chaos.sh. That one asserts a bounded, loud
# failure -- the P0 containment behaviour. This asserts the P3 behaviour: not stopping
# cleanly, but carrying on.
#
# WHY THIS NEEDS >= 3 GPUs, and why it is a CI test rather than a workstation one.
# Recovery is only observable when losing a shard still leaves a fleet, so generation
# needs dp_size >= 2 (2 GPUs at tp=1) plus at least one trainer. On a 2-GPU box the
# only possible split is 1 trainer + 1 generation shard, and killing that shard leaves
# nothing to recover onto -- the run can only fail, which tests the P0 path again
# rather than this one. The script self-skips below that threshold instead of
# pretending to pass.
#
# Usage:
#   bash tests/functional/grpo_sc_generation_shard_recovery.sh
#   NUM_GPUS=8 bash tests/functional/grpo_sc_generation_shard_recovery.sh
#   REFIT_TRANSPORT=nccl_reshard bash tests/functional/grpo_sc_generation_shard_recovery.sh

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR"/../..)
git config --global --add safe.directory "$PROJECT_ROOT"

set -eou pipefail

EXP_NAME=$(basename "$0" .sh)
EXP_DIR=$SCRIPT_DIR/$EXP_NAME
LOG_DIR=$EXP_DIR/logs
RUN_LOG=$EXP_DIR/run.log
JSON_METRICS=$EXP_DIR/metrics.json
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

rm -rf "$EXP_DIR"
mkdir -p "$EXP_DIR" "$LOG_DIR"
cd "$PROJECT_ROOT"

NUM_GPUS=${NUM_GPUS:-$(nvidia-smi --list-gpus | wc -l)}
GEN_GPUS=${GEN_GPUS:-2}          # two shards at tp=1, so one can die and one remains
TRAIN_GPUS=$((NUM_GPUS - GEN_GPUS))

if (( NUM_GPUS < 3 )); then
    echo "[recovery] SKIP: needs >= 3 GPUs (2 generation shards + >= 1 trainer), found $NUM_GPUS."
    echo "[recovery] With one generation shard there is nothing to recover onto, so this"
    echo "[recovery] scenario cannot be distinguished from the fail-fast path."
    exit 0
fi

# Enough steps that the kill lands mid-run and enough refits follow it that a rebuilt
# communicator has to actually carry weights, not just be constructed.
MAX_STEPS=${MAX_STEPS:-12}
# The kill must not land before the fleet is up and a refit has already succeeded, so
# that a failure here means recovery broke rather than startup did.
KILL_AFTER_STEP=${KILL_AFTER_STEP:-3}
# Generous: a rebuild plus the next refit, not a hang budget. The pass condition is
# completion, so this only bounds a wedge.
COMPLETION_DEADLINE_S=${COMPLETION_DEADLINE_S:-1800}
# Both NCCL transports rebuild, by different routes: the plain collective re-inits one
# group, nccl_reshard also rebuilds its per-PP-stage bulk groups and regenerates the
# refit plan. Worth running both, since only the reshard path has to keep a plan and a
# communicator agreeing about the fleet.
REFIT_TRANSPORT=${REFIT_TRANSPORT:-null}

echo "[recovery] $NUM_GPUS GPUs -> $TRAIN_GPUS train, $GEN_GPUS generation (dp_size=$GEN_GPUS), refit_transport=$REFIT_TRANSPORT"

uv run python "$PROJECT_ROOT"/examples/run_grpo_single_controller.py \
    --config "$PROJECT_ROOT"/examples/configs/grpo_math_1B_megatron_single_controller.yaml \
    policy.generation.colocated.enabled=false \
    policy.generation.colocated.resources.num_nodes=1 \
    policy.generation.colocated.resources.gpus_per_node="$GEN_GPUS" \
    policy.generation.vllm_cfg.tensor_parallel_size=1 \
    policy.generation.vllm_cfg.async_engine=true \
    policy.generation.refit_transport="$REFIT_TRANSPORT" \
    cluster.gpus_per_node="$NUM_GPUS" \
    grpo.max_num_steps="$MAX_STEPS" \
    grpo.val_period=-1 \
    grpo.val_at_start=false \
    checkpointing.enabled=false \
    logger.log_dir="$LOG_DIR" \
    logger.wandb_enabled=false \
    logger.tensorboard_enabled=true \
    logger.monitor_gpus=false \
    ++async_rl.fleet_health.enabled=true \
    ++async_rl.fleet_health.probe_interval_s=5.0 \
    ++async_rl.watchdog.interval_s=30.0 \
    ++async_rl.watchdog.stall_timeout_s=300.0 \
    "$@" \
    > "$RUN_LOG" 2>&1 &
TRAIN_PID=$!

cleanup() {
    kill -9 $TRAIN_PID 2>/dev/null || true
    # vLLM runs the engine in a VLLM::EngineCore child that outlives its parent actor;
    # leaving it behind holds tens of GB and makes the next run fail for the wrong reason.
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
    pkill -9 -f "megatron_policy_worker" 2>/dev/null || true
    ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "[recovery] pid=$TRAIN_PID, waiting for train step $KILL_AFTER_STEP..."
for _ in $(seq 1 240); do
    grep -q "train step ${KILL_AFTER_STEP}/" "$RUN_LOG" 2>/dev/null && break
    kill -0 $TRAIN_PID 2>/dev/null || {
        echo "[recovery] FAIL: job died before the kill"; tail -60 "$RUN_LOG"; exit 1; }
    sleep 5
done
grep -q "train step ${KILL_AFTER_STEP}/" "$RUN_LOG" || {
    echo "[recovery] FAIL: never reached step $KILL_AFTER_STEP"; tail -60 "$RUN_LOG"; exit 1; }

# Kill exactly one generation shard. Sorting by pid keeps the choice deterministic.
mapfile -t GEN_PIDS < <(pgrep -f "VllmAsyncGenerationWorker" | sort -n)
if (( ${#GEN_PIDS[@]} < 2 )); then
    echo "[recovery] FAIL: expected >= 2 generation workers, found ${#GEN_PIDS[@]}"
    ps -eo pid,args --no-headers | grep "ray::" | grep -v grep | head -20
    exit 1
fi
VICTIM=${GEN_PIDS[0]}
echo "[recovery] killing generation shard pid=$VICTIM (of ${#GEN_PIDS[@]})"
kill -9 "$VICTIM"
KILLED_AT=$(date +%s)

echo "[recovery] waiting up to ${COMPLETION_DEADLINE_S}s for the run to finish..."
FINISHED=0
for _ in $(seq 1 $((COMPLETION_DEADLINE_S / 10))); do
    if ! kill -0 $TRAIN_PID 2>/dev/null; then FINISHED=1; break; fi
    sleep 10
done
ELAPSED=$(( $(date +%s) - KILLED_AT ))

if (( FINISHED == 0 )); then
    echo "[recovery] FAIL: still running ${ELAPSED}s after the kill -- this is a wedge."
    echo "[recovery] watchdog lines:"; grep -E "watchdog|stall|inflight" "$RUN_LOG" | tail -20
    exit 1
fi

wait $TRAIN_PID; EXIT_CODE=$?
echo "[recovery] job exited $EXIT_CODE, ${ELAPSED}s after the kill"

if (( EXIT_CODE != 0 )); then
    echo "[recovery] FAIL: a surviving shard should have carried the run to completion"
    grep -E "Error|Traceback|NoSurvivingShards|RayActorError" "$RUN_LOG" | tail -20
    exit 1
fi

# Completion alone is not enough: a run that never noticed the death would also exit 0.
# These pin that the death was seen AND that the communicator was actually rebuilt.
REBUILD_RE="rebuilding (nccl_reshard )?communicators? without shards"
if ! grep -Eq "$REBUILD_RE" "$RUN_LOG"; then
    echo "[recovery] FAIL: job completed but never rebuilt the refit communicator."
    echo "[recovery] Either the death went unnoticed, or a refit was never needed after it."
    grep -E "refit|fleet|shard" "$RUN_LOG" | tail -20
    exit 1
fi
echo "[recovery] rebuild observed:"; grep -E "$REBUILD_RE" "$RUN_LOG" | head -3

uv run tests/json_dump_tb_logs.py "$LOG_DIR" --output_path "$JSON_METRICS"
uv run tests/check_metrics.py "$JSON_METRICS" \
    "len(data[\"train/reward\"]) == $MAX_STEPS" \
    'max(data["train/reward"]) > 0'

echo "[recovery] PASS: survived a shard loss and completed all $MAX_STEPS steps (refit_transport=$REFIT_TRANSPORT)"
