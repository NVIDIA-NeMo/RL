#!/bin/bash
# Chaos variant of grpo_dp_single_controller.sh: SIGKILL a vLLM generation worker
# mid-run and assert the job fails fast and loudly instead of wedging.
#
# This is the scenario the whole SC resiliency effort exists for. Before the P0
# work, a dead generation endpoint left rollouts parked forever -- each holding a
# max_inflight_prompts permit -- until the rollout pump blocked and the train pump
# spun, with no exception raised anywhere. The pass condition here is therefore not
# "training succeeds"; it is "the job stops, quickly, with an attributable error".
#
# Registered in the SingleController L1 lane (full mode). It was originally kept out as
# "timing-sensitive", but that no longer justifies exclusion: the death deadline is now
# 600s against an observed 222s, the victim selection is asserted rather than assumed, and
# the recovery tests in the same lane kill processes the same way. Leaving it out meant
# the containment behaviour had no end-to-end coverage at all -- and a wedge is precisely
# the failure that no other test can detect, because it raises nothing.
#
# Usage: bash tests/functional/grpo_dp_single_controller_chaos.sh

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR"/../..)
git config --global --add safe.directory "$PROJECT_ROOT"

set -eou pipefail

EXP_NAME=$(basename "$0" .sh)
EXP_DIR=$SCRIPT_DIR/$EXP_NAME
LOG_DIR=$EXP_DIR/logs
RUN_LOG=$EXP_DIR/run.log
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

# How long to wait for the job to die after the kill. The point of the test is that
# this is bounded at all: pre-P0 the job would still be sitting here at any deadline.
# 600s, not 300s: an observed run took 222s to die (5 re-dispatch attempts with capped
# exponential backoff, which is the designed containment behaviour). 300s left only 26%
# headroom, and CI is slower than this workstation -- a deadline that tight turns into a
# flaky "wedge" report.
DEATH_DEADLINE_S=${DEATH_DEADLINE_S:-600}
# Pattern matching the Ray worker process that serves generation.
#
# Matches BOTH the `ray::`-titled form and the venv-path form, because both occur: Ray
# retitles its workers via setproctitle, and a worker observed mid-run had the venv path
# in /proc/<pid>/cmdline while the one killed in a later run showed `ray::`.
#
# This was widened after a run where the old `ray::.*[Gg]eneration` pattern selected a
# process that was NOT the serving generation worker: the kill landed, the worker kept
# serving, training continued to step 3/50, and the test was on course to report a wedge
# that never happened -- a false failure of the very behaviour it exists to prove. The
# assertions below matter more than the pattern: they make a mis-targeted kill visible
# instead of letting it masquerade as a hang.
WORKER_PATTERN=${WORKER_PATTERN:-'ray::.*[Gg]enerationWorker|ray_venvs/.*[Gg]enerationWorker/bin/python'}

rm -rf "$EXP_DIR"
mkdir -p "$EXP_DIR" "$LOG_DIR"

cd "$PROJECT_ROOT"

# Enough steps that the run is still going when the kill lands.
uv run "$PROJECT_ROOT"/examples/run_grpo_single_controller.py \
    policy.model_name=Qwen/Qwen3-0.6B \
    grpo.num_prompts_per_step=2 \
    grpo.num_generations_per_prompt=4 \
    policy.train_global_batch_size=8 \
    policy.train_micro_batch_size=1 \
    cluster.gpus_per_node=2 \
    grpo.max_num_steps=50 \
    logger.tensorboard_enabled=true \
    logger.log_dir="$LOG_DIR" \
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
    ++async_rl.rollout_timeout_s=120 \
    ++async_rl.generation_timeout_s=60 \
    ++async_rl.rollout_failure.max_attempts_per_prompt=3 \
    ++async_rl.watchdog.interval_s=10 \
    ++async_rl.watchdog.stall_timeout_s=180 \
    ++async_rl.watchdog.stall_action=abort \
    > "$RUN_LOG" 2>&1 &
TRAIN_PID=$!

cleanup() {
    kill -9 $TRAIN_PID 2>/dev/null || true
    # Killing the driver is not enough. Ray actors outlive it, and vLLM's engine
    # runs in a VLLM::EngineCore child that survives its parent actor being killed
    # -- which is exactly what this test does on purpose. Left behind, it holds
    # tens of GB of device memory and the next run fails to place its placement
    # groups, which looks like an unrelated bug.
    ray stop --force >/dev/null 2>&1 || true
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
    pkill -9 -f "megatron_policy_worker" 2>/dev/null || true
}
trap cleanup EXIT

# Same reasoning in reverse: refuse to start on a dirty machine rather than
# misreport a leftover allocation as a failure of the code under test.
if command -v nvidia-smi >/dev/null 2>&1; then
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
    if [[ ${USED:-0} -gt 1024 ]]; then
        echo "[chaos] FAIL: ${USED}MiB already in use on a GPU; clean up before running"
        nvidia-smi --query-compute-apps=pid,used_memory --format=csv
        exit 1
    fi
fi

echo "[chaos] training pid=$TRAIN_PID, waiting for the first train step..."
for _ in $(seq 1 120); do
    grep -q "train step 1/" "$RUN_LOG" 2>/dev/null && break
    kill -0 $TRAIN_PID 2>/dev/null || { echo "[chaos] FAIL: job died before the kill"; tail -40 "$RUN_LOG"; exit 1; }
    sleep 5
done
grep -q "train step 1/" "$RUN_LOG" || { echo "[chaos] FAIL: never reached a train step"; tail -40 "$RUN_LOG"; exit 1; }

VICTIM=$(pgrep -f "$WORKER_PATTERN" | head -1)
if [[ -z "${VICTIM:-}" ]]; then
    echo "[chaos] FAIL: no process matched '$WORKER_PATTERN'"
    ps -eo pid,args --no-headers | grep -iE "generation|EngineCore" | grep -v grep | head -20
    exit 1
fi
# Print what is actually being killed. The whole test is meaningless if the victim is not
# a generation worker, and a mis-targeted kill otherwise looks exactly like a pass of the
# "job keeps running" kind -- i.e. it is reported as a wedge.
VICTIM_CMD=$(tr '\0' ' ' < "/proc/$VICTIM/cmdline" 2>/dev/null | cut -c1-120)
echo "[chaos] killing generation worker pid=$VICTIM"
echo "[chaos]   cmdline: $VICTIM_CMD"
case "$VICTIM_CMD" in
    *[Gg]enerationWorker*) ;;
    *) echo "[chaos] FAIL: victim does not look like a generation worker"; exit 1 ;;
esac
kill -9 "$VICTIM"
KILLED_AT=$(date +%s)
sleep 2
if kill -0 "$VICTIM" 2>/dev/null; then
    echo "[chaos] FAIL: victim $VICTIM survived SIGKILL; nothing was actually killed"
    exit 1
fi

echo "[chaos] waiting up to ${DEATH_DEADLINE_S}s for the job to stop..."
DIED=0
for _ in $(seq 1 $((DEATH_DEADLINE_S / 5))); do
    if ! kill -0 $TRAIN_PID 2>/dev/null; then DIED=1; break; fi
    sleep 5
done

ELAPSED=$(( $(date +%s) - KILLED_AT ))
if [[ $DIED -ne 1 ]]; then
    echo "[chaos] FAIL: still running ${ELAPSED}s after the kill -- this is the wedge."
    # A wedge is only actionable if you can see where it is wedged. Dump the
    # controller's stacks before tearing anything down; "it hung" on its own has
    # already cost one debugging cycle here.
    SC_PID=$(pgrep -f "ray::SingleControllerActor" | head -1 || true)
    if [[ -n "${SC_PID:-}" ]] && command -v py-spy >/dev/null 2>&1; then
        echo "[chaos] --- py-spy dump of SingleControllerActor pid=$SC_PID ---"
        py-spy dump --pid "$SC_PID" --locals 2>&1 | head -80 || true
    fi
    for name in MegatronPolicyWorker VllmAsyncGenerationWorker; do
        pid=$(pgrep -f "ray::${name}" | head -1 || true)
        if [[ -n "${pid:-}" ]] && command -v py-spy >/dev/null 2>&1; then
            echo "[chaos] --- py-spy dump of ${name} pid=${pid} ---"
            py-spy dump --pid "$pid" 2>&1 | head -40 || true
        fi
    done
    tail -40 "$RUN_LOG"
    exit 1
fi

wait $TRAIN_PID && EXIT_CODE=0 || EXIT_CODE=$?
echo "[chaos] job stopped after ${ELAPSED}s with exit code ${EXIT_CODE}"

if [[ $EXIT_CODE -eq 0 ]]; then
    echo "[chaos] FAIL: job exited 0 -- a killed generation worker must not look like success"
    exit 1
fi

# The failure must name the rollout path, not surface as a bare Ray traceback.
if grep -qE "RolloutRedispatchExhausted|GenerationUnavailable|RolloutStall|RolloutTimeout" "$RUN_LOG"; then
    echo "[chaos] PASS: bounded, attributable failure ${ELAPSED}s after the kill"
    grep -oE "RolloutRedispatchExhausted|GenerationUnavailable|RolloutStall|RolloutTimeout" "$RUN_LOG" | sort | uniq -c
    exit 0
fi

echo "[chaos] FAIL: job stopped but no typed rollout failure was reported"
tail -40 "$RUN_LOG"
exit 1
