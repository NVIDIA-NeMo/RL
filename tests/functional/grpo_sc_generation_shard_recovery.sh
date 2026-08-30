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

# Per variant, not per script. The lane runs several of these back to back from this one file,
# differing only by environment, and they all resolved to the same EXP_DIR -- which is
# `rm -rf`'d on entry. So each variant destroyed its predecessor's run.log and
# metrics.json, and by the time anyone looked only the last one's artifacts existed. That
# is the opposite of what you want from a lane whose failures are debugged after the fact.
#
# Named for the variant rather than the run, so a re-run still overwrites its own
# artifacts and the directory count stays fixed instead of growing per invocation.
#
# Read from the environment directly because the defaults for these are applied further
# down, after this path has to be known. `if` rather than `[[ ... ]] && ...`: under the
# `set -eou pipefail` above, a bare AND-list whose condition is false returns non-zero and
# takes the script with it.
EXP_VARIANT="${REFIT_TRANSPORT:-null}"
if [[ "${KILL_DURING_REFIT:-false}" == "true" ]]; then EXP_VARIANT+="-refit"; fi
if [[ "${FREEZE_VICTIM:-false}" == "true" ]]; then EXP_VARIANT+="-frozen"; fi
if [[ "${RESTART_DEAD_SHARDS:-false}" == "true" ]]; then EXP_VARIANT+="-restart"; fi

EXP_DIR=$SCRIPT_DIR/$EXP_NAME/$EXP_VARIANT
LOG_DIR=$EXP_DIR/logs
RUN_LOG=$EXP_DIR/run.log
JSON_METRICS=$EXP_DIR/metrics.json
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

rm -rf "$EXP_DIR"
mkdir -p "$EXP_DIR" "$LOG_DIR"
cd "$PROJECT_ROOT"

NUM_GPUS=${NUM_GPUS:-$(nvidia-smi --list-gpus | wc -l)}
GEN_GPUS=${GEN_GPUS:-2}          # two shards at tp=1, so one can die and one remains

# Training ranks, rounded DOWN to a power of two.
#
# Megatron asserts global_batch_size % (micro_batch_size * data_parallel_size) == 0
# (num_microbatches_calculator.py). This config inherits train_global_batch_size=512 and
# train_micro_batch_size=4 from grpo_math_1B.yaml, and tp=pp=cp=1, so dp is just the
# training GPU count. Taking every remaining GPU therefore breaks on common host sizes:
#
#   3 GPUs -> dp=1   512 %  4 = 0   ok
#   4 GPUs -> dp=2   512 %  8 = 0   ok
#   5 GPUs -> dp=3   512 % 12 = 8   assertion failure
#   8 GPUs -> dp=6   512 % 24 = 8   assertion failure
#  16 GPUs -> dp=14  512 % 56 = 8   assertion failure
#
# 8 is the usual CI runner size, so this test would have died in Megatron setup on exactly
# the machines where it is the only place the >= 3 GPU scenario can run at all -- and with
# an assertion that says nothing about shard recovery.
#
# 512 and 4 are both powers of two, so any power-of-two dp divides. Rounding down leaves
# some GPUs idle on hosts that are not GEN_GPUS + 2^k, which is the right trade for a test
# whose point is surviving a shard loss, not throughput.
TRAIN_GPUS=1
while (( TRAIN_GPUS * 2 <= NUM_GPUS - GEN_GPUS )); do TRAIN_GPUS=$((TRAIN_GPUS * 2)); done
USED_GPUS=$((GEN_GPUS + TRAIN_GPUS))

if (( NUM_GPUS < 3 )); then
    echo "[recovery] SKIP: needs >= 3 GPUs (2 generation shards + >= 1 trainer), found $NUM_GPUS."
    echo "[recovery] With one generation shard there is nothing to recover onto, so this"
    echo "[recovery] scenario cannot be distinguished from the fail-fast path."
    exit 0
fi

# How long to wait for device memory to come back: on entry, because the previous test in
# the lane may still be releasing it, and on exit, so this test cannot poison the next one.
# The lane runs the recovery variants back to back and then a chaos test, so every one of
# those handoffs is a chance to pass on a GPU that is still being reclaimed.
GPU_WAIT_S=${GPU_WAIT_S:-120}
GPU_SETTLE_S=${GPU_SETTLE_S:-60}

# GPUs whose used memory is low enough to place a worker on.
free_gpu_count() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
        | awk -v lim=1024 '$1 <= lim' | wc -l
}

# Waits up to $1 seconds for $USED_GPUS of them; non-zero if the budget runs out.
#
# Sampling once is wrong: SIGKILL does not free device memory synchronously, the driver
# takes seconds to tear down a context holding tens of GB, and killing a shard is the
# entire point of this test. See grpo_dp_single_controller_chaos.sh, where a one-shot
# sample 280ms after the previous test's cleanup aborted a GitHub CI run before it had
# executed a single line of product code.
wait_for_free_gpus() {
    local budget_s=$1 free
    for _ in $(seq 1 "$budget_s"); do
        free=$(free_gpu_count) || free=0
        if (( free >= USED_GPUS )); then
            return 0
        fi
        sleep 1
    done
    return 1
}

# Refuse to start on a dirty machine rather than misreport a leftover allocation as a
# failure of the code under test. Requires only the $USED_GPUS this test actually places
# on, not every GPU on the host -- on a large runner one unrelated process elsewhere says
# nothing about whether this test can run.
if command -v nvidia-smi >/dev/null 2>&1 && ! wait_for_free_gpus "$GPU_WAIT_S"; then
    echo "[recovery] FAIL: need $USED_GPUS free GPUs, still $(free_gpu_count) after ${GPU_WAIT_S}s; clean up before running"
    nvidia-smi --query-gpu=index,memory.used --format=csv
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv
    exit 1
fi

# Enough steps that the kill lands mid-run and enough refits follow it that a rebuilt
# communicator has to actually carry weights, not just be constructed.
#
# 24, not 12: on GB200 a step takes seconds -- the dp test runs its whole job in about
# 2.5 minutes, nearly all startup -- so 12 steps finished before the harness had even
# located a shard to kill (job 5886540). The run must still be going well after the kill,
# or 'it completed' proves nothing about recovery.
MAX_STEPS=${MAX_STEPS:-24}
# The kill must not land before the fleet is up and a refit has already succeeded, so
# that a failure here means recovery broke rather than startup did.
KILL_AFTER_STEP=${KILL_AFTER_STEP:-3}
# Generous: a rebuild plus the next refit, not a hang budget. The pass condition is
# completion, so this only bounds a wedge.
COMPLETION_DEADLINE_S=${COMPLETION_DEADLINE_S:-1800}
# Hard bound on ONE actor-discovery attempt. Generous enough for a healthy connect,
# short enough that ten failures cannot outlast the run.
ACTOR_QUERY_TIMEOUT_S=${ACTOR_QUERY_TIMEOUT_S:-20}
# Both NCCL transports rebuild, by different routes: the plain collective re-inits one
# group, nccl_reshard also rebuilds its per-PP-stage bulk groups and regenerates the
# refit plan. Worth running both, since only the reshard path has to keep a plan and a
# communicator agreeing about the fleet.
REFIT_TRANSPORT=${REFIT_TRANSPORT:-null}
# Deadline after which a worker aborts its own refit communicator so the controller can
# rebuild over the survivors. Enabled here in EVERY variant, not just KILL_DURING_REFIT:
# a kill at a step boundary can still land in the refit by chance -- that is exactly how
# job 5898311 wedged -- so leaving it off would let the flaky case stay flaky.
#
# 60s against a healthy refit of ~1.9s for this model on GB200. Deliberately far above,
# because firing early aborts a run that was merely slow, while firing late only means a
# wedge lasts a little longer before it is broken.
# The probe knobs are named here rather than inlined into the config overrides because
# the refit deadline has to be chosen against them -- see the FREEZE_VICTIM block below.
PROBE_INTERVAL_S=${PROBE_INTERVAL_S:-5.0}
UNHEALTHY_THRESHOLD=${UNHEALTHY_THRESHOLD:-3}
# How long a shard that stops answering survives before the probe condemns it to DEAD.
PROBE_CONDEMN_S=$(awk "BEGIN{printf \"%.1f\", $PROBE_INTERVAL_S * $UNHEALTHY_THRESHOLD}")

# Recorded before the default is applied: the frozen variant needs a much shorter
# deadline, but must still honour an explicit one from the caller.
REFIT_TIMEOUT_S_SET_BY_CALLER=${REFIT_TIMEOUT_S:+yes}
REFIT_TIMEOUT_S=${REFIT_TIMEOUT_S:-60}

# KILL_DURING_REFIT: kill while the refit collective is running, rather than at a step
# boundary.
#
# Defined before the run because it decides an env var the workers read. Timing alone
# cannot reach this window: a refit here takes ~0.10s, and job 5925668 aimed at it and
# landed in the RPC epilogue -- the run died of an ActorDiedError from a broadcast that
# had already completed, which is a different bug and left the abort path unexercised.
# So the harness holds one refit open instead: it creates HOLD_FILE, every generation
# worker parks at the top of its receive, the victim is killed there, and removing the
# file lets the survivors walk into a collective the victim will never join.
KILL_DURING_REFIT=${KILL_DURING_REFIT:-false}

# FREEZE_VICTIM: SIGSTOP the victim instead of SIGKILL, so it stays alive and simply
# stops participating.
#
# This is the only way to reach the refit watchdog. Job 6405953 ran both
# KILL_DURING_REFIT variants green and never once produced a RefitAborted: SIGKILL makes
# Ray notice within milliseconds, ActorDiedError reaches _sync_weights first, and the
# recovery runs off that instead. The deadline was armed at 60s the whole time and never
# fired. So those runs exercise the actor-death path -- worth having -- but say nothing
# about the abort path.
#
# A stopped process is exactly the case the abort exists for, and the one the failure
# message names: "a rank that is alive but not participating". Ray sees a healthy actor,
# NCCL waits forever, and only refit_timeout_s can break it. Requires
# KILL_DURING_REFIT=true, since freezing a shard at a step boundary just makes it look
# slow until the probe condemns it -- a different path again.
FREEZE_VICTIM=${FREEZE_VICTIM:-false}
if [[ "$FREEZE_VICTIM" == "true" && "$KILL_DURING_REFIT" != "true" ]]; then
    echo "[recovery] FATAL: FREEZE_VICTIM=true requires KILL_DURING_REFIT=true; freezing"
    echo "[recovery] a shard outside a refit never reaches the abort path."
    exit 1
fi

if [[ "$FREEZE_VICTIM" == "true" ]]; then
    # The deadline must outlast the hold. The hold sits INSIDE the watchdog window by
    # design (see test_the_hold_is_inside_the_watchdog_not_before_it), so the clock starts
    # when the refit begins, not when the victim is frozen. The harness then spends a
    # couple of seconds confirming every worker is parked, freezing one, and proving it
    # reached state T. Job 6414909 set the deadline to 5s and the abort fired during that
    # window: RefitAborted was raised correctly, but before any shard was frozen.
    #
    # HOLD_BUDGET_S covers confirm + freeze + verify. 5, not a guess: job 6414909 showed
    # that sequence demonstrably takes at least that long, and the `sleep 2` after SIGSTOP
    # is 2s on its own.
    HOLD_BUDGET_S=${HOLD_BUDGET_S:-5}

    # What this variant asserts, and what it deliberately does NOT.
    #
    # A frozen rank cannot die, so ActorDiedError never comes and the fleet probe is the
    # only thing that can condemn it. This test therefore covers the abort half only: the
    # deadline breaks the stalled collective, RefitAborted propagates out of the broadcast,
    # and the run ends ATTRIBUTABLY in seconds instead of hanging in NCCL forever. That is
    # the whole gain over pre-deadline behaviour and it is what job 6415757 measured.
    #
    # It does not assert recovery, because a frozen rank cannot produce one. Measured on
    # 4xGB200 (job 6415757): the victim reaches SUSPECT and stops there -- the recovery's
    # own mark_weights_partial then moves it to STALE -- and neither SUSPECT nor STALE is
    # absent, because both describe a shard that is alive and rejoins the next refit
    # (GenerationFleetHealth.absent_shards). absent comes back empty and
    # _reconcile_refit_membership refuses to rebuild over a fleet that still contains a
    # silent rank, which is the safe answer rather than a bug.
    #
    # An earlier version of this file offered a condemn-first mode that waited for the
    # probe to reach DEAD and then asserted a full rebuild. DEAD is not reachable for a
    # frozen rank on the timescale of a refit, so that mode could not pass at any deadline;
    # abort-and-recover is covered instead by the killed variants (recovery-refit), where
    # ActorDiedError makes the victim genuinely absent.
    if [[ -z "$REFIT_TIMEOUT_S_SET_BY_CALLER" ]]; then
        REFIT_TIMEOUT_S=$(awk "BEGIN{printf \"%.1f\", $HOLD_BUDGET_S + $PROBE_CONDEMN_S / 2}")
    fi
    if ! awk "BEGIN{exit !($REFIT_TIMEOUT_S > $HOLD_BUDGET_S)}"; then
        echo "[recovery] FATAL: refit_timeout_s=${REFIT_TIMEOUT_S}s does not outlast the"
        echo "[recovery] ~${HOLD_BUDGET_S}s the harness needs to park every worker, freeze one and"
        echo "[recovery] verify it stopped. The abort would fire before anything is frozen --"
        echo "[recovery] correct behaviour, but it proves nothing about a frozen rank (job 6414909)."
        exit 1
    fi
    echo "[recovery] frozen variant: deadline ${REFIT_TIMEOUT_S}s, above the ~${HOLD_BUDGET_S}s hold+freeze; expect the abort to end the run attributably"
fi

HOLD_FILE="$EXP_DIR/hold_refit"
rm -f "$HOLD_FILE"

echo "[recovery] $NUM_GPUS GPUs on host -> using $USED_GPUS: $TRAIN_GPUS train, $GEN_GPUS generation (dp_size=$GEN_GPUS), refit_transport=$REFIT_TRANSPORT"

# PYTHONUNBUFFERED: the harness detects progress by grepping RUN_LOG, and that only
# works if the driver actually writes to it. The actor prints with flush=True, but
# that just reaches the DRIVER -- Ray forwards actor output there, and the driver's
# own stdout is a redirected file, so Python block-buffers it. Job 5892910 wrote
# "train step 3/24" at 10:40:38 and the harness did not see it until 10:48:21, by
# which time the run had finished and there was nothing left to kill.
#
# NRL_REFIT_HOLD_FILE is exported in every variant, not just KILL_DURING_REFIT. The file
# is only ever created by the KILL_DURING_REFIT branch below, and the hook is a single
# os.path.exists when it is absent, so the other variants are unaffected -- and none of
# them has to remember to set an env var to stay correct.
PYTHONUNBUFFERED=1 NRL_REFIT_HOLD_FILE="$HOLD_FILE" \
uv run python "$PROJECT_ROOT"/examples/run_grpo_single_controller.py \
    --config "$PROJECT_ROOT"/examples/configs/grpo_math_1B_megatron_single_controller.yaml \
    policy.generation.colocated.enabled=false \
    policy.generation.colocated.resources.num_nodes=1 \
    policy.generation.colocated.resources.gpus_per_node="$GEN_GPUS" \
    policy.generation.vllm_cfg.tensor_parallel_size=1 \
    policy.generation.vllm_cfg.async_engine=true \
    policy.generation.refit_transport="$REFIT_TRANSPORT" \
    cluster.gpus_per_node="$USED_GPUS" \
    grpo.max_num_steps="$MAX_STEPS" \
    grpo.val_period=-1 \
    grpo.val_at_start=false \
    checkpointing.enabled=false \
    logger.log_dir="$LOG_DIR" \
    logger.wandb_enabled=false \
    logger.tensorboard_enabled=true \
    logger.monitor_gpus=false \
    ++async_rl.generation_fleet_health.enabled=true \
    ++async_rl.generation_fleet_health.probe_interval_s=$PROBE_INTERVAL_S \
    ++async_rl.generation_fleet_health.unhealthy_threshold=$UNHEALTHY_THRESHOLD \
    ++async_rl.generation_fleet_health.refit_timeout_s="$REFIT_TIMEOUT_S" \
    ++async_rl.stall_watchdog.interval_s=30.0 \
    ++async_rl.stall_watchdog.stall_timeout_s=300.0 \
    "$@" \
    > "$RUN_LOG" 2>&1 &
TRAIN_PID=$!

cleanup() {
    # Before anything else: a hold file surviving this run would park the next run's
    # refits until NRL_REFIT_HOLD_MAX_S, which reads as a hang with no visible cause.
    rm -f "$HOLD_FILE" 2>/dev/null || true
    # SIGCONT first: a SIGSTOPped process cannot be reaped and ignores SIGKILL until it
    # is scheduled again, so a frozen victim would survive this cleanup holding its share
    # of device memory and fail the next test for the wrong reason.
    [[ -n "${VICTIM:-}" ]] && kill -CONT "$VICTIM" 2>/dev/null || true
    kill -9 $TRAIN_PID 2>/dev/null || true
    # vLLM runs the engine in a VLLM::EngineCore child that outlives its parent actor;
    # leaving it behind holds tens of GB and makes the next run fail for the wrong reason.
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
    pkill -9 -f "megatron_policy_worker" 2>/dev/null || true
    ray stop --force >/dev/null 2>&1 || true
    # Signalling is not reclaiming: wait for the memory to actually come back before
    # handing the machine to the next test, which starts with no gap at all.
    if command -v nvidia-smi >/dev/null 2>&1 && ! wait_for_free_gpus "$GPU_SETTLE_S"; then
        echo "[recovery] WARN: ${GPU_SETTLE_S}s after cleanup, fewer than $USED_GPUS GPUs are free:"
        nvidia-smi --query-compute-apps=pid,used_memory --format=csv
    fi
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

# Kill exactly one generation shard.
#
# Ask Ray which processes its generation actors are, rather than inferring it from process
# titles. `pgrep -f VllmAsyncGenerationWorker` matched the venv child and the launcher shell
# as well as the actor -- three hits per shard -- and anchoring on Ray's `ray::` title fixed
# that on a workstation but found ZERO actors on a GB200 cluster (job 5861743: "expected
# exactly 2 generation actors, found 0" at train step 3, with generation working). Titles
# are a runtime implementation detail; the GCS actor table is the runtime's own record.
#
# This matters more here than in the chaos test: this one asserts the run COMPLETES, so
# killing a non-actor leaves both shards serving, the run finishes exactly as it would have
# anyway, and the test reports a pass having never exercised recovery.
# Retry: the actors are certainly up by train step 3, but a single query races Ray's
# GCS write and one empty result would abort a run that is otherwise fine.
# Each attempt is a full ray.init/shutdown of a few seconds, so this loop is also a
# multi-second delay -- and on fast hardware the remaining steps can finish inside it.
# Check the job on every attempt so "the run ended" is reported as itself rather than
# surfacing later as the far more confusing "expected 2 generation actors, found 0".
GEN_PIDS=()
ATTEMPT=0
for _ in $(seq 1 10); do
    ATTEMPT=$((ATTEMPT + 1))
    if ! kill -0 $TRAIN_PID 2>/dev/null; then
        echo "[recovery] FAIL: the run ended before a shard could be killed."
        echo "[recovery] It reached step $KILL_AFTER_STEP, then finished or died while the"
        echo "[recovery] harness was still locating the generation actors. If it completed"
        echo "[recovery] all $MAX_STEPS steps, raise MAX_STEPS or lower KILL_AFTER_STEP --"
        echo "[recovery] this hardware runs a step in seconds."
        echo "[recovery] --- helper stderr from the last attempt (job was still alive) ---"
        echo "${ACTORS_ERR:-<no attempt completed>}" | tail -25 | sed 's/^/[recovery]   /'
        echo "[recovery] --- last 60 lines of the training log ---"
        tail -60 "$RUN_LOG"
        exit 1
    fi
    # Keep stderr. Discarding it is why three rounds of this were undiagnosable: the
    # helper explains itself there, and the loop was throwing that away.
    # Print the helper's stderr INLINE, into this log.
    #
    # Two previous attempts routed it to $EXP_DIR/actors.err and neither produced a file
    # that survived to be read. A separate artifact is one more thing that has to be
    # written, kept, and collected; the harness log is already captured, so put it there.
    # stdout (the pids) goes to a temp file, stderr into a variable.
    : > "$EXP_DIR/pids.tmp"
    ACTORS_ERR=$(timeout "$ACTOR_QUERY_TIMEOUT_S" uv run --no-sync python \
        "$SCRIPT_DIR/_find_generation_actors.py" 2>&1 >"$EXP_DIR/pids.tmp" || true)
    mapfile -t GEN_PIDS < <(sort -n < "$EXP_DIR/pids.tmp")
    if (( ${#GEN_PIDS[@]} != GEN_GPUS )) && [[ -n "$ACTORS_ERR" && $ATTEMPT -le 2 ]]; then
        # Only the first couple of attempts, or a 10-attempt loop floods the log.
        echo "[recovery] discovery attempt $ATTEMPT found ${#GEN_PIDS[@]} pid(s); helper said:"
        echo "$ACTORS_ERR" | tail -25 | sed 's/^/[recovery]   /'
    fi
    (( ${#GEN_PIDS[@]} == GEN_GPUS )) && break
    sleep 3
done

if (( ${#GEN_PIDS[@]} != GEN_GPUS )); then
    echo "[recovery] FAIL: expected exactly $GEN_GPUS generation actors, found ${#GEN_PIDS[@]}"
    echo "[recovery] this is a harness problem, not a recovery failure -- killing the wrong"
    echo "[recovery] process would let the run complete and report a false pass."
    echo "[recovery] --- helper stderr from the LAST in-loop attempt (job was alive) ---"
    # This is the one that matters: the diagnostic call below runs after the job has gone,
    # so it can only ever say "cluster not found".
    echo "${ACTORS_ERR:-<no attempt completed>}" | tail -25 | sed 's/^/[recovery]   /'
    echo "[recovery] --- what Ray reports now (job already gone) ---"
    uv run --no-sync python "$SCRIPT_DIR/_find_generation_actors.py" || true
    echo "[recovery] --- every process with 'eneration' in its command line ---"
    # Unfiltered on purpose. The previous diagnostic grepped for "ray::" and so printed
    # nothing precisely when the ray:: assumption was the thing that was wrong.
    ps -eo pid=,args= 2>/dev/null | sed -E 's/^ *//' | grep -i "eneration" | grep -v grep | head -20
    echo "[recovery] --- last 60 lines of the training log ---"
    # Without this the log says only "found 0 actors", which reads as a harness bug even
    # when the real event is the training job ending. That cost a full debug round.
    tail -60 "$RUN_LOG"
    exit 1
fi
if [[ "$KILL_DURING_REFIT" == "true" ]]; then
    # Arm the hold, then wait for the workers to report that they are parked inside a
    # refit. Only then is "killed during the refit" a fact rather than a hope.
    echo "[recovery] arming the refit hold at $HOLD_FILE"
    : > "$HOLD_FILE"
    # Wait for EVERY generation worker to report, not just the first.
    #
    # The victim is a fixed pid (GEN_PIDS[0]) but the old check matched a message from
    # ANY worker, so the harness could freeze a shard that had not reached the hold yet
    # -- leaving it somewhere else entirely and making "frozen inside the refit" a hope
    # rather than a fact. Counting the reports removes the race.
    HELD=false
    for _ in $(seq 1 1200); do
        # NOT `$(grep -c ... || echo 0)`: with no match grep PRINTS 0 and EXITS 1, so the
        # fallback runs too and HOLDING becomes the two-line string "0\n0", which (( ))
        # then rejects. Harmless -- the failed test just retries the loop -- but it put
        # 120 syntax errors into a 297-line log (job 6428488). Assign, then default.
        HOLDING=$(grep -c "refit: holding the receive open" "$RUN_LOG" 2>/dev/null) || HOLDING=0
        (( HOLDING >= GEN_GPUS )) && { HELD=true; break; }
        kill -0 $TRAIN_PID 2>/dev/null || {
            echo "[recovery] FAIL: run ended before a refit could be held"
            tail -40 "$RUN_LOG"; exit 1; }
        sleep 0.1
    done
    if [[ "$HELD" != "true" ]]; then
        echo "[recovery] FAIL: only $HOLDING of $GEN_GPUS generation workers reported"
        echo "[recovery] holding a refit within 120s."
        echo "[recovery] The hook reads NRL_REFIT_HOLD_FILE; without it this test cannot"
        echo "[recovery] reach the mid-refit window at all and would silently test the"
        echo "[recovery] step-boundary case instead."
        tail -60 "$RUN_LOG"; exit 1
    fi
    echo "[recovery] all $GEN_GPUS generation workers are parked in the refit; the victim is one of them"
fi

VICTIM=${GEN_PIDS[0]}
VICTIM_CMD=$(tr '\0' ' ' < "/proc/$VICTIM/cmdline" 2>/dev/null | sed -E 's/ +$//')
if [[ "$FREEZE_VICTIM" == "true" ]]; then
    echo "[recovery] freezing generation shard pid=$VICTIM of ${#GEN_PIDS[@]}: $VICTIM_CMD"
    kill -STOP "$VICTIM"
    sleep 2
    # State T in field 3 of /proc/PID/stat is the only proof it actually stopped. `kill -0`
    # succeeds for a running process too, so it cannot tell the two apart -- and a victim
    # that kept running would make this test silently assert nothing.
    VICTIM_STATE=$(awk '{print $3}' "/proc/$VICTIM/stat" 2>/dev/null || echo "gone")
    if [[ "$VICTIM_STATE" != "T" ]]; then
        echo "[recovery] FAIL: victim $VICTIM is in state '$VICTIM_STATE', expected T (stopped)."
        echo "[recovery] Without a genuinely frozen rank the collective never stalls and the"
        echo "[recovery] refit deadline has nothing to fire on."
        exit 1
    fi
else
    echo "[recovery] killing generation shard pid=$VICTIM of ${#GEN_PIDS[@]}: $VICTIM_CMD"
    kill -9 "$VICTIM"
    sleep 2
    if kill -0 "$VICTIM" 2>/dev/null; then
        echo "[recovery] FAIL: victim $VICTIM survived SIGKILL; nothing was actually killed"
        exit 1
    fi
fi
# Release the survivors into a collective the victim will never join. Removed only after
# the kill is confirmed: dropping it earlier would let them enter the receive alongside a
# victim that is still alive, and the refit would simply succeed.
rm -f "$HOLD_FILE"
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
    if [[ "$FREEZE_VICTIM" == "true" && "$REFIT_TRANSPORT" == "nccl_reshard" ]]; then
        # This variant's PASS is a bounded attributable FAILURE, so a wedge is the one
        # outcome that looks superficially similar and means the opposite. Named here
        # because the generic message above reads as "did not recover", which is not the
        # point: recovery is out of scope on this transport (see sync_stream_within), and
        # what regressed is that the run no longer ENDS. Check the release_within lines --
        # an abort that never retires leaves the rebuild unable to bootstrap NCCL.
        echo "[recovery] This variant expects a bounded attributable failure, not recovery."
        echo "[recovery] A wedge means the abort never retired; look for 'did not release'."
        grep -E "did not release within|did not retire" "$RUN_LOG" | tail -5
    fi
    # Dump stacks BEFORE tearing anything down. The chaos test has done this for a while;
    # this one did not, and job 5893807 cost a whole cycle to a wedge whose location could
    # only be guessed at. "0 rollouts in flight" says the pump stopped, not where.
    if command -v py-spy >/dev/null 2>&1; then
        SC_PID=$(pgrep -f "ray::SingleControllerActor" | head -1 || true)
        if [[ -n "${SC_PID:-}" ]]; then
            echo "[recovery] --- py-spy dump of SingleControllerActor pid=$SC_PID ---"
            # --locals matters here: whether the pump is parked on _rollout_permitted, on
            # the _buffer_capacity semaphore, or in the sampler is exactly the question,
            # and the frame alone does not distinguish them.
            py-spy dump --pid "$SC_PID" --locals 2>&1 | head -100 || true
        fi
        for name in MegatronPolicyWorker VllmAsyncGenerationWorker; do
            for pid in $(pgrep -f "ray::${name}" | head -2); do
                echo "[recovery] --- py-spy dump of ${name} pid=${pid} ---"
                py-spy dump --pid "$pid" 2>&1 | head -40 || true
            done
        done
    else
        echo "[recovery] py-spy not available; cannot show where it is wedged"
    fi
    echo "[recovery] watchdog lines:"; grep -E "watchdog|stall|inflight" "$RUN_LOG" | tail -20
    echo "[recovery] fleet/refit activity:"
    grep -E "gen_fleet: shard|rebuilt refit communicator|_sync_weights: sync done" "$RUN_LOG" | tail -15
    exit 1
fi

# `wait` must not be a bare simple command: under the `set -eou pipefail` above, a
# non-zero training exit terminates the script here and every FAIL diagnostic below
# -- the job exit code, the error-pattern grep, the survivor assertion -- never runs,
# on exactly the failure this test exists to report. Same form as the chaos harness.
wait $TRAIN_PID && EXIT_CODE=0 || EXIT_CODE=$?
echo "[recovery] job exited $EXIT_CODE, ${ELAPSED}s after the kill"

# The frozen variants, and the two transports do NOT agree here -- the expectation is split
# because the outcomes genuinely differ, not because one of them is under-tested.
#
# A frozen rank never becomes absent: is_alive() is answered by the Ray actor and never
# touches the engine, so the probe can only ever see a dead process. The ledger knows
# anyway, because the frozen shard's own generations time out and drive it to SUSPECT
# before the refit aborts, and the recovery condemns that single suspect.
#
# On the packed-broadcast transport that is enough, and the run carries on.
#
# On nccl_reshard it is not, and cannot be. The bulk transfer aborts by way of
# sync_stream_within, which gives up on kernels already enqueued on the TRAINERS' streams
# -- and aborting a communicator does not retire them. Its docstring has said so since it
# was written: "In-flight kernels are orphaned and the caller's CUDA context should not be
# trusted afterwards, so the RefitAborted raised here is expected to end the run."
#
# Jobs 6521181 and 6523731 measured the consequence. Both trainers' py-spy dumps sat in
# init_nccl_communicator with the abandoned ncclCommAborts still in native code 25 minutes
# on, and the rebuild could not bootstrap a new communicator on a device holding a
# half-aborted one. Job 6523731 also proved it is not the peer's doing: the victim was
# ray.killed before the rebuild and the trainers wedged identically. Nothing done to the
# remote rank retires local GPU work.
#
# So on this transport the gain being pinned is the one the six bounds actually deliver --
# the run ends attributably in seconds instead of wedging in NCCL for 33 minutes (job
# 6258553) -- and asserting recovery here would assert a property the design explicitly
# places out of scope.
if [[ "$KILL_DURING_REFIT" == "true" && "$REFIT_TRANSPORT" == "nccl_reshard" ]]; then
    if (( EXIT_CODE == 0 )); then
        echo "[recovery] FAIL: the run completed. On nccl_reshard an aborted bulk transfer"
        echo "[recovery] orphans kernels on the trainers, so completing means the abort"
        echo "[recovery] never fired and this variant tested nothing."
        exit 1
    fi
    if ! grep -q "RefitAborted" "$RUN_LOG"; then
        echo "[recovery] FAIL: the run failed without a RefitAborted, so it died of"
        echo "[recovery] something other than the deadline this variant exists to exercise."
        grep -E "watchdog|deadline|Traceback" "$RUN_LOG" | tail -20
        exit 1
    fi
    # The whole point is bounded. Wedging for the full harness timeout is the failure this
    # replaced, and it would otherwise reach the check above looking like a pass.
    if (( ELAPSED > 900 )); then
        echo "[recovery] FAIL: the run ended attributably but took ${ELAPSED}s. The bounds"
        echo "[recovery] exist to make this fast; something is waiting that should not be."
        exit 1
    fi
    if [[ "$FREEZE_VICTIM" == "true" ]] && \
       [[ "$(awk '{print $3}' "/proc/$VICTIM/stat" 2>/dev/null || echo gone)" == "gone" ]]; then
        echo "[recovery] FAIL: the frozen victim disappeared; it was not the frozen-rank"
        echo "[recovery] scenario that failed, so the result does not mean what it says."
        exit 1
    fi
    # The guard has to be REACHED, not merely consistent with the exit code. Without this
    # a run that died of anything else in under 900s would read as a pass.
    if ! grep -q "Recovery is not possible from here" "$RUN_LOG"; then
        echo "[recovery] FAIL: the run failed attributably but never reported the"
        echo "[recovery] context-lost guard, so it did not fail for the reason this pins."
        grep -E "refit-context-lost|_sync_weights:" "$RUN_LOG" | tail -10
        exit 1
    fi
    echo "[recovery] abort observed:"; grep -m3 "RefitAborted" "$RUN_LOG"
    echo "[recovery] guard observed:"; grep -m1 "Recovery is not possible from here" "$RUN_LOG"
    echo "[recovery] PASS: a mid-refit fault on nccl_reshard was detected and ended the run"
    echo "[recovery] in ${ELAPSED}s rather than wedging (recovery on this path is a known"
    echo "[recovery] limitation -- see sync_stream_within and design doc 8.5.7)"
    exit 0
fi

# The packed-broadcast frozen variant, where the condemned suspect IS recoverable: the
# pass condition is completion, like the killed variants, plus evidence that it got there
# by attribution rather than by the shard dying.
if [[ "$FREEZE_VICTIM" == "true" ]]; then
    if (( EXIT_CODE != 0 )); then
        echo "[recovery] FAIL: the run exited $EXIT_CODE ${ELAPSED}s after the freeze."
        echo "[recovery] A single suspect should have been condemned and the run continued."
        grep -E "already suspect|identified as absent|RefitAborted|Traceback" "$RUN_LOG" | tail -20
        exit 1
    fi
    if ! grep -q "RefitAborted" "$RUN_LOG"; then
        echo "[recovery] FAIL: the run completed, but no RefitAborted appears in the log, so"
        echo "[recovery] the deadline never fired and this variant tested nothing."
        grep -E "watchdog|deadline" "$RUN_LOG" | tail -20
        exit 1
    fi
    # EITHER attribution route, because both are correct and which one runs is a race.
    #
    # The condemn is only needed when the shard never becomes absent on its own. But the
    # frozen shard's generations also time out, and if the probe drives it to DEAD before
    # the refit aborts, the ordinary absent-shard path handles it and the condemn is never
    # reached. Job 6524733 recovered that way -- exit 0, 420s, "shard 0 suspect -> dead
    # (TimeoutError)" -- and failed a check that demanded the other route's log line.
    #
    # What must be pinned is that the shard was attributed FROM THE LEDGER rather than by
    # dying, which both of these are and neither an actor death nor a silent stall is.
    ATTRIBUTION_RE="condemning it as the silent participant|-> dead \\(TimeoutError"
    if ! grep -Eq "$ATTRIBUTION_RE" "$RUN_LOG"; then
        echo "[recovery] FAIL: RefitAborted fired and the run survived, but the frozen shard"
        echo "[recovery] was never attributed -- neither condemned as the silent participant"
        echo "[recovery] nor driven to DEAD by its own probe. It recovered by some other"
        echo "[recovery] route, so this variant is no longer testing attribution."
        grep -E "already suspect|identified as absent|gen_fleet: shard" "$RUN_LOG" | tail -20
        exit 1
    fi
    # Never killed, so it must still be there -- stopped. If it is gone, something else
    # reaped it and this was the actor-death path after all.
    if [[ "$(awk '{print $3}' "/proc/$VICTIM/stat" 2>/dev/null || echo gone)" == "gone" ]]; then
        echo "[recovery] FAIL: the frozen victim disappeared; it was not the frozen-rank"
        echo "[recovery] scenario that recovered, so the result does not mean what it says."
        exit 1
    fi
    echo "[recovery] abort observed:"; grep -m3 "RefitAborted" "$RUN_LOG"
    echo "[recovery] attribution observed:"; grep -Em1 "$ATTRIBUTION_RE" "$RUN_LOG"
    echo "[recovery] PASS: a frozen rank was attributed from the ledger and the run carried"
    echo "[recovery] on, ${ELAPSED}s after the freeze (refit_transport=$REFIT_TRANSPORT)"
    exit 0
fi

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

# The whole point of the frozen variant: prove the deadline fired, not merely that the
# run survived. Job 6405953 passed both KILL_DURING_REFIT variants with RefitAborted
# appearing exactly zero times -- ActorDiedError won the race every time and the abort
# path went untested while four green ticks implied otherwise. A frozen rank cannot
# produce ActorDiedError, so if RefitAborted is still absent here the watchdog did not
# fire and this variant is testing nothing.

uv run tests/json_dump_tb_logs.py "$LOG_DIR" --output_path "$JSON_METRICS"
uv run tests/check_metrics.py "$JSON_METRICS" \
    "len(data[\"train/reward\"]) == $MAX_STEPS" \
    'max(data["train/reward"]) > 0'

echo "[recovery] PASS: survived a shard loss and completed all $MAX_STEPS steps (refit_transport=$REFIT_TRANSPORT)"
