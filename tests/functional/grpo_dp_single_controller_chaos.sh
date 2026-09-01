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
# 600s against an observed 222s, and the victim selection is asserted rather than assumed.
# (The shard-recovery harness added in part 3/4 of this series kills processes the same
# way, from the same lane.) Leaving it out meant
# the containment behaviour had no end-to-end coverage at all -- and a wedge is precisely
# the failure that no other test can detect, because it raises nothing.
#
# Usage: bash tests/functional/grpo_dp_single_controller_chaos.sh

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR"/../..)
git config --global --add safe.directory "$PROJECT_ROOT"

set -eou pipefail

# Suffixed with the victim state because BOTH lane entries run this script, and it opens
# with `rm -rf "$EXP_DIR"` -- so a shared EXP_NAME meant the serving run deleted the idle
# run's run.log and tensorboard artifacts. When serving then failed in CI, the idle
# evidence you would compare it against was already gone.
# VICTIM_STATE is not assigned until below, hence repeating its default here.
EXP_NAME=$(basename "$0" .sh)-${VICTIM_STATE:-idle}
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
# Which generation worker to kill, and in which state.
#
# Ray retitles a worker with setproctitle for the exact duration of a call, so a single
# actor cycles through several titles. Observed over one run (378 samples):
#
#   /opt/ray_venvs/...VllmAsyncGenerationWorker/bin/python   a CHILD process, not the actor
#   bash -c exec /opt/ray_venvs/...GenerationWorker...       the launcher shell
#   ray::VllmAsyncGenerationWorker                           the actor, between calls
#   ray::vllm_policy-0-0:VllmAsyncGenerationWorker.__init__  the actor, constructing
#   ray::VllmAsyncGenerationWorker.generate_async            the actor, serving a rollout
#   ray::VllmAsyncGenerationWorker.init_collective_async     the actor, setting up refit
#   ray::VllmAsyncGenerationWorker.shutdown                  the actor, tearing down
#
# A loose `[Gg]enerationWorker` substring matches every one of those -- three distinct
# processes across five states -- and `head -1` then picked whichever had the lowest pid.
# So the test silently chose a different scenario from run to run. It never failed, because
# every scenario does end in a bounded attributable failure, which is all it asserted; the
# difference only showed up as wildly different wall-clock times across branches (7s vs
# 222s). A `*GenerationWorker*` check on the victim does not help either: the launcher
# shell and the venv child both satisfy it.
#
# So select the ACTOR structurally (`ray::` prefix, optional `name:` infix) and pin the
# state:
#   idle    -- title has no method suffix. Nothing is in flight, so the loss must be
#              *detected*: by a health probe, or by the stall detector. This exercises the
#              resiliency machinery, so it is the default.
#   serving -- title is `.generate_async`. An in-flight rollout RPC dies with the worker
#              and the failure surfaces immediately, without detection doing any work.
#
# Deliberately not "any method suffix": killing during __init__, init_collective_async or
# shutdown are three further distinct scenarios, and lumping them in would reintroduce
# exactly the ambiguity this replaces.
VICTIM_STATE=${VICTIM_STATE:-idle}
# How long to wait for the worker to be observed in that state. Generous because `serving`
# is a narrow window -- generate_async was only ~2% of samples, against ~34% for idle.
VICTIM_WAIT_S=${VICTIM_WAIT_S:-300}
# Separate budget for finding the actors at all, before any state is considered.
# Generation takes a while to come up, and conflating the two made 'no actors' and
# 'actors never reached the state' indistinguishable in the log.
ACTOR_WAIT_S=${ACTOR_WAIT_S:-300}
# Hard bound on ONE discovery attempt: an unbounded Ray actor-table query against a GCS
# that is itself unwell hangs the harness rather than the run it is watching.
ACTOR_QUERY_TIMEOUT_S=${ACTOR_QUERY_TIMEOUT_S:-20}
# GPUs this test pins itself to, independent of how many the host has. One shard of
# generation and one trainer: killing the shard leaves the fleet empty, which is the
# scenario -- a bounded failure with nothing to fall back to. Defined once because the
# pre-flight check below has to agree with it.
GPUS=2
# How long to wait for device memory to come back: on entry, because the previous test in
# the lane may still be releasing it, and on exit, so this test cannot poison the next one.
GPU_WAIT_S=${GPU_WAIT_S:-120}
GPU_SETTLE_S=${GPU_SETTLE_S:-60}

# GPUs whose used memory is low enough to place a worker on.
free_gpu_count() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
        | awk -v lim=1024 '$1 <= lim' | wc -l
}

# Waits up to $1 seconds for $GPUS of them; non-zero if the budget runs out.
#
# Sampling once is wrong, because SIGKILL does not free device memory synchronously: the
# driver tears the context down, and a worker holding tens of GB stays visible to
# nvidia-smi for seconds after it dies. The lane runs tests back to back -- run_test is
# just `time "$@"` -- so a single sample reads the previous test's corpse. That is exactly
# what broke GitHub CI: chaos-idle passed at 19:51:42, its cleanup killed the
# VLLM::EngineCore it had orphaned on purpose, and chaos-serving sampled 280ms later,
# found GPU 0 still holding 50470MiB, and refused to start without running a single line
# of product code. It had always passed on the cluster only because submit-ci.sh sleeps 5s
# between tests -- and that hygiene belongs here, in the test that makes the mess, not in
# one particular driver that happens to run it.
wait_for_free_gpus() {
    local budget_s=$1 free
    for _ in $(seq 1 "$budget_s"); do
        free=$(free_gpu_count) || free=0
        if (( free >= GPUS )); then
            return 0
        fi
        sleep 1
    done
    return 1
}

# Refuse to start on a dirty machine rather than misreport a leftover allocation as a
# failure of the code under test.
#
# Counts how many GPUs are free and requires enough of them, rather than requiring that
# EVERY GPU on the host is free. The latter is an OR that gets likelier to trip the bigger
# the machine: this test pins itself to $GPUS GPUs, so on an 8-GPU CI runner one unrelated
# process on one GPU would abort it with six sitting idle -- a false failure that says
# nothing about the code. It still catches the case it was written for, a previous test in
# the lane leaking a VLLM::EngineCore, because that drops the free count below $GPUS.
#
# This runs BEFORE the training launch. It used to run after, which meant a refusal was
# not a refusal: the driver was already up, the EXIT trap shot it back down, and the log
# carried a bare "line 129: <pid> Killed" from bash job control on top of the real message.
if command -v nvidia-smi >/dev/null 2>&1 && ! wait_for_free_gpus "$GPU_WAIT_S"; then
    echo "[chaos] FAIL: need $GPUS free GPUs, still $(free_gpu_count) after ${GPU_WAIT_S}s; clean up before running"
    nvidia-smi --query-gpu=index,memory.used --format=csv
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv
    exit 1
fi

rm -rf "$EXP_DIR"
mkdir -p "$EXP_DIR" "$LOG_DIR"

cd "$PROJECT_ROOT"

# Enough steps that the run is still going when the kill lands.
# PYTHONUNBUFFERED: the harness detects progress by grepping RUN_LOG, and that only
# works if the driver actually writes to it. The actor prints with flush=True, but
# that just reaches the DRIVER -- Ray forwards actor output there, and the driver's
# own stdout is a redirected file, so Python block-buffers it. Job 5892910 wrote
# "train step 3/24" at 10:40:38 and the harness did not see it until 10:48:21, by
# which time the run had finished and there was nothing left to kill.
PYTHONUNBUFFERED=1 uv run "$PROJECT_ROOT"/examples/run_grpo_single_controller.py \
    policy.model_name=Qwen/Qwen3-0.6B \
    grpo.num_prompts_per_step=2 \
    grpo.num_generations_per_prompt=4 \
    policy.train_global_batch_size=8 \
    policy.train_micro_batch_size=1 \
    cluster.gpus_per_node=$GPUS \
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
    ++async_rl.rollout_failure.native.generation_timeout_s=60 \
    ++async_rl.rollout_failure.max_infra_attempts_per_prompt=3 \
    ++async_rl.stall_watchdog.interval_s=10 \
    ++async_rl.stall_watchdog.stall_timeout_s=180 \
    ++async_rl.stall_watchdog.stall_action=abort \
    ++async_rl.generation_fleet_health.enabled=true \
    ++async_rl.generation_fleet_health.probe_interval_s=5 \
    ++async_rl.generation_fleet_health.probe_timeout_s=2 \
    ++async_rl.generation_fleet_health.unhealthy_threshold=3 \
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
    # Signalling is not reclaiming: wait for the memory to actually come back before
    # handing the machine to the next test, which starts with no gap at all.
    if command -v nvidia-smi >/dev/null 2>&1 && ! wait_for_free_gpus "$GPU_SETTLE_S"; then
        echo "[chaos] WARN: ${GPU_SETTLE_S}s after cleanup, fewer than $GPUS GPUs are free:"
        nvidia-smi --query-compute-apps=pid,used_memory --format=csv
    fi
}
trap cleanup EXIT

echo "[chaos] training pid=$TRAIN_PID, waiting for the first train step..."
for _ in $(seq 1 120); do
    grep -q "train step 1/" "$RUN_LOG" 2>/dev/null && break
    kill -0 $TRAIN_PID 2>/dev/null || { echo "[chaos] FAIL: job died before the kill"; tail -40 "$RUN_LOG"; exit 1; }
    sleep 5
done
grep -q "train step 1/" "$RUN_LOG" || { echo "[chaos] FAIL: never reached a train step"; tail -40 "$RUN_LOG"; exit 1; }

# Which processes are the generation actors? Ask Ray, not ps.
#
# Anchored `ray::` title matching worked on a workstation and found ZERO actors on a GB200
# cluster (job 5861743) -- both chaos variants reported "job died before the kill" because
# the poll below never matched anything, and the run simply finished. Titles are a runtime
# implementation detail; the GCS actor table is authoritative.
#
# The STATE (idle vs serving) still comes from the title, because Ray only exposes the
# running method through the dashboard state API and init_ray disables the dashboard. So:
# discover actors authoritatively, then refine by title when titles are available, and say
# plainly when they are not rather than silently killing an arbitrary one.
ACTOR_RE='^ray::([A-Za-z0-9_.:-]+:)?[A-Za-z_]*GenerationWorker'
case "$VICTIM_STATE" in
    idle)    STATE_RE="${ACTOR_RE}\$" ;;
    serving) STATE_RE="${ACTOR_RE}\.generate_async\$" ;;
    any)     STATE_RE="" ;;
    *) echo "[chaos] FAIL: VICTIM_STATE must be idle, serving or any; got '$VICTIM_STATE'"; exit 1 ;;
esac

# Discover the actors ONCE, then poll their titles.
#
# Not once per 0.2s tick: each helper call is a full ray.init/shutdown of about three
# seconds, so polling it at tick rate would make VICTIM_WAIT_S=300 mean roughly 60 checks
# instead of 1500, and burn the budget on Ray handshakes rather than on watching for the
# state. The actor SET is stable once generation is up; only the STATE changes.
echo "[chaos] discovering generation actors (up to ${ACTOR_WAIT_S}s)..."
ACTORS=()
ATTEMPT=0
for _ in $(seq 1 $((ACTOR_WAIT_S / 3))); do
    kill -0 $TRAIN_PID 2>/dev/null || { echo "[chaos] FAIL: job died before any actor appeared"; break; }
    # Keep stderr: discarding it cost three debugging rounds, because a failed discovery
    # is indistinguishable from an empty fleet once the reason is thrown away.
    # stderr inline, into this log -- a separate artifact twice failed to survive to be
    # read. stdout (the pids) to a temp file, stderr to a variable.
    : > "$EXP_DIR/pids.tmp"
    ACTORS_ERR=$(timeout "$ACTOR_QUERY_TIMEOUT_S" uv run --no-sync python \
        "$SCRIPT_DIR/_find_generation_actors.py" 2>&1 >"$EXP_DIR/pids.tmp" || true)
    mapfile -t ACTORS < <(sort -n < "$EXP_DIR/pids.tmp")
    ATTEMPT=$((ATTEMPT + 1))
    if (( ${#ACTORS[@]} == 0 )) && [[ -n "$ACTORS_ERR" && $ATTEMPT -le 2 ]]; then
        echo "[chaos] discovery attempt $ATTEMPT found no actors; helper said:"
        echo "$ACTORS_ERR" | tail -25 | sed 's/^/[chaos]   /'
    fi
    (( ${#ACTORS[@]} > 0 )) && break
    sleep 3
done

if (( ${#ACTORS[@]} == 0 )); then
    echo "[chaos] FAIL: no generation actors found in ${ACTOR_WAIT_S}s"
    echo "[chaos] --- helper stderr from the LAST in-loop attempt (job was alive) ---"
    echo "${ACTORS_ERR:-<no attempt completed>}" | tail -25 | sed 's/^/[chaos]   /'
    echo "[chaos] --- what Ray reports now (job may already be gone) ---"
    uv run --no-sync python "$SCRIPT_DIR/_find_generation_actors.py" || true
    echo "[chaos] --- every process with 'eneration' in its command line ---"
    ps -eo pid=,args= 2>/dev/null | sed -E 's/^ *//' | grep -i "eneration" | grep -v grep | head -20
    echo "[chaos] --- last 60 lines of the training log ---"
    # "job died before any actor appeared" is ambiguous between a crash and simply
    # finishing all the steps, and those need opposite fixes. The log settles it.
    tail -60 "$RUN_LOG"
    exit 1
fi
echo "[chaos] generation actors: ${ACTORS[*]}"

echo "[chaos] waiting up to ${VICTIM_WAIT_S}s for one in state '$VICTIM_STATE'..."
VICTIM=""
TITLES_SEEN=0
for _ in $(seq 1 $((VICTIM_WAIT_S * 5))); do
    kill -0 $TRAIN_PID 2>/dev/null || { echo "[chaos] FAIL: job died before the kill"; break; }
    for pid in "${ACTORS[@]}"; do
        [[ -z "$pid" ]] && continue
        title=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | sed -E 's/ +$//')
        [[ "$title" =~ ^ray:: ]] && TITLES_SEEN=1
        if [[ -z "$STATE_RE" || "$title" =~ $STATE_RE ]]; then
            VICTIM=$pid; VICTIM_CMD=$title; break 2
        fi
    done
    sleep 0.2
done

if [[ -z "$VICTIM" ]]; then
    echo "[chaos] FAIL: no generation actor reached state '$VICTIM_STATE' in ${VICTIM_WAIT_S}s"
    echo "[chaos] --- what Ray reports ---"
    uv run --no-sync python "$SCRIPT_DIR/_find_generation_actors.py" || true
    if (( TITLES_SEEN == 0 )); then
        echo "[chaos] No actor ever presented a 'ray::' process title, so idle-vs-serving"
        echo "[chaos] cannot be distinguished on this platform. Re-run with VICTIM_STATE=any"
        echo "[chaos] to kill a generation actor without pinning the state -- the test still"
        echo "[chaos] asserts a bounded, attributable failure, it just stops distinguishing"
        echo "[chaos] the detection path from the in-flight-RPC path."
    fi
    echo "[chaos] --- every process with 'eneration' in its command line ---"
    # Unfiltered: the previous diagnostic grepped 'ray::' and so printed nothing exactly
    # when the ray:: assumption was itself the problem.
    ps -eo pid=,args= 2>/dev/null | sed -E 's/^ *//' | grep -i "eneration" | grep -v grep | head -20
    exit 1
fi

NOW_CMD=$(tr '\0' ' ' < "/proc/$VICTIM/cmdline" 2>/dev/null | sed -E 's/ +$//')
if [[ -n "$STATE_RE" && ! "$NOW_CMD" =~ $STATE_RE ]]; then
    echo "[chaos] FAIL: pid $VICTIM left state '$VICTIM_STATE' before the kill"
    echo "[chaos]   was: $VICTIM_CMD"
    echo "[chaos]   now: $NOW_CMD"
    exit 1
fi
echo "[chaos] killing generation actor pid=$VICTIM in state '$VICTIM_STATE'"
echo "[chaos]   cmdline: ${NOW_CMD:-<unavailable>}"
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
#
# NoSurvivingShards belongs here alongside GenerationFleetExhausted because losing the
# only shard can surface as EITHER, and which one wins is a matter of timing rather
# than of what went wrong:
#
#   GenerationFleetExhausted  too few shards SERVING, raised by the watchdog tick
#   NoSurvivingShards         no shard can take part in the REFIT, raised by the
#                             reconcile that runs before every weight sync
#
# The refit runs every step and the watchdog every watchdog.interval_s, so once
# detection got faster (probe_interval_s, 30s -> 5s) the reconcile started winning and
# this test failed -- while the run was doing exactly what it should: dying fast, with
# the cause named. Job 5931102.
#
# Deliberately NOT unified by having the reconcile call raise_if_exhausted() first.
# That looks tidier and is a trap: an aborted refit marks every surviving shard STALE
# via mark_weights_partial, so the serving set is legitimately empty for the moment
# between the abort and the retry -- and raise_if_exhausted() there would kill the run
# in the middle of the recovery that was about to succeed.
if grep -qE "RolloutRedispatchExhausted|GenerationUnavailable|RolloutStall|RolloutTimeout|GenerationFleetExhausted|NoSurvivingShards" "$RUN_LOG"; then
    echo "[chaos] PASS: bounded, attributable failure ${ELAPSED}s after the kill"
    grep -oE "RolloutRedispatchExhausted|GenerationUnavailable|RolloutStall|RolloutTimeout|GenerationFleetExhausted|NoSurvivingShards" "$RUN_LOG" | sort | uniq -c
    exit 0
fi

echo "[chaos] FAIL: job stopped but no typed rollout failure was reported"
tail -40 "$RUN_LOG"
exit 1
