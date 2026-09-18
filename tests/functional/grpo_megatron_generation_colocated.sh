#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath $SCRIPT_DIR/../..)
# Mark the current repo as safe, since wandb fetches metadata about the repo
git config --global --add safe.directory $PROJECT_ROOT

set -eou pipefail

EXP_NAME=$(basename $0 .sh)
EXP_DIR=$SCRIPT_DIR/$EXP_NAME
LOG_DIR=$EXP_DIR/logs
JSON_METRICS=$EXP_DIR/metrics.json
RUN_LOG=$EXP_DIR/run.log
DIAGNOSTIC_DIR=$EXP_DIR/diagnostics
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}
export PYTHONFAULTHANDLER=1

rm -rf $EXP_DIR $LOG_DIR
mkdir -p $EXP_DIR $LOG_DIR "$DIAGNOSTIC_DIR"

cd $PROJECT_ROOT

stage() {
    STAGE=$1
    printf '[DEBUG-4138] time=%s stage=%s\n' "$(date -u +%FT%TZ)" "$STAGE" \
        | tee -a "$DIAGNOSTIC_DIR/status.log" || true
}

{
    printf '[DEBUG-4138] RL HEAD and Bridge/MLM gitlinks\n'
    git rev-parse HEAD
    git ls-tree HEAD 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
    git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge ls-tree HEAD 3rdparty/Megatron-LM
    sha256sum uv.lock
    nvidia-smi --query-gpu=name,driver_version --format=csv
    printf '[DEBUG-4138] Driver environment package versions\n'
    uv run --no-sync python -c 'from importlib.metadata import version; print("\n".join(f"{name}={version(name)}" for name in ("ray", "torch", "coverage")))'
} > "$DIAGNOSTIC_DIR/versions.txt" 2>&1 || true
if [[ -r /sys/fs/cgroup/memory.events ]]; then
    cp /sys/fs/cgroup/memory.events "$DIAGNOSTIC_DIR/memory.events.before" || true
fi

# cluster.segment_size only engages when Ray nodes carry nvlink_domain_* labels,
# which ray.sub probes from `nvidia-smi -q` ClusterUUID on NVLink-fabric clusters
# (e.g. GB200 NVL72); CI runners have none. Pre-start a Ray head with a synthetic
# domain label so init_ray() attaches to it (externally managed cluster) and the
# topology-aware megatron placement path runs for real.
cleanup() {
    local exit_code=$?
    trap - EXIT
    set +e
    printf '[DEBUG-4138] time=%s stage=%s exit_code=%s\n' \
        "$(date -u +%FT%TZ)" "$STAGE" "$exit_code" | tee -a "$DIAGNOSTIC_DIR/status.log"
    if [[ -r /sys/fs/cgroup/memory.events ]]; then
        cp /sys/fs/cgroup/memory.events "$DIAGNOSTIC_DIR/memory.events.after"
    fi
    # Post-hoc only: nothing above runs before the training, so the run happens
    # under exactly the conditions that produced the failures on this baseline.
    # Everything here is best-effort and must never change $exit_code -- the
    # training's own 139 is the primary datum.
    if [[ "$exit_code" -ne 0 ]]; then
        {
            printf 'training_exit_code=%s\n' "$exit_code"
            printf 'core_pattern=%s\n' "$(cat /proc/sys/kernel/core_pattern 2>&1)"
            printf 'core_uses_pid=%s\n' "$(cat /proc/sys/kernel/core_uses_pid 2>&1)"
            printf 'ulimit_c=%s\n' "$(ulimit -c)"
            printf 'libc6=%s\n' "$(dpkg-query -W -f='${Version}' libc6 2>/dev/null || printf unknown)"
        } > "$DIAGNOSTIC_DIR/forensics.txt" 2>&1
        mkdir -p "$DIAGNOSTIC_DIR/fx"

        # Record what the crash left behind BEFORE touching the package set: an
        # install that fails or drags libraries with it must not be able to
        # destroy or misrepresent the evidence.
        shopt -s nullglob
        cores=("$PROJECT_ROOT"/core*)
        shopt -u nullglob
        for c in "${cores[@]}"; do
            [[ -f $c ]] && printf 'core_found=%s size=%s mtime=%s\n' \
                "$c" "$(stat -c %s "$c")" "$(stat -c %Y "$c")" >> "$DIAGNOSTIC_DIR/forensics.txt"
        done
        if [[ ${#cores[@]} -eq 0 ]]; then
            printf 'core_found=NONE\n' >> "$DIAGNOSTIC_DIR/forensics.txt"
        fi

        # Compress the evidence before doing anything that can hang. Every step
        # below this point is fallible in a way that takes the whole job with it:
        # an apt mirror that never answers runs the job into the CI timeout, and
        # a killed job uploads nothing at all. Ordering the cheap, self-contained
        # captures first means the worst case is a run with no backtrace rather
        # than a run with no evidence.
        #
        # This is not "the core is safe". Compression only puts it in the
        # workspace; it survives only if the artifact upload, which runs after
        # this function returns, succeeds. The key below says awaiting-upload for
        # that reason.
        #
        # dmesg leads because it is the quickest source that can name a faulting
        # library ("segfault at .. in libfoo.so"). Treat it as a lead, not a
        # verdict: reading it from inside the container has failed before, and
        # the library in the message is where the fault surfaced, not
        # necessarily what is responsible.
        dmesg > "$DIAGNOSTIC_DIR/fx/dmesg.txt" 2>&1
        for c in "${cores[@]}"; do
            [[ -f $c ]] || continue
            base=$(basename "$c")
            # Bounded like everything else that can stall: a core that will not
            # compress in time leaves the remaining evidence and the upload
            # intact instead of running the job into its limit. A killed gzip
            # exits non-zero, so the truncated output lands in the failure
            # branch and is removed rather than passed off as an archive.
            if timeout -k 60s 900s gzip -1 -c "$c" > "$DIAGNOSTIC_DIR/fx/$base.gz" 2>/dev/null; then
                kept=$(stat -c %s "$DIAGNOSTIC_DIR/fx/$base.gz" 2>/dev/null || printf 0)
                if [[ "$kept" -gt 0 && "$kept" -le 3000000000 ]]; then
                    printf 'core_compressed=%s.gz bytes=%s state=awaiting-upload\n' "$base" "$kept" \
                        >> "$DIAGNOSTIC_DIR/forensics.txt"
                else
                    rm -f "$DIAGNOSTIC_DIR/fx/$base.gz"
                    printf 'core_NOT_COMPRESSED=%s compressed_bytes=%s -- exceeds artifact budget, evidence lost on cleanup\n' \
                        "$base" "$kept" >> "$DIAGNOSTIC_DIR/forensics.txt"
                fi
            else
                rm -f "$DIAGNOSTIC_DIR/fx/$base.gz"
                printf 'core_NOT_COMPRESSED=%s -- compression failed or timed out, evidence lost on cleanup\n' \
                    "$base" >> "$DIAGNOSTIC_DIR/forensics.txt"
            fi
        done

        # gdb is absent from this image. Installing it now cannot change a crash
        # that already happened, but it can move the libraries gdb would resolve
        # the core against, so the hold is a precondition, not a best effort: if
        # it does not take, skip the install and keep the core instead of
        # producing a backtrace read against libraries the crash never used.
        libc6_before_fx=$(dpkg-query -W -f='${Version}' libc6 2>/dev/null || printf unknown)
        if [[ ${#cores[@]} -eq 0 ]]; then
            printf 'gdb_install=skipped-no-core\n' >> "$DIAGNOSTIC_DIR/forensics.txt"
        elif command -v gdb > /dev/null 2>&1; then
            printf 'gdb_install=already-present\n' >> "$DIAGNOSTIC_DIR/forensics.txt"
        elif ! apt-mark hold libc6 libc-bin libc6-dev > "$DIAGNOSTIC_DIR/fx/apt-hold.log" 2>&1; then
            printf 'gdb_install=skipped-hold-failed\n' >> "$DIAGNOSTIC_DIR/forensics.txt"
        else
            # Bounded: an unreachable mirror must cost us a backtrace, not the
            # job. The core is already compressed above either way. -k because
            # plain timeout only sends TERM, and a process that ignores it goes
            # on waiting -- the bound has to be enforceable to be worth writing.
            {
                timeout -k 30s 600s apt-get update -qq \
                    && timeout -k 30s 900s apt-get install -y -qq --no-install-recommends gdb
            } > "$DIAGNOSTIC_DIR/fx/gdb-install.log" 2>&1
            apt-mark unhold libc6 libc-bin libc6-dev >> "$DIAGNOSTIC_DIR/fx/apt-hold.log" 2>&1
            printf 'gdb_install=attempted\n' >> "$DIAGNOSTIC_DIR/forensics.txt"
        fi
        libc6_after_fx=$(dpkg-query -W -f='${Version}' libc6 2>/dev/null || printf unknown)
        printf 'gdb=%s libc6_before=%s libc6_after=%s\n' \
            "$(command -v gdb || printf none)" "$libc6_before_fx" "$libc6_after_fx" \
            >> "$DIAGNOSTIC_DIR/forensics.txt"

        # gdb needs the executable the core came from. Resolving it used to sit
        # above the compression and had no bound, which was the same defect as
        # the unbounded install one step further up: a stalled `uv run` meant the
        # core was never reached at all. It is bounded now, and it runs here
        # because nothing before this point needs it. An unresolved interpreter
        # means no backtrace -- reading a core against the wrong executable
        # produces frames that look real and are not.
        PY_EXE_FX=$(timeout -k 15s 120s uv run --no-sync python -c 'import sys; print(sys.executable)' 2>/dev/null) \
            || PY_EXE_FX=""
        printf 'py_exe=%s\n' "${PY_EXE_FX:-unresolved}" >> "$DIAGNOSTIC_DIR/forensics.txt"

        for c in "${cores[@]}"; do
            [[ -f $c ]] || continue
            base=$(basename "$c")
            gdb_rc=127
            if [[ -n "$PY_EXE_FX" ]] && command -v gdb > /dev/null 2>&1; then
                gdb_rc=0
                timeout -k 30s 900s gdb -batch -q -ex 'thread apply all bt' -ex 'info sharedlibrary' \
                    "$PY_EXE_FX" "$c" > "$DIAGNOSTIC_DIR/fx/gdb-$base.txt" 2>&1 || gdb_rc=$?
            fi
            # A backtrace is only trustworthy if the binaries gdb read it
            # against are the ones the crash actually used. Absence of warnings
            # does not show that: gdb prints frames happily after saying the
            # core does not match, and says nothing at all when it silently had
            # no library map to check. Require positive evidence -- an unchanged
            # libc across the install, a known libc version, a loaded shared
            # library map, frames, and a clean gdb -- and call everything else
            # unreliable.
            parse_note=""
            if [[ -z "$PY_EXE_FX" ]]; then
                parse_note="python-path-unresolved"
            elif [[ "$gdb_rc" -ne 0 ]]; then
                parse_note="gdb-rc-$gdb_rc"
            elif [[ "$libc6_before_fx" == unknown || "$libc6_after_fx" == unknown ]]; then
                parse_note="libc-version-unknown"
            elif [[ "$libc6_before_fx" != "$libc6_after_fx" ]]; then
                parse_note="libc-moved-during-install"
            elif ! grep -q '^#0 ' "$DIAGNOSTIC_DIR/fx/gdb-$base.txt" 2>/dev/null; then
                parse_note="no-frames"
            elif grep -qiE 'may not match|No shared library information|could not( be)? read' \
                    "$DIAGNOSTIC_DIR/fx/gdb-$base.txt" 2>/dev/null; then
                parse_note="binary-or-library-mismatch"
            elif ! grep -qE '^0x[0-9a-f]+ +0x[0-9a-f]+ +(Yes|No)' \
                    "$DIAGNOSTIC_DIR/fx/gdb-$base.txt" 2>/dev/null; then
                parse_note="no-library-map"
            fi
            # Even a clean parse is not a verified one. gdb's "Yes" column means
            # it read symbols, not that the file on disk is the build the crash
            # used, and a mixed map can show Yes for libc while showing No for
            # the library the stack actually blames. Deciding that needs
            # Build-ID matching, which this round does not do -- so report the
            # stack as obtained with identity unverified, and never let that
            # verdict skip the archive.
            #
            # grep -c prints the count for a match (status 0) and for no match
            # (status 1) alike; only a read error (status 2) leaves it absent.
            # The `|| printf 0` that used to guard these fired on the no-match
            # status, appending a second zero to a count grep had already
            # printed, and the variable became "0\n0" -- which split the
            # single-line record in two.
            lib_yes=$(grep -cE '^0x[0-9a-f]+ +0x[0-9a-f]+ +Yes' "$DIAGNOSTIC_DIR/fx/gdb-$base.txt" 2>/dev/null)
            [[ $? -le 1 ]] || lib_yes=unknown
            lib_no=$(grep -cE '^0x[0-9a-f]+ +0x[0-9a-f]+ +No' "$DIAGNOSTIC_DIR/fx/gdb-$base.txt" 2>/dev/null)
            [[ $? -le 1 ]] || lib_no=unknown
            if [[ -z "$parse_note" ]]; then
                printf 'core_parsed=%s stack=obtained identity=unverified libs_with_symbols=%s libs_without=%s\n' \
                    "$base" "$lib_yes" "$lib_no" >> "$DIAGNOSTIC_DIR/forensics.txt"
            else
                printf 'core_parsed=%s unreliable=%s libs_with_symbols=%s libs_without=%s\n' \
                    "$base" "$parse_note" "$lib_yes" "$lib_no" >> "$DIAGNOSTIC_DIR/forensics.txt"
            fi
            # No archiving here: the core was secured before the install, so a
            # verdict reached at this point can never cost us the evidence.
        done
    fi
    if [[ "$exit_code" -ne 0 && -d /tmp/ray/session_latest/logs ]]; then
        mkdir -p "$DIAGNOSTIC_DIR/ray"
        cp -a /tmp/ray/session_latest/logs/. "$DIAGNOSTIC_DIR/ray/" \
            2> "$DIAGNOSTIC_DIR/ray-copy.log"
    fi
    uv run ray stop --force > "$DIAGNOSTIC_DIR/ray-stop.log" 2>&1
    exit "$exit_code"
}
stage ray_start
trap cleanup EXIT
uv run ray stop --force > "$DIAGNOSTIC_DIR/ray-stop-before.log" 2>&1 || true # don't attach to a stale cluster
uv run ray start --head --disable-usage-stats \
    --resources='{"nvlink_domain_ci_synthetic": 1, "topo_rank": 1}'

stage training
# Capture both statuses before any command overwrites PIPESTATUS.
set +e
uv run coverage run -a --data-file=$PROJECT_ROOT/tests/.coverage --source=$PROJECT_ROOT/nemo_rl \
    $PROJECT_ROOT/examples/run_grpo.py \
    --config $PROJECT_ROOT/examples/configs/grpo_math_1B_megatron.yaml \
    policy.model_name=Qwen/Qwen2.5-0.5B \
    grpo.num_prompts_per_step=2 \
    grpo.num_generations_per_prompt=4 \
    policy.train_global_batch_size=4 \
    policy.logprob_batch_size=4 \
    policy.train_micro_batch_size=1 \
    policy.generation.backend=megatron \
    policy.generation.refit_transport=mcore \
    policy.generation.mcore_generation_config.refit_backend=nccl \
    cluster.gpus_per_node=2 \
    cluster.segment_size=1 \
    grpo.max_num_steps=2 \
    logger.tensorboard_enabled=true \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=false \
    logger.monitor_gpus=true \
    checkpointing.enabled=false \
    $@ \
    2>&1 | tee $RUN_LOG
TRAIN_PIPESTATUS=("${PIPESTATUS[@]}")
set -e
printf '[DEBUG-4138] stage=training coverage_uv_exit=%s tee_exit=%s\n' \
    "${TRAIN_PIPESTATUS[0]}" "${TRAIN_PIPESTATUS[1]}" | tee -a "$DIAGNOSTIC_DIR/status.log" || true
# Preserve pipefail: the rightmost nonzero status is the pipeline status.
TRAIN_EXIT=${TRAIN_PIPESTATUS[1]}
if [[ "$TRAIN_EXIT" -eq 0 ]]; then
    TRAIN_EXIT=${TRAIN_PIPESTATUS[0]}
fi
if [[ "$TRAIN_EXIT" -ne 0 ]]; then
    exit "$TRAIN_EXIT"
fi

stage topology
# Guard against the silent fallback: with no (or unreadable) domain labels the run
# would succeed without ever exercising the topology placement path under test.
grep -q "Topology-aware allocation" $RUN_LOG || {
    echo "ERROR: topology-aware allocation did not engage (no segment selection logged)" >&2
    exit 1
}
# NOTE: `! grep` is exempt from `set -e`, hence the explicit if.
if grep -q "no NVLink domain info" $RUN_LOG; then
    echo "ERROR: segment_size fell back to unordered allocation" >&2
    exit 1
fi

stage metrics_export
uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

stage metrics_check
uv run tests/check_metrics.py $JSON_METRICS \
    'max(data["train/token_mult_prob_error"]) < 1.05'

stage async_counter
# This counter includes both overlapped and non-overlapped async orderings;
# a positive value confirms that async_sched_mode reached and ran in the engine.
ASYNC_SCHED_STEPS=$(grep -o 'mcore async scheduling steps (cumul): [0-9]*' $RUN_LOG | grep -o '[0-9]*$' | sort -n | tail -1 || true)
if [[ -z "${ASYNC_SCHED_STEPS:-}" ]]; then
    echo "FAIL: async scheduling counter not found"
    exit 1
fi
if [[ "$ASYNC_SCHED_STEPS" -eq 0 ]]; then
    echo "FAIL: async scheduler reported 0 scheduling steps"
    exit 1
fi
echo "async scheduling steps: $ASYNC_SCHED_STEPS"
stage completed
