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
