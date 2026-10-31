#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
# Mirrors grpo-llama3.2-1b-instruct-1n8g-megatron.sh (delegated base).
NUM_NODES=1
STEPS_PER_RUN=20
MAX_STEPS=20
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
# The full run (20 steps, profile window, report write-out) measured about 13 minutes.
NUM_MINUTES=30
# ===== END CONFIG =====

# End-to-end check of nsys profiling. Runs the base recipe under this wrapper's identity
# (own log dir and YAML, which sets vLLM TP=2 and the async engine) with NeMo-RL profiling
# on, then counts the .nsys-rep reports per worker class. Exit code alone proves nothing:
# a run whose inner vLLM workers write no report still completes.
#
# NRL_NSYS_EXTRA_OPTIONS is passed through unchanged, so a platform can adjust nsys flags.
export NRL_NSYS_WORKER_PATTERNS="*policy*,*vllm*"
export NRL_NSYS_PROFILE_STEP_RANGE="2:4"
# Inner vLLM reports are written after the profile window closes, while the workers keep
# running, so the run continues for MAX_STEPS past the window. Once the driver exits, no
# new report is finalized; the short wait below only covers file-system latency.
NSYS_REPORT_TIMEOUT_S=${NSYS_REPORT_TIMEOUT_S:-120}
# Single-node recipe: every Ray worker writes to this node's Ray session dirs. Ray's temp
# dir is $RAY_TMPDIR, else $TMPDIR, else /tmp (plus /ray); search all of them.
if [[ -n "${RAY_SESSION_ROOT:-}" ]]; then
    REPORT_ROOTS=("$RAY_SESSION_ROOT")
else
    mapfile -t REPORT_ROOTS < <(printf '%s/ray\n' "${RAY_TMPDIR:-/tmp}" "${TMPDIR:-/tmp}" /tmp | awk '!seen[$0]++')
fi

if ! command -v nsys >/dev/null; then
    echo "[ERROR] nsys is not on PATH; this test needs Nsight Systems on the node"
    exit 1
fi
echo "[INFO] $(nsys --version)"

cd $PROJECT_ROOT

# Expected counts from the resolved config (recipe + CLI overrides). Generation is
# colocated, so there is one Megatron policy worker per GPU and, at vLLM TP>1, one inner
# vLLM TP worker per GPU (engines x TP x PP). Fails before the run if the config cannot
# produce inner workers.
EXPECTED_COUNTS=$(uv run python - "$CONFIG_PATH" "$@" <<'EOF' | tail -n 1
import sys

from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)

register_omegaconf_resolvers()
cfg = load_config(sys.argv[1])
if len(sys.argv) > 2:
    cfg = parse_hydra_overrides(cfg, sys.argv[2:])
vllm_cfg = cfg.policy.generation.vllm_cfg
assert cfg.cluster.num_nodes == 1, "counts reports on this node only"
assert cfg.policy.generation.colocated.enabled, "expects colocated generation"
assert vllm_cfg.tensor_parallel_size > 1, "needs vLLM TP>1 (TP=1 has no inner workers)"
gpus = cfg.cluster.num_nodes * cfg.cluster.gpus_per_node
model_parallel = vllm_cfg.tensor_parallel_size * vllm_cfg.pipeline_parallel_size
engines = gpus // model_parallel
print(gpus, engines * model_parallel)
EOF
)
read -r EXPECTED_POLICY EXPECTED_INNER <<<"$EXPECTED_COUNTS"
echo "[INFO] Expecting $EXPECTED_POLICY megatron_policy_worker and $EXPECTED_INNER vllm_tp_worker reports"

START_MARKER=$EXP_DIR/.nsys_start_marker
touch "$START_MARKER"

list_reports() {  # extra args are passed to find
    find "${REPORT_ROOTS[@]}" -path '*/logs/nsight/*' -name '*.nsys-rep' -newer "$START_MARKER" "$@" 2>/dev/null
}
count_reports() {
    list_reports | grep -c "/${1}_${NRL_NSYS_PROFILE_STEP_RANGE}_[0-9]*\.nsys-rep$" || true
}
print_reports() {
    echo "[INFO] Searched ${REPORT_ROOTS[*]}; nsys reports per worker class:"
    list_reports | sed -E 's#.*/##; s/_[0-9]+:[0-9]+_[0-9]+\.nsys-rep$//' | sort | uniq -c
}
# List what was written on any failure, including a failed training run.
trap 'rc=$?; [[ $rc -ne 0 ]] && print_reports; exit $rc' EXIT

# Run the base recipe under this wrapper's identity (own log dir, YAML and wandb name).
export EXP_NAME
bash $SCRIPT_DIR/grpo-llama3.2-1b-instruct-1n8g-megatron.sh \
    grpo.max_num_steps=$MAX_STEPS \
    checkpointing.enabled=False \
    "$@"

MAX_RECORDED_STEP=$(jq -r 'if has("train/loss") then (."train/loss" | keys | map(tonumber) | max // 0) else 0 end' "$JSON_METRICS")
if [[ $MAX_RECORDED_STEP -lt $MAX_STEPS ]]; then
    echo "[ERROR] Expected train/loss through step $MAX_STEPS, found step $MAX_RECORDED_STEP"
    exit 1
fi

DEADLINE=$((SECONDS + NSYS_REPORT_TIMEOUT_S))
while true; do
    NUM_POLICY=$(count_reports megatron_policy_worker)
    NUM_INNER=$(count_reports vllm_tp_worker)
    if [[ $NUM_POLICY -ge $EXPECTED_POLICY && $NUM_INNER -ge $EXPECTED_INNER ]] || [[ $SECONDS -ge $DEADLINE ]]; then
        break
    fi
    sleep 10
done
NUM_OUTER=$(count_reports vllm_async_generation_worker)
print_reports

if [[ -n $(list_reports -name 'worker_process_*') ]]; then
    echo "[ERROR] Found worker_process_*.nsys-rep: the inner vLLM workers used vLLM's stock nsight" \
        "config, so the vllm.general_plugins entry point nemo_rl_nsight_inner did not load in the" \
        "vLLM environment (reinstall nemo-rl there; keep it in VLLM_PLUGINS if that is set)"
    exit 1
fi
# nsys keeps in-progress captures (.qdstrm) in its temp dir until the report is written;
# expand the glob now, after the run, so temp dirs created by the run are searched too.
shopt -s nullglob; NSYS_TMP_DIRS=("${TMPDIR:-/tmp}"/nsys-*); shopt -u nullglob
if [[ -n $(find "${NSYS_TMP_DIRS[@]}" "${REPORT_ROOTS[@]}" -name '*.qdstrm' -newer "$START_MARKER" 2>/dev/null) ]]; then
    echo "[ERROR] Found unfinished .qdstrm captures: reports were not written before the run" \
        "ended; run more steps after NRL_NSYS_PROFILE_STEP_RANGE"
    exit 1
fi
if [[ $NUM_POLICY -ne $EXPECTED_POLICY ]]; then
    echo "[ERROR] Expected $EXPECTED_POLICY megatron_policy_worker reports, found $NUM_POLICY"
    exit 1
fi
if [[ $NUM_INNER -ne $EXPECTED_INNER ]]; then
    echo "[ERROR] Expected $EXPECTED_INNER vllm_tp_worker reports (one per inner vLLM TP rank), found $NUM_INNER"
    exit 1
fi
# The outer async actor only forwards cudaProfilerStart/Stop to the inner workers, so it
# never opens a capture of its own.
if [[ $NUM_OUTER -ne 0 ]]; then
    echo "[ERROR] Expected no vllm_async_generation_worker reports at vLLM TP>1, found $NUM_OUTER"
    exit 1
fi
if [[ -n $(list_reports -size 0) ]]; then
    echo "[ERROR] Found empty .nsys-rep reports"
    exit 1
fi

# One inner report must open and contain CUDA API calls from the profile window.
INNER_REPORT=$(list_reports -name "vllm_tp_worker_*" | head -n 1)
nsys export --type=sqlite --force-overwrite=true -o "$EXP_DIR/inner_report.sqlite" "$INNER_REPORT"
NUM_CUDA_API_CALLS=$(uv run python - "$EXP_DIR/inner_report.sqlite" <<'EOF' | tail -n 1
import sqlite3
import sys

con = sqlite3.connect(sys.argv[1])
tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
table = "CUPTI_ACTIVITY_KIND_RUNTIME"
print(con.execute(f"select count(*) from {table}").fetchone()[0] if table in tables else 0)
EOF
)
if [[ $NUM_CUDA_API_CALLS -le 0 ]]; then
    echo "[ERROR] $INNER_REPORT has no CUDA API calls: its capture range never opened"
    exit 1
fi
echo "[INFO] $(basename "$INNER_REPORT"): $NUM_CUDA_API_CALLS CUDA API calls"
echo "[INFO] nsys report check passed"
