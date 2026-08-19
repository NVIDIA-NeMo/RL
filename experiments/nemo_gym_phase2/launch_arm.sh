#!/bin/bash

set -euo pipefail

usage() {
  echo "Usage: $0 {direct|cache_aware|consistent_hash} {smoke|formal} REPEAT_ID [MATRIX_ID]" >&2
  exit 2
}

[[ $# -ge 3 && $# -le 4 ]] || usage
ARM=$1
MODE=$2
REPEAT_ID=$3
MATRIX_ID=${4:-$(date -u +%Y%m%dT%H%M%SZ)}

[[ $REPEAT_ID =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "REPEAT_ID must contain only letters, digits, dot, underscore, or hyphen" >&2
  exit 2
}
[[ $MATRIX_ID =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "MATRIX_ID must contain only letters, digits, dot, underscore, or hyphen" >&2
  exit 2
}

case "$ARM" in
  direct|cache_aware|consistent_hash) ;;
  *) usage ;;
esac
case "$MODE" in
  smoke)
    NUM_PROMPTS=4
    NUM_GENERATIONS=1
    WALLTIME=01:00:00
    ;;
  formal)
    NUM_PROMPTS=64
    NUM_GENERATIONS=4
    WALLTIME=02:00:00
    ;;
  *) usage ;;
esac

REPO=$(git rev-parse --show-toplevel)
: "${PHASE2_RUNTIME_ENV:?point to the prepared Python environment}"
: "${PHASE2_NEMO_GYM_ENV:?point to the prepared NeMo Gym Python environment}"
: "${PHASE2_VLLM_ENV_VERIFICATION:?point to the vLLM verification JSON}"
: "${PHASE2_NEMO_GYM_ENV_VERIFICATION:?point to the NeMo Gym verification JSON}"
: "${PHASE2_PYTHON_INSTALL_DIR:?point to the persistent uv Python directory}"
: "${PHASE2_VENV_DIR:?point to the lock-specific actor environment directory}"
: "${PHASE2_UV_CACHE_DIR:?point to the lock-specific shared uv cache}"
: "${PHASE2_CONTAINER:?point to the Slurm container image}"
: "${PHASE2_CONTAINER_DIGEST:?set the audited sha256:<digest>}"
: "${PHASE2_RL_INSIGHT_SOURCE:?point to the RL-Insight 0.2.1 source root}"
: "${PHASE2_PROMETHEUS_BIN:?point to Prometheus 2.54.1}"
: "${PHASE2_UV_BIN_DIR:?point to a directory containing uv 0.11.28}"
: "${PHASE2_WORKPLACE_DATA:?point to Workplace validation JSONL}"
: "${PHASE2_MODEL_SNAPSHOT:?point to the pinned HF snapshot}"
: "${PHASE2_MODEL_REVISION:?set the pinned HF revision}"

[[ $PHASE2_CONTAINER_DIGEST =~ ^sha256:[0-9a-fA-F]{64}$ ]] || {
  echo "PHASE2_CONTAINER_DIGEST must be sha256:<64 hex digits>" >&2
  exit 2
}
for required_path in \
  "$PHASE2_RUNTIME_ENV/bin/python" \
  "$PHASE2_RUNTIME_ENV/bin/ray" \
  "$PHASE2_NEMO_GYM_ENV/bin/python" \
  "$PHASE2_VLLM_ENV_VERIFICATION" \
  "$PHASE2_NEMO_GYM_ENV_VERIFICATION" \
  "$PHASE2_PYTHON_INSTALL_DIR" \
  "$PHASE2_VENV_DIR" \
  "$PHASE2_UV_CACHE_DIR" \
  "$PHASE2_CONTAINER" \
  "$PHASE2_RL_INSIGHT_SOURCE" \
  "$PHASE2_PROMETHEUS_BIN" \
  "$PHASE2_UV_BIN_DIR/uv" \
  "$PHASE2_WORKPLACE_DATA" \
  "$PHASE2_MODEL_SNAPSHOT/config.json" \
  "$PHASE2_MODEL_SNAPSHOT/tokenizer_config.json"; do
  [[ -e $required_path ]] || {
    echo "Required Phase 2 path does not exist: $required_path" >&2
    exit 1
  }
done
"$PHASE2_UV_BIN_DIR/uv" --version | grep -E '^uv 0\.11\.28([[:space:]]|$)'
[[ $(basename "$PHASE2_MODEL_SNAPSHOT") == "$PHASE2_MODEL_REVISION" ]] || {
  echo "Model snapshot basename must equal PHASE2_MODEL_REVISION" >&2
  exit 1
}

require_clean_tracked_source() {
  local component=$1
  local source_root=$2
  local status
  status=$(git -C "$source_root" status --porcelain=v1 --untracked-files=no)
  [[ -z $status ]] || {
    echo "$component has tracked changes; commit them before Phase 2 launch:" >&2
    echo "$status" >&2
    exit 1
  }
}

require_clean_tracked_source "NeMo RL" "$REPO"
require_clean_tracked_source "NeMo Gym" "$REPO/3rdparty/Gym-workspace/Gym"
require_clean_tracked_source "RL-Insight" "$PHASE2_RL_INSIGHT_SOURCE"
export UV_PYTHON_INSTALL_DIR=$PHASE2_PYTHON_INSTALL_DIR
export DG_USE_LOCAL_VERSION=0
# ray.sub starts the head and worker daemons before the driver command runs.
# Pin its explicit CLI because Pyxis restores the image PATH at container start.
export RAY_CLI=$PHASE2_RUNTIME_ENV/bin/ray
UV_CACHE_DIR=$PHASE2_UV_CACHE_DIR "$PHASE2_RUNTIME_ENV/bin/python" \
  "$REPO/experiments/nemo_gym_phase2/verify_runtime.py" \
  --repo "$REPO" \
  --environment "$PHASE2_RUNTIME_ENV" \
  --python-install-dir "$PHASE2_PYTHON_INSTALL_DIR" \
  --uv-bin "$PHASE2_UV_BIN_DIR/uv" \
  --output "$PHASE2_VLLM_ENV_VERIFICATION" \
  --label vllm \
  --extra vllm
UV_CACHE_DIR=$PHASE2_UV_CACHE_DIR "$PHASE2_NEMO_GYM_ENV/bin/python" \
  "$REPO/experiments/nemo_gym_phase2/verify_runtime.py" \
  --repo "$REPO" \
  --environment "$PHASE2_NEMO_GYM_ENV" \
  --python-install-dir "$PHASE2_PYTHON_INSTALL_DIR" \
  --uv-bin "$PHASE2_UV_BIN_DIR/uv" \
  --output "$PHASE2_NEMO_GYM_ENV_VERIFICATION" \
  --label nemo_gym \
  --extra nemo_gym \
  --group nemo_gym_router

RUNS_ROOT=${PHASE2_RUNS_ROOT:-$REPO/experiments/nemo_gym_phase2/runs}
RUN_ID=$MATRIX_ID-$ARM-r$REPEAT_ID
RUN_ROOT=$RUNS_ROOT/$RUN_ID
[[ ! -e "$RUN_ROOT" ]] || {
  echo "Refusing to reuse run directory: $RUN_ROOT" >&2
  exit 1
}
mkdir -p "$RUN_ROOT"

"$PHASE2_RUNTIME_ENV/bin/python" \
  "$REPO/experiments/nemo_gym_phase2/prepare_workload.py" \
  --source "$PHASE2_WORKPLACE_DATA" \
  --workload "$RUN_ROOT/workload.jsonl" \
  --warmup "$RUN_ROOT/warmup-workload.jsonl" \
  --num-prompts "$NUM_PROMPTS" \
  --warmup-requests 1 > "$RUN_ROOT/workload-manifest.json"

export PHASE2_REPO=$REPO
export PHASE2_RUN_ROOT=$RUN_ROOT
export PHASE2_ARM=$ARM
export PHASE2_RUN_ID=$RUN_ID
export PHASE2_REPEAT_ID=$REPEAT_ID
export PHASE2_MODE=$MODE
export PHASE2_NUM_PROMPTS=$NUM_PROMPTS
export PHASE2_NUM_GENERATIONS=$NUM_GENERATIONS
export BASE_LOG_DIR=$RUN_ROOT/ray
export COMMAND=$REPO/experiments/nemo_gym_phase2/run_arm.sh
export CONTAINER=$PHASE2_CONTAINER
export GPUS_PER_NODE=8
export HF_HOME=${PHASE2_HF_HOME:-$(dirname "$(dirname "$(dirname "$(dirname "$PHASE2_MODEL_SNAPSHOT")")")")}
export HF_HUB_OFFLINE=1
export MOUNTS=${PHASE2_MOUNTS:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/nliang:/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/nliang}
export NEMO_RL_VENV_DIR=$PHASE2_VENV_DIR
export UV_CACHE_DIR_OVERRIDE=$PHASE2_UV_CACHE_DIR
export UV_PYTHON_INSTALL_DIR=$PHASE2_PYTHON_INSTALL_DIR
unset NEMO_RL_PY_EXECUTABLES_SYSTEM NEMO_RL_SYSTEM_PY_EXECUTABLE
export RESULTS_DIR=$RUN_ROOT/results
export TRANSFORMERS_OFFLINE=1

{
  printf 'PHASE2_REPO=%q\n' "$PHASE2_REPO"
  printf 'PHASE2_RUN_ROOT=%q\n' "$PHASE2_RUN_ROOT"
  printf 'PHASE2_ARM=%q\n' "$PHASE2_ARM"
  printf 'PHASE2_RUN_ID=%q\n' "$PHASE2_RUN_ID"
  printf 'PHASE2_REPEAT_ID=%q\n' "$PHASE2_REPEAT_ID"
  printf 'PHASE2_MODE=%q\n' "$PHASE2_MODE"
  printf 'PHASE2_NUM_PROMPTS=%q\n' "$PHASE2_NUM_PROMPTS"
  printf 'PHASE2_NUM_GENERATIONS=%q\n' "$PHASE2_NUM_GENERATIONS"
  printf 'PHASE2_RUNTIME_ENV=%q\n' "$PHASE2_RUNTIME_ENV"
  printf 'PHASE2_RAY_CLI=%q\n' "$PHASE2_RUNTIME_ENV/bin/ray"
  printf 'PHASE2_NEMO_GYM_ENV=%q\n' "$PHASE2_NEMO_GYM_ENV"
  printf 'PHASE2_VLLM_ENV_VERIFICATION=%q\n' "$PHASE2_VLLM_ENV_VERIFICATION"
  printf 'PHASE2_NEMO_GYM_ENV_VERIFICATION=%q\n' "$PHASE2_NEMO_GYM_ENV_VERIFICATION"
  printf 'PHASE2_PYTHON_INSTALL_DIR=%q\n' "$PHASE2_PYTHON_INSTALL_DIR"
  printf 'PHASE2_VENV_DIR=%q\n' "$PHASE2_VENV_DIR"
  printf 'PHASE2_UV_CACHE_DIR=%q\n' "$PHASE2_UV_CACHE_DIR"
  printf 'PHASE2_RL_INSIGHT_SOURCE=%q\n' "$PHASE2_RL_INSIGHT_SOURCE"
  printf 'PHASE2_PROMETHEUS_BIN=%q\n' "$PHASE2_PROMETHEUS_BIN"
  printf 'PHASE2_UV_BIN_DIR=%q\n' "$PHASE2_UV_BIN_DIR"
  printf 'PHASE2_CONTAINER_DIGEST=%q\n' "$PHASE2_CONTAINER_DIGEST"
  printf 'PHASE2_MODEL_SNAPSHOT=%q\n' "$PHASE2_MODEL_SNAPSHOT"
  printf 'PHASE2_MODEL_REVISION=%q\n' "$PHASE2_MODEL_REVISION"
  printf 'PHASE2_CONTAINER=%q\n' "$PHASE2_CONTAINER"
  printf 'PHASE2_WORKPLACE_DATA=%q\n' "$PHASE2_WORKPLACE_DATA"
  printf 'HF_HOME=%q\n' "$HF_HOME"
  printf 'MOUNTS=%q\n' "$MOUNTS"
  printf '%q\n' "$COMMAND"
} > "$RUN_ROOT/command.txt"

JOB_ID=$(sbatch --parsable \
  --nodes=1 \
  --exclusive \
  --account="${PHASE2_SLURM_ACCOUNT:-coreai_dlalgo_nemorl}" \
  --partition="${PHASE2_SLURM_PARTITION:-batch_short}" \
  --time="$WALLTIME" \
  --gres=gpu:8 \
  --job-name="p2-$ARM-$MODE-r$REPEAT_ID" \
  --output="$RUN_ROOT/slurm-%j.out" \
  "$REPO/ray.sub")
printf '%s\n' "$JOB_ID" > "$RUN_ROOT/job_id"
printf '%s\n' "$JOB_ID"
