#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
RL_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd -P)"
RAY_SUB="$RL_ROOT/ray.sub"

usage() {
  cat <<EOF
Usage: $0 [ENV_FILE]

ENV_FILE defaults to $SCRIPT_DIR/.env. Start from env.example and keep the
result outside git because it contains sandbox API credentials.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if (( $# > 1 )); then
  usage >&2
  exit 2
fi

ENV_FILE="${1:-$SCRIPT_DIR/.env}"
if [[ ! -r "$ENV_FILE" ]]; then
  echo "Environment file is not readable: $ENV_FILE" >&2
  usage >&2
  exit 2
fi

set -a
source "$ENV_FILE"
set +a
if [[ -n "${OSWORLD_GRPO_VAL_DATA_OVERRIDE:-}" ]]; then
  export OSWORLD_GRPO_VAL_DATA="$OSWORLD_GRPO_VAL_DATA_OVERRIDE"
fi

OSWORLD_DRIVER_SCRIPT="${OSWORLD_DRIVER_SCRIPT:-driver.sh}"
OSWORLD_NUM_NODES="${OSWORLD_NUM_NODES:-4}"
OSWORLD_RECIPE="${OSWORLD_RECIPE:-flash-b8n8-dr-grpo}"
case "$OSWORLD_RECIPE" in
  molt-b8n8-checkpoint|flash-b8n8-reinforce|flash-b8n8-dr-grpo|flash-b17n16-dr-grpo|flash-b17n16-reinforce) ;;
  *)
    echo "Unsupported OSWORLD_RECIPE: $OSWORLD_RECIPE" >&2
    exit 2
    ;;
esac
OSWORLD_JOB_NAME="${OSWORLD_JOB_NAME:-osw-v2-$OSWORLD_RECIPE}"
case "$OSWORLD_DRIVER_SCRIPT" in
  driver.sh|eval_driver.sh) ;;
  *)
    echo "Unsupported OSWORLD_DRIVER_SCRIPT: $OSWORLD_DRIVER_SCRIPT" >&2
    exit 2
    ;;
esac
if [[ "$OSWORLD_DRIVER_SCRIPT" != "eval_driver.sh" ]]; then
  unset OSWORLD_EVAL_MODE
fi

export OSWORLD_SANDBOX_PROVIDER="${OSWORLD_SANDBOX_PROVIDER:-opensandbox}"
case "$OSWORLD_SANDBOX_PROVIDER" in
  opensandbox)
    if [[ -z "${OPENSANDBOX_DOMAIN:-}" && -n "${OPENSANDBOX_BASE_URL:-}" ]]; then
      opensandbox_host="${OPENSANDBOX_BASE_URL#*://}"
      export OPENSANDBOX_DOMAIN="${opensandbox_host%%/*}"
    fi
    export OPENSANDBOX_API_KEY="${OPENSANDBOX_API_KEY:-}"
    ;;
  agentenv)
    export AGENTENV_TEMPLATE="${AGENTENV_TEMPLATE:-osworld-slim-pixel-parity-20261001}"
    ;;
  *)
    echo "OSWORLD_SANDBOX_PROVIDER must be opensandbox or agentenv" >&2
    exit 2
    ;;
esac

required_vars=(
  CONTAINER
  NANO_OMNI_MODEL_NAME
  NANO_OMNI_CHAT_TEMPLATE
  SLURM_ACCOUNT
  SLURM_PARTITION
  OSWORLD_RESULTS_ROOT
)
if [[ "$OSWORLD_SANDBOX_PROVIDER" == "agentenv" ]]; then
  required_vars+=(AGENTENV_ENDPOINT AGENTENV_API_KEY AGENTENV_TLS_CA AGENTENV_TEMPLATE)
else
  required_vars+=(OPENSANDBOX_DOMAIN)
fi
if [[ "$OSWORLD_DRIVER_SCRIPT" == "eval_driver.sh" ]]; then
  case "${OSWORLD_EVAL_MODE:-}" in
    sft|checkpoint) ;;
    *)
      echo "OSWORLD_EVAL_MODE must be sft or checkpoint" >&2
      exit 2
      ;;
  esac
  required_vars+=(OSWORLD_EVAL_MODE OSWORLD_GRPO_VAL_DATA)
  if [[ "${OSWORLD_EVAL_MODE:-}" == "checkpoint" ]]; then
    required_vars+=(OSWORLD_EVAL_CHECKPOINT_DIR)
  fi
else
  required_vars+=(OSWORLD_GRPO_TRAIN_DATA)
fi
for name in "${required_vars[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Required variable is unset: $name" >&2
    exit 2
  fi
done

canonical_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Required file does not exist: $path" >&2
    exit 2
  fi
  readlink -f "$path"
}

canonical_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "Required directory does not exist: $path" >&2
    exit 2
  fi
  readlink -f "$path"
}

export CONTAINER
CONTAINER="$(canonical_file "$CONTAINER")"
export NANO_OMNI_MODEL_NAME
NANO_OMNI_MODEL_NAME="$(canonical_dir "$NANO_OMNI_MODEL_NAME")"
export NANO_OMNI_CHAT_TEMPLATE
NANO_OMNI_CHAT_TEMPLATE="$(canonical_file "$NANO_OMNI_CHAT_TEMPLATE")"
if [[ "$OSWORLD_SANDBOX_PROVIDER" == "agentenv" ]]; then
  export AGENTENV_TLS_CA
  AGENTENV_TLS_CA="$(canonical_file "$AGENTENV_TLS_CA")"
fi
if [[ "$OSWORLD_DRIVER_SCRIPT" != "eval_driver.sh" ]]; then
  export OSWORLD_GRPO_TRAIN_DATA
  OSWORLD_GRPO_TRAIN_DATA="$(canonical_file "$OSWORLD_GRPO_TRAIN_DATA")"
fi
if [[ -n "${OSWORLD_RLVR_SNAPSHOT:-}" ]]; then
  export OSWORLD_RLVR_SNAPSHOT
  OSWORLD_RLVR_SNAPSHOT="$(canonical_dir "$OSWORLD_RLVR_SNAPSHOT")"
  test -s "$OSWORLD_RLVR_SNAPSHOT/SNAPSHOT.json"
  test -d "$OSWORLD_RLVR_SNAPSHOT/tmp_funcs"
fi
if [[ "$OSWORLD_DRIVER_SCRIPT" == "eval_driver.sh" ]]; then
  export OSWORLD_GRPO_VAL_DATA
  OSWORLD_GRPO_VAL_DATA="$(canonical_file "$OSWORLD_GRPO_VAL_DATA")"
fi
if [[ "$OSWORLD_DRIVER_SCRIPT" == "eval_driver.sh" && "${OSWORLD_EVAL_MODE:-}" == "checkpoint" ]]; then
  export OSWORLD_EVAL_CHECKPOINT_DIR
  OSWORLD_EVAL_CHECKPOINT_DIR="$(canonical_dir "$OSWORLD_EVAL_CHECKPOINT_DIR")"
fi

test -r "$NANO_OMNI_MODEL_NAME/config.json"
if [[ "$OSWORLD_DRIVER_SCRIPT" != "eval_driver.sh" ]]; then
  test -s "$OSWORLD_GRPO_TRAIN_DATA"
fi
test -r "$RAY_SUB"

OSWORLD_CHAIN_LENGTH="${OSWORLD_CHAIN_LENGTH:-1}"
OSWORLD_GRPO_MAX_STEPS="${OSWORLD_GRPO_MAX_STEPS:-300}"
if ! [[ "$OSWORLD_CHAIN_LENGTH" =~ ^[1-9][0-9]*$ ]]; then
  echo "OSWORLD_CHAIN_LENGTH must be a positive integer" >&2
  exit 2
fi
if ! [[ "$OSWORLD_GRPO_MAX_STEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "OSWORLD_GRPO_MAX_STEPS must be a positive integer" >&2
  exit 2
fi
if ! [[ "$OSWORLD_NUM_NODES" =~ ^[1-9][0-9]*$ ]]; then
  echo "OSWORLD_NUM_NODES must be a positive integer" >&2
  exit 2
fi
export OSWORLD_CHAIN_LENGTH OSWORLD_GRPO_MAX_STEPS

OSWORLD_CHAIN_DEPENDENCY="${OSWORLD_CHAIN_DEPENDENCY:-afterok}"
case "$OSWORLD_CHAIN_DEPENDENCY" in
  afterok|afterany) ;;
  *)
    echo "OSWORLD_CHAIN_DEPENDENCY must be afterok or afterany" >&2
    exit 2
    ;;
esac

mkdir -p "$OSWORLD_RESULTS_ROOT"
OSWORLD_RESULTS_ROOT="$(readlink -f "$OSWORLD_RESULTS_ROOT")"
export OSWORLD_RESULTS_ROOT

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_name="${OSWORLD_RUN_NAME:-${OSWORLD_RUN_PREFIX:-osworld-v2-$OSWORLD_RECIPE}-$timestamp}"
if [[ "$run_name" == */* ]]; then
  echo "OSWORLD_RUN_NAME must not contain '/': $run_name" >&2
  exit 2
fi
series_root="${OSWORLD_SERIES_ROOT:-$OSWORLD_RESULTS_ROOT/$run_name}"
mkdir -p "$series_root/segments"
series_root="$(readlink -f "$series_root")"
export OSWORLD_SERIES_ROOT="$series_root"

checkpoint_dir="${OSWORLD_CHECKPOINT_DIR:-$series_root/checkpoints}"
cache_root="${OSWORLD_CACHE_ROOT:-$OSWORLD_RESULTS_ROOT/.cache}"
runtime_lib_dir="${OSWORLD_RUNTIME_LIB_DIR:-$cache_root/runtime-libs}"
gym_venv_dir="${OSWORLD_GYM_VENV_DIR:-$cache_root/gym-venvs}"
mkdir -p "$checkpoint_dir" "$cache_root" "$runtime_lib_dir" "$gym_venv_dir"
export OSWORLD_CHECKPOINT_DIR="$(readlink -f "$checkpoint_dir")"
export OSWORLD_CACHE_ROOT="$(readlink -f "$cache_root")"
export OSWORLD_RUNTIME_LIB_DIR="$(readlink -f "$runtime_lib_dir")"
export OSWORLD_GYM_VENV_DIR="$(readlink -f "$gym_venv_dir")"

copy_runtime_lib() {
  local library="$1"
  local destination="$OSWORLD_RUNTIME_LIB_DIR/$library"
  local source=""
  local ldconfig_bin=""
  if [[ -r "$destination" ]]; then
    return
  fi
  if command -v ldconfig >/dev/null 2>&1; then
    ldconfig_bin="$(command -v ldconfig)"
  elif [[ -x /sbin/ldconfig ]]; then
    ldconfig_bin=/sbin/ldconfig
  fi
  if [[ -n "$ldconfig_bin" ]]; then
    source="$("$ldconfig_bin" -p 2>/dev/null | awk -v lib="$library" '$1 == lib && $NF ~ /^\// && !found {print $NF; found=1}')"
  fi
  if [[ -z "$source" || ! -r "$source" ]]; then
    for candidate in \
      "/lib/x86_64-linux-gnu/$library" \
      "/usr/lib/x86_64-linux-gnu/$library"; do
      if [[ -r "$candidate" ]]; then
        source="$candidate"
        break
      fi
    done
  fi
  if [[ -z "$source" || ! -r "$source" ]]; then
    echo "Cannot locate $library; set OSWORLD_RUNTIME_LIB_DIR to a prepared directory" >&2
    exit 2
  fi
  cp -fL "$source" "$destination"
}

for library in libopenblas.so.0 libgfortran.so.5 libquadmath.so.0 libgcc_s.so.1; do
  copy_runtime_lib "$library"
done

EXPECTED_CONTAINER_SHA256="1b4edcdeac017210e6abe25885372e025ac111c45aeb038bda334659436eb74e"
if [[ "${OSWORLD_VERIFY_CONTAINER_SHA256:-1}" == "1" ]]; then
  actual_container_sha256="$(sha256sum "$CONTAINER" | awk '{print $1}')"
  if [[ "$actual_container_sha256" != "$EXPECTED_CONTAINER_SHA256" ]]; then
    echo "Qualified container checksum mismatch" >&2
    echo "expected: $EXPECTED_CONTAINER_SHA256" >&2
    echo "actual:   $actual_container_sha256" >&2
    exit 2
  fi
fi

for script in \
  "$RAY_SUB" \
  "$SCRIPT_DIR/runtime_env.sh" \
  "$SCRIPT_DIR/node_setup.sh" \
  "$SCRIPT_DIR/$OSWORLD_DRIVER_SCRIPT"; do
  bash -n "$script"
done

declare -A identity_mounts=()
mount_specs=("$RL_ROOT:/opt/nemo-rl")
add_identity_mount() {
  local path="$1"
  local mounted_path
  if [[ ! -d "$path" ]]; then
    path="$(dirname "$path")"
  fi
  path="$(readlink -f "$path")"
  if [[ "$path" == *[,:[:space:]]* ]]; then
    echo "Pyxis mount paths cannot contain commas, colons, or whitespace: $path" >&2
    exit 2
  fi
  for mounted_path in "${!identity_mounts[@]}"; do
    if [[ "$path" == "$mounted_path" || "$path" == "$mounted_path/"* ]]; then
      return
    fi
  done
  if [[ -z "${identity_mounts[$path]+set}" ]]; then
    identity_mounts["$path"]=1
    mount_specs+=("$path:$path")
  fi
}

add_identity_mount "$NANO_OMNI_MODEL_NAME"
add_identity_mount "$NANO_OMNI_CHAT_TEMPLATE"
if [[ "$OSWORLD_SANDBOX_PROVIDER" == "agentenv" ]]; then
  add_identity_mount "$AGENTENV_TLS_CA"
fi
if [[ "$OSWORLD_DRIVER_SCRIPT" != "eval_driver.sh" ]]; then
  add_identity_mount "$OSWORLD_GRPO_TRAIN_DATA"
fi
if [[ -n "${OSWORLD_RLVR_SNAPSHOT:-}" ]]; then
  add_identity_mount "$OSWORLD_RLVR_SNAPSHOT"
fi
if [[ "$OSWORLD_DRIVER_SCRIPT" == "eval_driver.sh" ]]; then
  add_identity_mount "$OSWORLD_GRPO_VAL_DATA"
fi
if [[ "$OSWORLD_DRIVER_SCRIPT" == "eval_driver.sh" && "${OSWORLD_EVAL_MODE:-}" == "checkpoint" ]]; then
  add_identity_mount "$OSWORLD_EVAL_CHECKPOINT_DIR"
fi
add_identity_mount "$OSWORLD_RESULTS_ROOT"
add_identity_mount "$OSWORLD_CHECKPOINT_DIR"
add_identity_mount "$OSWORLD_CACHE_ROOT"
add_identity_mount "$OSWORLD_RUNTIME_LIB_DIR"
add_identity_mount "$OSWORLD_GYM_VENV_DIR"
if [[ -n "${OSWORLD_HF_HOME:-}" ]]; then
  mkdir -p "$OSWORLD_HF_HOME"
  export OSWORLD_HF_HOME="$(readlink -f "$OSWORLD_HF_HOME")"
  add_identity_mount "$OSWORLD_HF_HOME"
fi

export MOUNTS
MOUNTS="$(IFS=,; echo "${mount_specs[*]}")"
export GPUS_PER_NODE=8
export NUM_NODES="$OSWORLD_NUM_NODES"
export OSWORLD_EXECUTION_ROOT=/opt/nemo-rl/examples/nemo_gym/osworld_v2
export OSWORLD_DRIVER_VENV="${OSWORLD_DRIVER_VENV:-/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker}"
export RAY_CLI="$OSWORLD_DRIVER_VENV/bin/ray"
export SETUP_COMMAND="bash '$OSWORLD_EXECUTION_ROOT/node_setup.sh'"
export COMMAND="bash '$OSWORLD_EXECUTION_ROOT/$OSWORLD_DRIVER_SCRIPT'"
export OSWORLD_POOL_REF="${OSWORLD_POOL_REF:-osworld-kvm}"
export OSWORLD_RECIPE
case "$OSWORLD_RECIPE" in
  flash-*) default_osworld_max_steps=200 ;;
  *) default_osworld_max_steps=150 ;;
esac
export OSWORLD_MAX_STEPS="${OSWORLD_MAX_STEPS:-$default_osworld_max_steps}"
export OSWORLD_MAX_PARALLEL_ROLLOUTS="${OSWORLD_MAX_PARALLEL_ROLLOUTS:-8}"
export OSWORLD_CHECKPOINT_MUST_SAVE_BY="${OSWORLD_CHECKPOINT_MUST_SAVE_BY:-00:03:40:00}"
export OSWORLD_SEED="${OSWORLD_SEED:-42}"

exclude_args=()
if [[ -n "${OSWORLD_EXCLUDE_NODES:-}" ]]; then
  exclude_args+=(--exclude="$OSWORLD_EXCLUDE_NODES")
fi
gres_args=()
if [[ -n "${OSWORLD_SBATCH_GRES-gpu:8}" ]]; then
  gres_args+=(--gres="${OSWORLD_SBATCH_GRES-gpu:8}")
fi
comment_args=()
if [[ -n "${OSWORLD_REAPER_COMMENT:-}" ]]; then
  comment_args+=(--comment="$OSWORLD_REAPER_COMMENT")
fi

if [[ "${OSWORLD_SUBMIT_DRY_RUN:-0}" == "1" ]]; then
  printf 'dry_run=ok\nseries_root=%s\ncheckpoint_dir=%s\nmounts=%s\n' \
    "$series_root" "$OSWORLD_CHECKPOINT_DIR" "$MOUNTS"
  exit 0
fi

next_segment=1
while [[ -e "$series_root/segments/segment-$(printf '%03d' "$next_segment")" ]]; do
  next_segment=$((next_segment + 1))
done

previous_job=""
submission_manifest="$series_root/submissions.tsv"
if [[ ! -e "$submission_manifest" ]]; then
  printf 'segment\tjob_id\tdependency\tlog_dir\n' > "$submission_manifest"
fi

for ((offset = 0; offset < OSWORLD_CHAIN_LENGTH; offset++)); do
  segment_number=$((next_segment + offset))
  segment_name="segment-$(printf '%03d' "$segment_number")"
  segment_root="$series_root/segments/$segment_name"
  mkdir -p "$segment_root"
  export OSWORLD_RUN_ROOT="$segment_root"
  export OSWORLD_RESULTS_DIR="$segment_root/exp_001"
  export BASE_LOG_DIR="$segment_root"

  dependency_args=()
  dependency_label=""
  if [[ -n "$previous_job" ]]; then
    dependency_label="$OSWORLD_CHAIN_DEPENDENCY:$previous_job"
    dependency_args+=(--dependency="$dependency_label")
  fi

  submission="$(
    sbatch --parsable \
      --export=ALL \
      --chdir="$RL_ROOT" \
      --nodes="$OSWORLD_NUM_NODES" \
      --exclusive \
      "${gres_args[@]}" \
      --account="$SLURM_ACCOUNT" \
      --partition="$SLURM_PARTITION" \
      --time="${OSWORLD_JOB_TIME:-04:00:00}" \
      --job-name="$OSWORLD_JOB_NAME" \
      --output="$segment_root/slurm-%j.out" \
      "${exclude_args[@]}" \
      "${comment_args[@]}" \
      "${dependency_args[@]}" \
      "$RAY_SUB"
  )"
  job_id="${submission%%;*}"
  printf '%s\t%s\t%s\t%s\n' \
    "$segment_name" "$job_id" "$dependency_label" "$segment_root" \
    | tee -a "$submission_manifest"
  previous_job="$job_id"
done

printf 'series_root=%s\ncheckpoint_dir=%s\nlast_job=%s\n' \
  "$series_root" "$OSWORLD_CHECKPOINT_DIR" "$previous_job"
