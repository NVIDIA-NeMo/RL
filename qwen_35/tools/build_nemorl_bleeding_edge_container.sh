#!/usr/bin/env bash
set -euo pipefail

DEFAULT_BASE_IMAGE="/lustre/fs1/portfolios/coreai/projects/coreai_mlperf_training/containers/nemorl_v0.6_prebaked_arm_clean_w_logging.sqsh"
DEFAULT_OUTPUT_IMAGE="/lustre/fs1/portfolios/coreai/projects/coreai_mlperf_training/containers/nemorl_v0.6_prebaked_arm_clean_w_logging_bleeding_edge.sqsh"

BASE_IMAGE="${BASE_IMAGE:-$DEFAULT_BASE_IMAGE}"
OUTPUT_IMAGE="${OUTPUT_IMAGE:-$DEFAULT_OUTPUT_IMAGE}"
PARTITION="${PARTITION:-cpu}"
ACCOUNT="${ACCOUNT:-coreai_mlperf_training}"
QOS="${QOS:-${SBATCH_QOS:-}}"
GRES="${GRES:-${SBATCH_GRES:-}}"
SEGMENT="${SEGMENT:-${SBATCH_SEGMENT:-}}"
TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
JOB_NAME="${JOB_NAME:-nemorl-bleeding-edge}"
WORKDIR="${WORKDIR:-$HOME/nemorl_bleeding_edge_build}"
NEMO_RL_REF="${NEMO_RL_REF:-main}"
MEGATRON_BRIDGE_REF="${MEGATRON_BRIDGE_REF:-main}"
MEGATRON_LM_REF="${MEGATRON_LM_REF:-main}"
CONTAINER_ROOT="${CONTAINER_ROOT:-/opt/nemo-rl}"
PYTHON_BIN="${PYTHON_BIN:-/opt/nemo_rl_venv/bin/python}"
EXTRA_MOUNTS="${EXTRA_MOUNTS:-}"
SUBMIT=0
RUN_INSIDE=0

usage() {
  cat <<'EOF'
Build a NeMo RL bleeding-edge sqsh from an existing sqsh via Slurm/Pyxis.

Default action is a dry-run summary. Use --submit on the login node.
The Slurm job runs inside --container-image and writes a new image using
--container-save.

Options:
  --submit                    Submit the Slurm build job.
  --run-inside                Internal mode used inside the container.
  --base-image PATH           Source sqsh image.
  --output-image PATH         Destination sqsh image.
  --partition NAME            Slurm partition, default: cpu.
  --account NAME              Slurm account. Empty string omits #SBATCH -A.
  --qos NAME                  Slurm QOS. Empty string omits #SBATCH --qos.
  --gres SPEC                 Slurm GRES. Empty string omits #SBATCH --gres.
  --segment N                 Slurm segment size. Empty string omits #SBATCH --segment.
  --time HH:MM:SS             Slurm time limit, default: 01:00:00.
  --workdir PATH              Host work/log directory.
  --job-name NAME             Slurm job name.
  --nemo-rl-ref REF           NeMo RL git ref, default: main.
  --megatron-bridge-ref REF   Megatron-Bridge git ref, default: main.
  --megatron-lm-ref REF       Megatron-LM git ref, default: main.
  --extra-mounts SPEC         Extra Pyxis mounts, appended to work/script mount.

Environment variables with the same uppercase names are also honored.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --submit) SUBMIT=1; shift ;;
    --run-inside) RUN_INSIDE=1; shift ;;
    --base-image) BASE_IMAGE="$2"; shift 2 ;;
    --output-image) OUTPUT_IMAGE="$2"; shift 2 ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --account) ACCOUNT="$2"; shift 2 ;;
    --qos) QOS="$2"; shift 2 ;;
    --gres) GRES="$2"; shift 2 ;;
    --segment) SEGMENT="$2"; shift 2 ;;
    --time) TIME_LIMIT="$2"; shift 2 ;;
    --workdir) WORKDIR="$2"; shift 2 ;;
    --job-name) JOB_NAME="$2"; shift 2 ;;
    --nemo-rl-ref) NEMO_RL_REF="$2"; shift 2 ;;
    --megatron-bridge-ref) MEGATRON_BRIDGE_REF="$2"; shift 2 ;;
    --megatron-lm-ref) MEGATRON_LM_REF="$2"; shift 2 ;;
    --extra-mounts) EXTRA_MOUNTS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*"
}

quote_sbatch() {
  printf '%q' "$1"
}

is_git_repo() {
  git -C "$1" rev-parse --is-inside-work-tree >/dev/null 2>&1
}

pip_install() {
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$PYTHON_BIN" "$@"
  else
    "$PYTHON_BIN" -m pip install "$@"
  fi
}

pip_install_for_python() {
  local python_bin="$1"
  shift
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$python_bin" "$@"
  else
    "$python_bin" -m pip install "$@"
  fi
}

project_venv_root() {
  dirname "$(dirname "$PYTHON_BIN")"
}

uv_sync_project() {
  local venv_root
  venv_root="$(project_venv_root)"
  log "Syncing full NeMo RL project dependencies into ${venv_root}"
  (
    cd "$CONTAINER_ROOT"
    export VIRTUAL_ENV="$venv_root"
    export UV_PROJECT_ENVIRONMENT="$venv_root"
    export PATH="${venv_root}/bin:${PATH}"
    uv sync --active
  )
}

repair_ray_worker_venvs() {
  local ray_version
  ray_version="$("$PYTHON_BIN" - <<'PY'
import ray
print(ray.__version__)
PY
)"

  local ray_venv_root="${RAY_VENV_ROOT:-/opt/ray_venvs}"
  if [[ ! -d "$ray_venv_root" ]]; then
    log "No Ray worker venv root found at ${ray_venv_root}; skipping worker venv repair"
    return
  fi

  log "Repairing Ray worker venvs under ${ray_venv_root} to ray[default]==${ray_version}"
  while IFS= read -r -d '' python_bin; do
    log "Repairing Ray in ${python_bin}"
    pip_install_for_python "$python_bin" --reinstall "ray[default]==${ray_version}" simplejson
    pip_install_for_python "$python_bin" --no-deps tensordict
    pip_install_for_python "$python_bin" --no-deps \
      "nvidia-resiliency-ext @ git+https://github.com/NVIDIA/nvidia-resiliency-ext.git@v0.6.0"
    "$python_bin" - <<'PY'
import ray
import importlib.metadata
print(f"worker ray={ray.__version__} from {getattr(ray, '__file__', '<unknown>')}")
print(f"worker tensordict={importlib.metadata.version('tensordict')}")
print(f"worker nvidia-resiliency-ext={importlib.metadata.version('nvidia-resiliency-ext')}")
PY
  done < <(find "$ray_venv_root" -mindepth 2 -maxdepth 3 -path '*/bin/python' \( -type f -o -type l \) -print0)
}

repair_build_backend_metadata() {
  log "Repairing build backend metadata if the base venv has stale setuptools metadata"
  mapfile -t site_dirs < <("$PYTHON_BIN" - <<'PY'
import site
import sysconfig

paths = set()
for getter in (site.getsitepackages,):
    try:
        paths.update(getter())
    except Exception:
        pass
for key in ("purelib", "platlib"):
    path = sysconfig.get_path(key)
    if path:
        paths.add(path)
for path in sorted(paths):
    print(path)
PY
)

  for site_dir in "${site_dirs[@]}"; do
    [[ -d "$site_dir" ]] || continue
    rm -rf \
      "$site_dir"/setuptools \
      "$site_dir"/setuptools-*.dist-info \
      "$site_dir"/pkg_resources \
      "$site_dir"/wheel \
      "$site_dir"/wheel-*.dist-info \
      "$site_dir"/packaging \
      "$site_dir"/packaging-*.dist-info \
      "$site_dir"/pybind11 \
      "$site_dir"/pybind11-*.dist-info
  done
}

run_inside_container() {
  log "Running inside container"
  log "Target NeMo RL ref: ${NEMO_RL_REF}"
  log "Target Megatron-Bridge ref: ${MEGATRON_BRIDGE_REF}"
  log "Target Megatron-LM ref: ${MEGATRON_LM_REF}"

  if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="$(command -v python3 || command -v python)"
  fi
  if [[ -z "$PYTHON_BIN" ]]; then
    echo "Could not find python inside the container" >&2
    exit 1
  fi

  mkdir -p "$WORKDIR"
  {
    echo "date=$(date -Is)"
    echo "base_image=$BASE_IMAGE"
    echo "output_image=$OUTPUT_IMAGE"
    echo "nemo_rl_ref=$NEMO_RL_REF"
    echo "megatron_bridge_ref=$MEGATRON_BRIDGE_REF"
    echo "megatron_lm_ref=$MEGATRON_LM_REF"
    echo "python=$PYTHON_BIN"
  } > "$WORKDIR/build_metadata.env"

  git config --global --add safe.directory '*' || true

  cd /
  if [[ -e "$CONTAINER_ROOT" ]]; then
    rm -rf "$CONTAINER_ROOT"
  fi

  log "Cloning NeMo RL into $CONTAINER_ROOT"
  git clone --depth 1 --branch "$NEMO_RL_REF" https://github.com/NVIDIA-NeMo/RL.git "$CONTAINER_ROOT"

  log "Initializing NeMo RL submodules"
  git -C "$CONTAINER_ROOT" submodule update --init --depth 1 \
    3rdparty/Automodel-workspace/Automodel \
    3rdparty/Megatron-Bridge-workspace/Megatron-Bridge \
    3rdparty/Gym-workspace/Gym

  bridge_repo="$CONTAINER_ROOT/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"
  if is_git_repo "$bridge_repo"; then
    log "Checking out Megatron-Bridge ref $MEGATRON_BRIDGE_REF"
    git -C "$bridge_repo" fetch --depth 1 origin "$MEGATRON_BRIDGE_REF"
    git -C "$bridge_repo" checkout --detach FETCH_HEAD
    git -C "$bridge_repo" submodule update --init --depth 1 3rdparty/Megatron-LM
  fi

  mlm_repo="$bridge_repo/3rdparty/Megatron-LM"
  if is_git_repo "$mlm_repo"; then
    log "Checking out Megatron-LM ref $MEGATRON_LM_REF"
    git -C "$mlm_repo" fetch --depth 1 origin "$MEGATRON_LM_REF"
    git -C "$mlm_repo" checkout --detach FETCH_HEAD
  fi

  log "Installing editable packages without dependency resolution"
  repair_build_backend_metadata
  pip_install setuptools wheel packaging pybind11
  uv_sync_project
  repair_ray_worker_venvs
  pip_install --no-deps --no-build-isolation -e "$CONTAINER_ROOT"

  bridge_ws="$CONTAINER_ROOT/3rdparty/Megatron-Bridge-workspace"
  if [[ -f "$bridge_ws/pyproject.toml" || -f "$bridge_ws/setup.py" ]]; then
    pip_install --no-deps --no-build-isolation -e "$bridge_ws"
  elif [[ -f "$bridge_repo/pyproject.toml" || -f "$bridge_repo/setup.py" ]]; then
    pip_install --no-deps --no-build-isolation -e "$bridge_repo"
  else
    echo "Could not find Megatron-Bridge package root under $bridge_ws" >&2
    exit 1
  fi

  if [[ -f "$mlm_repo/pyproject.toml" || -f "$mlm_repo/setup.py" ]]; then
    pip_install --no-deps --no-build-isolation -e "$mlm_repo"
  else
    echo "Could not find Megatron-LM package root under $mlm_repo" >&2
    exit 1
  fi

  log "Recording installed revisions and import smoke"
  {
    echo "=== git revisions ==="
    git -C "$CONTAINER_ROOT" rev-parse HEAD
    git -C "$bridge_repo" rev-parse HEAD
    git -C "$mlm_repo" rev-parse HEAD
    echo "=== imports ==="
    "$PYTHON_BIN" - <<'PY'
import importlib
import importlib.util

nemo_rl = importlib.import_module("nemo_rl")
print(f"nemo_rl: {getattr(nemo_rl, '__file__', '<no file>')}")
for name in ("megatron", "megatron.bridge"):
    spec = importlib.util.find_spec(name)
    print(f"{name}: {spec.origin if spec else '<not found>'}")
try:
    import torch
    print(f"torch: {getattr(torch, '__file__', None)}, Tensor={hasattr(torch, 'Tensor')}")
except Exception as exc:
    print(f"torch import failed during build smoke: {exc!r}")
PY
  } | tee "$WORKDIR/import_smoke.log"

  log "Container mutation complete; Slurm will save the image on exit"
}

submit_job() {
  mkdir -p "$WORKDIR"
  script_path="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
  script_dir="$(dirname "$script_path")"
  mounts="${script_dir}:${script_dir},${WORKDIR}:${WORKDIR}"
  if [[ -n "$EXTRA_MOUNTS" ]]; then
    mounts="${mounts},${EXTRA_MOUNTS}"
  fi

  sbatch_file="$WORKDIR/${JOB_NAME}.sbatch"
  {
    echo "#!/usr/bin/env bash"
    echo "#SBATCH -J ${JOB_NAME}"
    echo "#SBATCH -p ${PARTITION}"
    if [[ -n "$ACCOUNT" ]]; then
      echo "#SBATCH -A ${ACCOUNT}"
    fi
    if [[ -n "$QOS" ]]; then
      echo "#SBATCH --qos=${QOS}"
    fi
    if [[ -n "$GRES" ]]; then
      echo "#SBATCH --gres=${GRES}"
    fi
    if [[ -n "$SEGMENT" ]]; then
      echo "#SBATCH --segment=${SEGMENT}"
    fi
    echo "#SBATCH -N 1"
    echo "#SBATCH -n 1"
    echo "#SBATCH -t ${TIME_LIMIT}"
    echo "#SBATCH -o ${WORKDIR}/%x-%j.out"
    echo "#SBATCH -e ${WORKDIR}/%x-%j.err"
    echo "#SBATCH --container-image=${BASE_IMAGE}"
    echo "#SBATCH --container-save=${OUTPUT_IMAGE}"
    echo "#SBATCH --container-mounts=${mounts}"
    echo "set -euo pipefail"
    echo "bash $(quote_sbatch "$script_path") --run-inside --base-image $(quote_sbatch "$BASE_IMAGE") --output-image $(quote_sbatch "$OUTPUT_IMAGE") --workdir $(quote_sbatch "$WORKDIR") --nemo-rl-ref $(quote_sbatch "$NEMO_RL_REF") --megatron-bridge-ref $(quote_sbatch "$MEGATRON_BRIDGE_REF") --megatron-lm-ref $(quote_sbatch "$MEGATRON_LM_REF")"
  } > "$sbatch_file"

  log "Submitting $sbatch_file"
  job_id="$(sbatch --parsable "$sbatch_file")"
  log "Submitted job $job_id"
  log "stdout: $WORKDIR/${JOB_NAME}-${job_id}.out"
  log "stderr: $WORKDIR/${JOB_NAME}-${job_id}.err"
  log "output image: $OUTPUT_IMAGE"
}

if [[ "$RUN_INSIDE" -eq 1 ]]; then
  run_inside_container
elif [[ "$SUBMIT" -eq 1 ]]; then
  submit_job
else
  usage
  echo
  echo "Current defaults:"
  echo "  base image:    $BASE_IMAGE"
  echo "  output image:  $OUTPUT_IMAGE"
  echo "  partition:     $PARTITION"
  echo "  account:       ${ACCOUNT:-<none>}"
  echo "  qos:           ${QOS:-<none>}"
  echo "  gres:          ${GRES:-<none>}"
  echo "  segment:       ${SEGMENT:-<none>}"
  echo "  time:          $TIME_LIMIT"
  echo "  workdir:       $WORKDIR"
fi
