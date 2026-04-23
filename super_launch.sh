#!/usr/bin/env bash
set -euo pipefail

# ---- Required vars ----
: "${EXP_NAME:?EXP_NAME is required}"
: "${TRAIN_PATH:?TRAIN_PATH is required}"
: "${VAL_PATH:?VAL_PATH is required}"
: "${CONFIG_PATH:?CONFIG_PATH is required}"
: "${MODEL_PATH:?MODEL_PATH is required}"
: "${CONTAINER:?CONTAINER is required}"
: "${SANDBOX_CONTAINER:?SANDBOX_CONTAINER is required}"
: "${PERSISTENT_CACHE:?PERSISTENT_CACHE is required}"
: "${SLURM_PARTITION:?SLURM_PARTITION is required}"
: "${SLURM_ACCOUNT:?SLURM_ACCOUNT is required}"

# ---- Optional vars with defaults ----
WANDB_PROJ="${WANDB_PROJ:-super-v3-posttraining}"
SLURM_TIME_LIMIT="${SLURM_TIME_LIMIT:-4:0:0}"
DRY_RUN="${DRY_RUN:-false}"
# Comma-separated host:container mount pairs for shared filesystems (e.g. "/scratch:/scratch,/lustre:/lustre").
EXTRA_MOUNTS="${EXTRA_MOUNTS:-}"
# Optional: path to directory containing .sif images for SWE-bench (Stage 2.2).
SIF_DIR="${SIF_DIR:-}"
CONTAINER_FORMATTER="${CONTAINER_FORMATTER:-}"

# ---- Derived paths ----
CODE_DIR=$(realpath "$PWD")
WANDB_NAME="${EXP_NAME}"
CHECKPOINT_DIR="results/${EXP_NAME}"
LOG_DIR="logs/${EXP_NAME}"

VLLM_CACHE_DIR="${PERSISTENT_CACHE}/vllm_compile_cache"
FLASHINFER_CUBIN_CACHE="${PERSISTENT_CACHE}/flashinfer_cubins"
FLASHINFER_WS_BASE="${PERSISTENT_CACHE}/flashinfer_workspace"
PREBAKED_VENVS="/opt/gym_venvs"

echo "========================================"
echo " Experiment : ${EXP_NAME}"
echo " Config     : ${CONFIG_PATH}"
echo " Model      : ${MODEL_PATH}"
echo " Train data : ${TRAIN_PATH}"
echo " Val data   : ${VAL_PATH}"
echo " Ckpt dir   : ${CHECKPOINT_DIR}"
echo " Log dir    : ${LOG_DIR}"
echo " Wandb      : ${WANDB_PROJ} / ${WANDB_NAME}"
echo " Container  : ${CONTAINER}"
echo " Sandbox    : ${SANDBOX_CONTAINER}"
echo " Cache root : ${PERSISTENT_CACHE}"
echo " Partition  : ${SLURM_PARTITION}"
echo " Account    : ${SLURM_ACCOUNT}"
echo "========================================"

# ---- Create cache dirs ----
mkdir -p "${VLLM_CACHE_DIR}" "${FLASHINFER_CUBIN_CACHE}" "${FLASHINFER_WS_BASE}"

export OMP_NUM_THREADS=16

# ---- Code snapshot ----
SNAPSHOT_DIR=$(realpath "$(bash "${CODE_DIR}/tools/code_snapshot.sh" "${EXP_NAME}")")

cd "${SNAPSHOT_DIR}"

export VLLM_PRECOMPILED_WHEEL_LOCATION="https://github.com/vllm-project/vllm/releases/download/v0.13.0/vllm-0.13.0-cp38-abi3-manylinux_2_31_x86_64.whl"
export RAY_DEDUP_LOGS=1

# ---- Sandbox configuration ----
export LISTEN_PORT=6000
export NGINX_PORT=6000
export NEMO_SKILLS_SANDBOX_PORT=6000
export SANDBOX_CONTAINER
export SANDBOX_COMMAND="/start-with-nginx.sh"
export SANDBOX_ENV_VARS="NEMO_SKILLS_SANDBOX_PORT=${NEMO_SKILLS_SANDBOX_PORT}"

# ---- SWeRL Apptainer setup ----
read -r -d '' SETUP_COMMAND <<'SETUP_EOF' || true
apt-get update && apt-get install -y git build-essential gcc wget
RET=1
while [ $RET -ne 0 ]; do
  cd /tmp && \
  wget --no-check-certificate https://github.com/apptainer/apptainer/releases/download/v1.3.1/apptainer_1.3.1_amd64.deb && \
  apt install -y ./apptainer_1.3.1_amd64.deb && \
  ln -sf /usr/bin/apptainer /usr/bin/singularity
  if command -v apptainer >/dev/null 2>&1; then
    echo "apptainer installed successfully."
    RET=0
  else
    echo "apptainer NOT installed. Retrying in 10 seconds..."
    sleep 10
    RET=1
  fi
done
SETUP_EOF
SETUP_COMMAND="${SETUP_COMMAND}
cd ${SNAPSHOT_DIR}"
export SETUP_COMMAND

# ---- Build the run command ----
export COMMAND="date ; \
    NRL_WG_USE_RAY_REF=1 \
    VLLM_CACHE_ROOT=${VLLM_CACHE_DIR} \
    DG_JIT_CACHE_DIR=${VLLM_CACHE_DIR}/deep_gemm \
    VLLM_DEEP_GEMM_WARMUP=skip \
    FLASHINFER_CUBIN_DIR=${FLASHINFER_CUBIN_CACHE} \
    FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WS_BASE} \
    NEMO_GYM_VENV_DIR=${PREBAKED_VENVS} \
    NRL_VLLM_USE_V1=1 \
    NRL_IGNORE_VERSION_MISMATCH=1 \
    VLLM_ATTENTION_BACKEND=FLASH_ATTN \
    NRL_FORCE_REBUILD_VENVS=false \
    RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
    UV_HTTP_TIMEOUT=10 \
    VLLM_USE_PRECOMPILED=1 \
    VLLM_PRECOMPILED_WHEEL_LOCATION=${VLLM_PRECOMPILED_WHEEL_LOCATION} \
    python ./examples/nemo_gym/run_grpo_nemo_gym.py \
    --config ${CONFIG_PATH} \
    policy.model_name=${MODEL_PATH} \
    checkpointing.checkpoint_dir=${CHECKPOINT_DIR} \
    logger.log_dir=${LOG_DIR} \
    logger.wandb_enabled=True \
    logger.wandb.name=${WANDB_NAME} \
    logger.wandb.project=${WANDB_PROJ} \
    data.train.data_path=${TRAIN_PATH} \
    data.validation.data_path=${VAL_PATH}"

if [[ -n "$SIF_DIR" ]]; then
    COMMAND="$COMMAND sif_dir=${SIF_DIR}"
fi

if [[ -n "$CONTAINER_FORMATTER" ]]; then
    COMMAND="$COMMAND container_formatter=${CONTAINER_FORMATTER}"
fi

export CONTAINER

# ---- Container mounts ----
BASE_MOUNTS="${SNAPSHOT_DIR}:${SNAPSHOT_DIR}"
BASE_MOUNTS+=",${CODE_DIR}/3rdparty/Megatron-LM-workspace/Megatron-LM:/opt/nemo-rl/3rdparty/Megatron-LM-workspace/Megatron-LM"

export MOUNTS="${EXTRA_MOUNTS:+${EXTRA_MOUNTS},}${BASE_MOUNTS}"

# ---- Read num_nodes from the config's cluster.num_nodes field ----
NUM_NODES=$(awk '/^cluster:/{found=1} found && /num_nodes:/{print $2; exit}' "${CONFIG_PATH}")
if [[ -z "$NUM_NODES" ]]; then
    echo "Error: could not read cluster.num_nodes from ${CONFIG_PATH}"
    exit 1
fi

SBATCH_CMD=(
    sbatch
    --nodes="${NUM_NODES}"
    --account="${SLURM_ACCOUNT}"
    --job-name="${WANDB_NAME}"
    --partition="${SLURM_PARTITION}"
    --time="${SLURM_TIME_LIMIT}"
    --gres=gpu:8
    --exclusive
    --dependency=singleton
    ray.sub
)

if [[ "$DRY_RUN" == true ]]; then
    echo ""
    echo "[dry-run] COMMAND:"
    echo "${COMMAND}"
    echo ""
    echo "[dry-run] sbatch invocation:"
    echo "${SBATCH_CMD[*]}"
else
    echo "Submitting job: ${WANDB_NAME}"
    "${SBATCH_CMD[@]}"
fi
