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

# Transformers derives trust_remote_code module names from local path basenames.
# A trailing slash gives an empty basename and can produce import cache collisions.
while [[ "${MODEL_PATH}" == */ && "${MODEL_PATH}" != "/" ]]; do
    MODEL_PATH="${MODEL_PATH%/}"
done

# ---- Optional vars with defaults ----
WANDB_PROJ="${WANDB_PROJ:-super-v3-posttraining}"
SLURM_TIME_LIMIT="${SLURM_TIME_LIMIT:-4:0:0}"
DRY_RUN="${DRY_RUN:-false}"
# Existing code snapshots are reused by tools/code_snapshot.sh. Refresh tracked
# files so restarts pick up dependency/code changes without deleting old logs.
REFRESH_CODE_SNAPSHOT="${REFRESH_CODE_SNAPSHOT:-true}"
# Override Slurm node count independently of cluster.num_nodes in the config.
# Use this when Gym judge servers need extra nodes beyond what NeMo-RL allocates
# (e.g. SBATCH_NUM_NODES=21 with cluster.num_nodes=16 reserves 5 nodes for Gym).
# Defaults to cluster.num_nodes from the config if unset.
SBATCH_NUM_NODES="${SBATCH_NUM_NODES:-}"
# HF->Megatron checkpoint conversion cache. Must be on a shared FS visible to all nodes.
# Defaults to $PERSISTENT_CACHE/megatron_ckpt_cache if not set.
NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR:-${PERSISTENT_CACHE}/megatron_ckpt_cache}"
# Shared lock dir for Megatron Bridge's HF config loader across multi-node runs.
MEGATRON_CONFIG_LOCK_DIR="${MEGATRON_CONFIG_LOCK_DIR:-${PERSISTENT_CACHE}/hf_config_locks}"
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
GYM_VENV_DIR="${GYM_VENV_DIR:-${PERSISTENT_CACHE}/gym_venvs}"
HF_MODULES_CACHE_DIR="${HF_MODULES_CACHE:-${PERSISTENT_CACHE}/hf_modules/${EXP_NAME}}"
# NEMO_RL_VENV_DIR="${NEMO_RL_VENV_DIR:-${PERSISTENT_CACHE}/ray_venvs}"
# RAY_CLI="${RAY_CLI:-${NEMO_RL_VENV_DIR}/nemo_rl.environments.nemo_gym.NemoGym/bin/ray}"
# UV_CACHE_DIR_OVERRIDE="${UV_CACHE_DIR_OVERRIDE:-${PERSISTENT_CACHE}/uv_cache}"
UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
NRL_FORCE_REBUILD_VENVS="${NRL_FORCE_REBUILD_VENVS:-false}"

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
echo " HF locks   : ${MEGATRON_CONFIG_LOCK_DIR}"
echo " HF modules : ${HF_MODULES_CACHE_DIR}"
echo " Gym venvs  : ${GYM_VENV_DIR}"
# echo " Ray venvs  : ${NEMO_RL_VENV_DIR}"
# echo " Ray CLI    : ${RAY_CLI}"
# echo " UV cache   : ${UV_CACHE_DIR_OVERRIDE}"
echo " Rebuild Ray venvs: ${NRL_FORCE_REBUILD_VENVS}"
echo " Partition  : ${SLURM_PARTITION}"
echo " Account    : ${SLURM_ACCOUNT}"
echo " Refresh snapshot: ${REFRESH_CODE_SNAPSHOT}"
echo "========================================"

# ---- Create cache dirs ----
mkdir -p "${VLLM_CACHE_DIR}" "${FLASHINFER_CUBIN_CACHE}" "${FLASHINFER_WS_BASE}" "${MEGATRON_CONFIG_LOCK_DIR}" "${HF_MODULES_CACHE_DIR}" # "${NEMO_RL_VENV_DIR}" "${UV_CACHE_DIR_OVERRIDE}" "${GYM_VENV_DIR}"


export OMP_NUM_THREADS=16
# export NEMO_RL_VENV_DIR
# export RAY_CLI
# export UV_CACHE_DIR_OVERRIDE

# ---- Code snapshot ----
SNAPSHOT_DIR=$(realpath "$(bash "${CODE_DIR}/tools/code_snapshot.sh" "${EXP_NAME}")")

if [[ "${REFRESH_CODE_SNAPSHOT}" == true ]]; then
    echo "Refreshing tracked files in code snapshot: ${SNAPSHOT_DIR}"
    (
        cd "${CODE_DIR}"
        rsync -a --files-from=<(git ls-files --recurse-submodules --cached --full-name) ./ "${SNAPSHOT_DIR}/"
    )
fi

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
export COMMAND="export HF_MODULES_CACHE=${HF_MODULES_CACHE_DIR} ; \
    python -c \"from transformers import AutoConfig, AutoTokenizer; p='${MODEL_PATH}'; AutoConfig.from_pretrained(p, trust_remote_code=True); AutoTokenizer.from_pretrained(p, trust_remote_code=True, use_fast=True); print('Prewarmed HF dynamic modules cache')\" ; \
    date ; \
    NRL_WG_USE_RAY_REF=1 \
    NRL_MEGATRON_CHECKPOINT_DIR=${NRL_MEGATRON_CHECKPOINT_DIR} \
    MEGATRON_CONFIG_LOCK_DIR=${MEGATRON_CONFIG_LOCK_DIR} \
    HF_MODULES_CACHE=${HF_MODULES_CACHE_DIR} \
    VLLM_CACHE_ROOT=${VLLM_CACHE_DIR} \
    DG_JIT_CACHE_DIR=${VLLM_CACHE_DIR}/deep_gemm \
    VLLM_DEEP_GEMM_WARMUP=skip \
    FLASHINFER_CUBIN_DIR=${FLASHINFER_CUBIN_CACHE} \
    FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WS_BASE} \
    NEMO_GYM_VENV_DIR=${GYM_VENV_DIR} \
    NRL_VLLM_USE_V1=1 \
    NRL_IGNORE_VERSION_MISMATCH=1 \
    VLLM_ATTENTION_BACKEND=FLASH_ATTN \
    NRL_FORCE_REBUILD_VENVS=${NRL_FORCE_REBUILD_VENVS} \
    RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
    UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT} \
    VLLM_USE_PRECOMPILED=1 \
    VLLM_PRECOMPILED_WHEEL_LOCATION=${VLLM_PRECOMPILED_WHEEL_LOCATION} \
    PYTHONPATH=${SNAPSHOT_DIR}:\${PYTHONPATH:-} \
    python ./examples/nemo_gym/run_grpo_nemo_gym.py \
    --config ${CONFIG_PATH} \
    env.nemo_gym.uv_venv_dir=${GYM_VENV_DIR} \
    env.nemo_gym.skip_venv_if_present=true \
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
    COMMAND="$COMMAND env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.container_formatter=${CONTAINER_FORMATTER}"
fi

export CONTAINER

# ---- Container mounts ----
BASE_MOUNTS="${SNAPSHOT_DIR}:${SNAPSHOT_DIR}"
BASE_MOUNTS+=",${CODE_DIR}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM:${SNAPSHOT_DIR}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM"

export MOUNTS="${EXTRA_MOUNTS:+${EXTRA_MOUNTS},}${BASE_MOUNTS}"

# ---- Read num_nodes from the config's cluster.num_nodes field ----
NUM_NODES=$(awk '/^cluster:/{found=1} found && /num_nodes:/{print $2; exit}' "${CONFIG_PATH}")
if [[ -z "$NUM_NODES" ]]; then
    echo "Error: could not read cluster.num_nodes from ${CONFIG_PATH}"
    exit 1
fi

# Use SBATCH_NUM_NODES if set, otherwise fall back to cluster.num_nodes
SBATCH_NUM_NODES="${SBATCH_NUM_NODES:-${NUM_NODES}}"
echo " Slurm nodes : ${SBATCH_NUM_NODES} (NeMo-RL cluster.num_nodes=${NUM_NODES})"

SBATCH_CMD=(
    sbatch
    --nodes="${SBATCH_NUM_NODES}"
    --account="${SLURM_ACCOUNT}"
    --job-name="${WANDB_NAME}"
    --partition="${SLURM_PARTITION}"
    --time="${SLURM_TIME_LIMIT}"
    --gres=gpu:8
    --exclusive
    --dependency=singleton
    --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"240","reason":"interactive","description":"interactive debugging"}}'
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
