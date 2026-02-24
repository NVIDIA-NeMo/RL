# Run from the root of NeMo-RL repo

NUM_NODES=${1:-21} # 21, 109
WANDB_PROJ=${2:-super-v3-posttraining}

EXP_SUFFIX="prod-stage1-${NUM_NODES}node"
WANDB_NAME="${EXP_SUFFIX}"

# ---------- Prebaked venvs (inside container, fast imports) ----------
PREBAKED_VENVS="/opt/prebaked_gym/venvs" #Yash

# ================================ Persistent Cache =================================
# ---------- persistent cache root on /scratch ----------
PERSISTENT_CACHE="/scratch/fsw/portfolios/coreai/users/yifuw/persistent_cache"
VLLM_CACHE_DIR="${PERSISTENT_CACHE}/vllm_compile_cache"
FLASHINFER_CUBIN_CACHE="${PERSISTENT_CACHE}/flashinfer_cubins"
FLASHINFER_WS_BASE="${PERSISTENT_CACHE}/flashinfer_workspace"
mkdir -p "${VLLM_CACHE_DIR}" "${FLASHINFER_CUBIN_CACHE}" "${FLASHINFER_WS_BASE}"

# GRPO
DATE=$(date +%Y%m%d)

CODE_DIR=$PWD

export OMP_NUM_THREADS=16

TRAIN_PATH=/scratch/fsw/portfolios/llmservice/users/jiaqiz/data/gym/rl-data-tools/blends/curriculum_v32_mindful-marmot.train.efforts0p3_qamath.safety_replaced.jsonl
VAL_PATH=/scratch/fsw/portfolios/llmservice/users/jiaqiz/data/gym/rl-data-tools/blends/curriculum_v32_mindful-marmot.train.jsonl

# Create code snapshot using the tool (only copies git-tracked files)
SNAPSHOT_DIR=$(bash ${CODE_DIR}/tools/code_snapshot.sh ${EXP_SUFFIX})

# Symlink 3rdparty/vllm submodule if not already present (submodule may not be initialized)
if [ -d "${CODE_DIR}/3rdparty/vllm" ] && [ ! -e "${SNAPSHOT_DIR}/3rdparty/vllm" ]; then
    echo "Symlinking 3rdparty/vllm to snapshot..."
    mkdir -p ${SNAPSHOT_DIR}/3rdparty
    ln -s ${CODE_DIR}/3rdparty/vllm ${SNAPSHOT_DIR}/3rdparty/vllm
fi

MODEL_PATH="/scratch/fsw/portfolios/llmservice/users/yianz/projects/nemotron3/super/copied_scripts/jiaqiz/results/bf16_baseline_quantum_apex/evals/step_150/hf"

echo "Submitting job with experiment suffix: ${EXP_SUFFIX}"

# Change to the snapshot directory
cd $SNAPSHOT_DIR

export VLLM_PRECOMPILED_WHEEL_LOCATION="https://github.com/vllm-project/vllm/releases/download/v0.13.0/vllm-0.13.0-cp38-abi3-manylinux_2_31_x86_64.whl"

export RAY_DEDUP_LOGS=1

# ================ Sandbox Configuration ================
export LISTEN_PORT=6000
export NGINX_PORT=6000
export NEMO_SKILLS_SANDBOX_PORT=6000
export SANDBOX_CONTAINER="/lustre/fsw/portfolios/llmservice/users/igitman/images/nemo-skills-sandbox-latest.sqsh"
# Flask application - port 6000 is hardcoded in main.py, but set env var for consistency
export SANDBOX_COMMAND="/start-with-nginx.sh"
export SANDBOX_ENV_VARS="NEMO_SKILLS_SANDBOX_PORT=${NEMO_SKILLS_SANDBOX_PORT}"

# ================ SWeRL Apptainer Setup ================ #
# Install apptainer/singularity for SWeRL evaluation (runs SWE-bench instances in containers)
read -r -d '' SETUP_COMMAND <<EOF
apt-get update && apt-get install -y git build-essential gcc wget
RET=1
while [ \$RET -ne 0 ]; do
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
cd ${SNAPSHOT_DIR}
EOF
export SETUP_COMMAND
export COMMAND="date ; \
    NRL_WG_USE_RAY_REF=1 \
    VLLM_CACHE_ROOT=${VLLM_CACHE_DIR} \
    DG_JIT_CACHE_DIR=${VLLM_CACHE_DIR}/deep_gemm \
    VLLM_DEEP_GEMM_WARMUP=skip \
    FLASHINFER_CUBIN_DIR=${FLASHINFER_CUBIN_CACHE} \
    FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WS_BASE} \
    NEMO_GYM_PREBAKED_VENVS_ROOT=${PREBAKED_VENVS} \
    NRL_VLLM_USE_V1=1 \
    NRL_IGNORE_VERSION_MISMATCH=1 \
    VLLM_ATTENTION_BACKEND=FLASH_ATTN \
    NRL_FORCE_REBUILD_VENVS=false \
    RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
    UV_HTTP_TIMEOUT=10 \
    VLLM_USE_PRECOMPILED=1 \
    VLLM_PRECOMPILED_WHEEL_LOCATION=${VLLM_PRECOMPILED_WHEEL_LOCATION} \
    uv run ./examples/nemo_gym/run_grpo_nemo_gym.py \
    --config examples/configs/grpo_superv3_stage1_${NUM_NODES}node.yaml \
    policy.model_name=${MODEL_PATH} \
    logger.wandb_enabled=True \
    logger.wandb.name=${WANDB_NAME} \
    logger.wandb.project=${WANDB_PROJ} \
    data.train.data_path=${TRAIN_PATH} \
    data.validation.data_path=${VAL_PATH}" \

export CONTAINER="/scratch/fsw/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/yifuw/containers/superv3-prebaked-828673-20260224-064957.sqsh"
export MOUNTS="/scratch:/scratch,/lustre:/lustre,${SNAPSHOT_DIR}:${SNAPSHOT_DIR},${SNAPSHOT_DIR}/3rdparty/Gym-workspace/Gym:/opt/nemo-rl/3rdparty/Gym-workspace/Gym,${CODE_DIR}/3rdparty/vllm:/opt/nemo-rl/3rdparty/vllm,${CODE_DIR}/3rdparty/Megatron-LM-workspace/Megatron-LM:/opt/nemo-rl/3rdparty/Megatron-LM-workspace/Megatron-LM"

sbatch \
    --nodes=${NUM_NODES} \
    --account=coreai_dlalgo_nemorl \
    --job-name=${WANDB_NAME} \
    --partition=batch \
    --time=4:0:0 \
    --gres=gpu:8 \
    --exclusive \
    --dependency=singleton \
    ray.sub
