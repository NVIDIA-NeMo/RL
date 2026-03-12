#!/bin/bash
# ============================================================================
# Dummy Turing VIF GRPO Training Launch Script (1 node, small model)
# ============================================================================
# Lightweight version of the multichallenge experiment for fast iteration.
# Uses Qwen2.5-1.5B for both policy and judge on a single node.
#
# USAGE:
#   bash experiments/dummy_turing_vif/launch_dummy_turing_vif_training.sh
#
# ============================================================================

set -euo pipefail

# ============================================================================
# CLUSTER DETECTION
# ============================================================================
HOSTNAME_STR=$(hostname)
if [[ "$HOSTNAME_STR" == *"lax"* ]]; then
    CLUSTER="lax"
elif [[ "$HOSTNAME_STR" == *"dfw"* ]]; then
    CLUSTER="dfw"
else
    CLUSTER="unknown"
fi
echo "Detected cluster: ${CLUSTER} (hostname: ${HOSTNAME_STR})"

# ============================================================================
# CONFIGURATION
# ============================================================================

EXP_NAME="${EXP_NAME:-dummy_turing_vif}"
NUM_NODES="${NUM_NODES:-1}"
JUDGE_NUM_NODES="${JUDGE_NUM_NODES:-1}"

SLURM_ACCOUNT="${SLURM_ACCOUNT:-${ACCOUNT:-llmservice_modelalignment_ppo}}"
SLURM_PARTITION="interactive"
SLURM_TIME="${SLURM_TIME:-1:00:00}"

# Container image
CONTAINER_IMAGE_LAX="/scratch/fsw/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/yifuw/enroot-images/gitlab-master.nvidia.com/yifuw/images/nemo-rl:vllm-0_13_0-geshen__super_mtp-20260106.squashfs"
CONTAINER_IMAGE_DFW="/lustre/fsw/portfolios/llmservice/users/mfathi/containers/nemo-rl:vllm-0_13_0-geshen__super_mtp-20260106.squashfs"

if [ "$CLUSTER" = "lax" ]; then
    CONTAINER_IMAGE="${CONTAINER_IMAGE:-$CONTAINER_IMAGE_LAX}"
else
    CONTAINER_IMAGE="${CONTAINER_IMAGE:-$CONTAINER_IMAGE_DFW}"
fi

REPO_LOCATION="${REPO_LOCATION:-$(readlink -f $(pwd))}"
GPFS="${GPFS:-$REPO_LOCATION}"

CONFIG_FILE="${CONFIG_FILE:-experiments/dummy_turing_vif/grpo_dummy_turing_vif.yaml}"

# Small model for fast iteration
POLICY_MODEL="${POLICY_MODEL:-Qwen/Qwen2.5-1.5B}"

# Data paths
TRAIN_DATA="${TRAIN_DATA:-3rdparty/Gym-workspace/Gym/resources_servers/turing_vif/data/multichallenge_vanilla.jsonl}"
VAL_DATA="${VAL_DATA:-3rdparty/Gym-workspace/Gym/resources_servers/turing_vif/data/multichallenge_vanilla.jsonl}"

# ============================================================================
# JUDGE CONFIGURATION — Small judge on the same node
# ============================================================================
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen2.5-1.5B}"
JUDGE_TP="${JUDGE_TP:-1}"
JUDGE_ROUTER_DP_SIZE="${JUDGE_ROUTER_DP_SIZE:-1}"
JUDGE_GPU_UTIL="${JUDGE_GPU_UTIL:-0.85}"
JUDGE_MAX_LEN="${JUDGE_MAX_LEN:-32768}"

# Hugging Face configuration
HF_HOME_PATH="${HF_HOME:-${HSH:-$HOME}/.cache/huggingface}"
HF_TOKEN_VALUE="${HF_TOKEN:-}"

# Weights & Biases configuration
WANDB_API_KEY_VALUE="${WANDB_API_KEY:-${WANDB_TOKEN:-}}"
WANDB_PROJECT="${WANDB_PROJECT:-nemo-rl-dummy-turing-vif}"

# VLLM configuration
VLLM_PRECOMPILED_WHEEL_LOCATION="${VLLM_PRECOMPILED_WHEEL_LOCATION:-https://github.com/vllm-project/vllm/releases/download/v0.13.0/vllm-0.13.0-cp38-abi3-manylinux_2_31_x86_64.whl}"
export VLLM_PRECOMPILED_WHEEL_LOCATION

NUM_JOBS="${NUM_JOBS:-1}"

# ============================================================================
# DERIVED VARIABLES
# ============================================================================

cd "$REPO_LOCATION"

# Submodule check
if [ ! -f "${REPO_LOCATION}/3rdparty/Gym-workspace/Gym/.git" ] && [ ! -d "${REPO_LOCATION}/3rdparty/Gym-workspace/Gym/.git" ]; then
    echo "Initializing Gym submodule..."
    git submodule update --init --recursive -- 3rdparty/Gym-workspace/Gym
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to initialize Gym submodule."
        exit 1
    fi
fi

EXEC_DIR="$REPO_LOCATION"

LOG_BASE_DIR="${LOG_BASE_DIR:-logs/${EXP_NAME}}"
mkdir -p "$LOG_BASE_DIR"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-${LOG_BASE_DIR}/checkpoints}"

# Total nodes = policy + judge
TOTAL_NODES=$((NUM_NODES + JUDGE_NUM_NODES))

# ============================================================================
# BUILD THE COMMAND
# ============================================================================

# Judge arguments (local_vllm_model spins up its own vLLM instance)
JUDGE_ARGS="++env.nemo_gym.num_gpu_nodes=${JUDGE_NUM_NODES} \\
    ++env.nemo_gym.turing_vif.resources_servers.turing_vif.judge_server_name=judge_model \\
    ++env.nemo_gym.turing_vif.resources_servers.turing_vif.judge_api_key=dummy_key \\
    ++env.nemo_gym.turing_vif.resources_servers.turing_vif.judge_model=${JUDGE_MODEL} \\
    ++env.nemo_gym.judge_model.responses_api_models.local_vllm_model.entrypoint=app.py \\
    ++env.nemo_gym.judge_model.responses_api_models.local_vllm_model.model=${JUDGE_MODEL} \\
    ++env.nemo_gym.judge_model.responses_api_models.local_vllm_model.return_token_id_information=false \\
    ++env.nemo_gym.judge_model.responses_api_models.local_vllm_model.uses_reasoning_parser=false \\
    ++env.nemo_gym.judge_model.responses_api_models.local_vllm_model.vllm_serve_kwargs.tensor_parallel_size=${JUDGE_TP} \\
    ++env.nemo_gym.judge_model.responses_api_models.local_vllm_model.vllm_serve_kwargs.pipeline_parallel_size=1 \\
    ++env.nemo_gym.judge_model.responses_api_models.local_vllm_model.vllm_serve_kwargs.data_parallel_size=${JUDGE_ROUTER_DP_SIZE} \\
    ++env.nemo_gym.judge_model.responses_api_models.local_vllm_model.vllm_serve_kwargs.gpu_memory_utilization=${JUDGE_GPU_UTIL} \\
    ++env.nemo_gym.judge_model.responses_api_models.local_vllm_model.vllm_serve_kwargs.max_model_len=${JUDGE_MAX_LEN} \\
    ++env.nemo_gym.judge_model.responses_api_models.local_vllm_model.vllm_serve_env_vars.VLLM_RAY_DP_PACK_STRATEGY=strict"

read -r -d '' COMMAND <<EOF || true
cd ${EXEC_DIR}

# Set environment variables
export HF_HOME=${HF_HOME_PATH}
export HF_TOKEN=${HF_TOKEN_VALUE}
export WANDB_API_KEY=${WANDB_API_KEY_VALUE}
export OMP_NUM_THREADS=16
export RAY_DEDUP_LOGS=1

# Set NeMo-RL and vLLM environment variables
export NRL_VLLM_USE_V1=1
export NRL_IGNORE_VERSION_MISMATCH=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export NRL_FORCE_REBUILD_VENVS=true
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export UV_HTTP_TIMEOUT=300
export VLLM_USE_PRECOMPILED=1
export VLLM_PRECOMPILED_WHEEL_LOCATION=${VLLM_PRECOMPILED_WHEEL_LOCATION}

# Run GRPO training with NeMo-Gym (Dummy Turing VIF)
uv run python examples/nemo_gym/run_grpo_nemo_gym.py \\
    --config ${CONFIG_FILE} \\
    ++cluster.num_nodes=${NUM_NODES} \\
    ++policy.model_name=${POLICY_MODEL} \\
    ++logger.wandb.name=${EXP_NAME} \\
    ++logger.wandb.project=${WANDB_PROJECT} \\
    ++logger.log_dir=${LOG_BASE_DIR}/\${SLURM_JOB_ID}-logs \\
    ++checkpointing.checkpoint_dir=${CHECKPOINT_DIR} \\
    ++loss_fn.force_on_policy_ratio=true \\
    ${JUDGE_ARGS} \\
    \$@
EOF

# ============================================================================
# LAUNCH THE JOB
# ============================================================================

echo "============================================================================"
echo "Launching Dummy Turing VIF GRPO Training (1-node)"
echo "============================================================================"
echo "Experiment Name: ${EXP_NAME}"
echo "Policy Model:    ${POLICY_MODEL}"
echo "Judge Model:     ${JUDGE_MODEL}"
echo "Total Nodes:     ${TOTAL_NODES}"
echo "Container:       ${CONTAINER_IMAGE}"
echo "SLURM Account:   ${SLURM_ACCOUNT}"
echo "SLURM Partition: ${SLURM_PARTITION}"
echo "Time Limit:      ${SLURM_TIME}"
echo "Log Directory:   ${LOG_BASE_DIR}/<job_id>-logs/"
echo "============================================================================"
echo ""
echo "Training Command:"
echo "$COMMAND"
echo ""
echo "============================================================================"

# Mount paths
MOUNT_STRING="${EXEC_DIR}:${EXEC_DIR},${EXEC_DIR}/3rdparty/Gym-workspace/Gym:/opt/nemo-rl/3rdparty/Gym-workspace/Gym,${GPFS}/3rdparty/Megatron-LM-workspace/Megatron-LM:/opt/nemo-rl/3rdparty/Megatron-LM-workspace/Megatron-LM"

if [ "$CLUSTER" = "lax" ]; then
    MOUNT_STRING="/scratch:/scratch,/lustre:/lustre,${MOUNT_STRING}"
else
    MOUNT_STRING="/lustre:/lustre,${MOUNT_STRING}"
fi

if [ -d "${GPFS}/3rdparty/vllm" ]; then
    MOUNT_STRING="${MOUNT_STRING},${GPFS}/3rdparty/vllm:/opt/nemo-rl/3rdparty/vllm"
fi

# Submit job(s)
PREV_JOB_ID=""
SUBMITTED_JOBS=()

for JOB_NUM in $(seq 1 "${NUM_JOBS}"); do
    if [ -z "$PREV_JOB_ID" ]; then
        DEP_FLAG="--dependency=singleton"
    else
        DEP_FLAG="--dependency=afterok:${PREV_JOB_ID}"
    fi

    JOB_ID=$(COMMAND="$COMMAND" \
    CONTAINER="$CONTAINER_IMAGE" \
    MOUNTS="${MOUNT_STRING}" \
    CPUS_PER_WORKER="${CPUS_PER_WORKER:-112}" \
    BASE_LOG_DIR="$LOG_BASE_DIR" \
    sbatch --parsable \
        --nodes="${TOTAL_NODES}" \
        --account="${SLURM_ACCOUNT}" \
        --partition="${SLURM_PARTITION}" \
        --time="${SLURM_TIME}" \
        --job-name="${EXP_NAME}" \
        --gres=gpu:8 \
        --exclusive \
        ${DEP_FLAG} \
        ray.sub)

    PREV_JOB_ID="$JOB_ID"
    SUBMITTED_JOBS+=("$JOB_ID")
    echo "  Submitted job ${JOB_NUM}/${NUM_JOBS}: ${JOB_ID} (${DEP_FLAG})"
done

echo ""
echo "All ${NUM_JOBS} job(s) submitted!"
echo "   Job IDs:     ${SUBMITTED_JOBS[*]}"
echo "   Logs:        ${LOG_BASE_DIR}/<job_id>-logs/"
echo "   Checkpoints: ${CHECKPOINT_DIR}/"
