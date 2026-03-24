#!/bin/bash
# ============================================================================
# Turing VIF Inverse IF GRPO Training Launch Script
# ============================================================================
# This script launches a GRPO training job on a SLURM cluster using the
# Turing VIF environment (Inverse IF data) from NeMo-Gym.
#
# Turing VIF is a single-turn benchmark where each task carries its own
# per-task judge prompt template and system prompt. The judge evaluates
# one criterion at a time and returns {"result": "PASS"/"FAIL"}.
#
# USAGE:
#   # Edit the configuration variables below, then run:
#   bash experiments/turing_vif_inverse_if/launch_turing_vif_inverse_if_training.sh
#
#   # Or with command-line overrides:
#   EXP_NAME=my_exp NUM_NODES=2 bash experiments/turing_vif_inverse_if/launch_turing_vif_inverse_if_training.sh
#
#   # Launch a chain of 3 dependent jobs (each resumes from the previous checkpoint):
#   NUM_JOBS=3 bash experiments/turing_vif_inverse_if/launch_turing_vif_inverse_if_training.sh
#
# JUDGE CONFIGURATION:
#   By default, the LLM judge uses a dedicated vLLM server.
#   Configure via env.yaml or command-line overrides:
#
#   # Option 1: Use same endpoint as policy (from env.yaml)
#   # No extra configuration needed - uses policy_base_url/api_key/model_name
#
#   # Option 2: Use a different judge endpoint
#   bash launch_turing_vif_inverse_if_training.sh \
#     ++env.nemo_gym.judge_model.responses_api_models.openai_model.openai_base_url="https://judge-endpoint.com/v1" \
#     ++env.nemo_gym.judge_model.responses_api_models.openai_model.openai_api_key="your-key" \
#     ++env.nemo_gym.judge_model.responses_api_models.openai_model.openai_model="judge-model-name"
#
# ============================================================================

set -euo pipefail

# ============================================================================
# CLUSTER DETECTION
# ============================================================================
# Detect which cluster we're on based on hostname
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
# CONFIGURATION - Edit these variables as needed
# ============================================================================

# Experiment name (used for job name, logging, and checkpoints)
EXP_NAME="${EXP_NAME:-turing_vif_inverse_if_grpo}"

# Number of nodes for policy (colocated: training + inference on same GPUs)
# Matching reference run_grpo_math_nano_next_megatron_passat1.sh:
#   - 16 nodes for Megatron training + vLLM generation (colocated)
#   - TP=4, CP=1, EP=8, DP=32
#   - vLLM TP=2, gpu_memory_utilization=0.5
# Note: Colocated mode is more memory efficient with high DP
NUM_NODES="${NUM_NODES:-16}"

# SLURM configuration (defaults from ~/.bashrc)
SLURM_ACCOUNT="${SLURM_ACCOUNT:-${ACCOUNT:-llmservice_modelalignment_ppo}}"
SLURM_PARTITION="${SLURM_PARTITION:-${PARTITION:-batch}}"
SLURM_TIME="${SLURM_TIME:-4:00:00}"

# Container image - use the working NeMo-RL container with proper vLLM and Ray setup
# This container has /opt/nemo_rl_venv properly configured for Ray execution
# Define cluster-specific container paths
CONTAINER_IMAGE_LAX="/scratch/fsw/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/yifuw/enroot-images/gitlab-master.nvidia.com/yifuw/images/nemo-rl:vllm-0_13_0-geshen__super_mtp-20260106.squashfs"
CONTAINER_IMAGE_DFW="/lustre/fsw/portfolios/llmservice/users/mfathi/containers/nemo-rl:vllm-0_13_0-geshen__super_mtp-20260106.squashfs"

# Select container based on detected cluster (can be overridden via CONTAINER_IMAGE env var)
if [ "$CLUSTER" = "lax" ]; then
    CONTAINER_IMAGE="${CONTAINER_IMAGE:-$CONTAINER_IMAGE_LAX}"
else
    CONTAINER_IMAGE="${CONTAINER_IMAGE:-$CONTAINER_IMAGE_DFW}"
fi

# Repository path - use canonical path to avoid symlink issues
# On this cluster /lustre is a symlink to /scratch, so we need the canonical path
REPO_LOCATION="${REPO_LOCATION:-$(readlink -f $(pwd))}"

# GPFS is used for mounting - also needs canonical path
GPFS="${GPFS:-$REPO_LOCATION}"

# Config file location
CONFIG_FILE="${CONFIG_FILE:-experiments/turing_vif_inverse_if/grpo_turing_vif_inverse_if.yaml}"

# Model to train (override to use different models)
# Note: Use local path instead of HF ID to avoid network issues with config download
POLICY_MODEL="${POLICY_MODEL:-/lustre/fsw/portfolios/llmservice/users/mfathi/hf_models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}"

# Data paths (defaults to preprocessed Inverse IF data)
TRAIN_DATA="${TRAIN_DATA:-3rdparty/Gym-workspace/Gym/resources_servers/turing_vif/data/inverse_if_8k.jsonl}"
VAL_DATA="${VAL_DATA:-3rdparty/Gym-workspace/Gym/resources_servers/turing_vif/data/inverse_if_8k.jsonl}"

# ============================================================================
# LLM JUDGE CONFIGURATION
# ============================================================================
# Spins up a separate vLLM instance for the LLM judge.
# Based on test_v23.sh reference script.
#
# DEFAULT: Qwen3-235B-A22B-Instruct FP8 (best quality + efficiency)
#   - 2 nodes (16 GPUs), TP=8, 2 replicas
#   - FP8 quantization for memory efficiency
#   - 64K context length
#   bash launch_turing_vif_inverse_if_training.sh
#   # Total: 6 nodes (4 policy + 2 judge)
#
# ALTERNATIVE CONFIGURATIONS:
#
# 1. Small Judge - Qwen3-8B (fast, lower quality, saves resources):
#    JUDGE_MODEL="/lustre/fsw/portfolios/llmservice/users/jiaqiz/models/Qwen3-8B" \
#    JUDGE_TP=4 \
#    JUDGE_NUM_NODES=1 \
#    JUDGE_ROUTER_DP_SIZE=1 \
#    JUDGE_GPU_UTIL=0.85 \
#    JUDGE_MAX_LEN=16384 \
#    JUDGE_ENABLE_EP=false \
#    JUDGE_MULTITHREAD_LOAD=false \
#    bash launch_turing_vif_inverse_if_training.sh
#    # Total: 5 nodes (4 policy + 1 judge)
#
# 2. Medium Judge - Qwen3-30B-A3B (balanced):
#    JUDGE_MODEL="/lustre/fsw/portfolios/llmservice/users/jiaqiz/models/Qwen3-30B-A3B-Thinking-2507" \
#    JUDGE_TP=4 \
#    JUDGE_NUM_NODES=1 \
#    JUDGE_ROUTER_DP_SIZE=1 \
#    bash launch_turing_vif_inverse_if_training.sh
#    # Total: 5 nodes (4 policy + 1 judge)
#
# To disable separate judge and use policy model:
#   USE_SEPARATE_JUDGE=false bash launch_turing_vif_inverse_if_training.sh
# ============================================================================
USE_SEPARATE_JUDGE="${USE_SEPARATE_JUDGE:-true}"

# Default: Qwen3-235B-A22B-Instruct FP8 (best quality + efficiency)
# Uses pre-downloaded local path for fast startup (same as test_v23.sh reference)
# LAX path: /scratch/fsw/portfolios/llmservice/users/jiaqiz/models/Qwen3-235B-A22B-Instruct-2507-FP8
# DFW path: /lustre/fsw/portfolios/llmservice/users/jiaqiz/models/Qwen3-235B-A22B-Instruct-2507-FP8
if [ "$CLUSTER" = "lax" ]; then
    DEFAULT_JUDGE_MODEL="/scratch/fsw/portfolios/llmservice/users/jiaqiz/models/Qwen3-235B-A22B-Instruct-2507-FP8"
else
    DEFAULT_JUDGE_MODEL="/lustre/fsw/portfolios/llmservice/users/jiaqiz/models/Qwen3-235B-A22B-Instruct-2507-FP8"
fi
JUDGE_MODEL="${JUDGE_MODEL:-$DEFAULT_JUDGE_MODEL}"
JUDGE_TP="${JUDGE_TP:-8}"                    # TP=8 for 235B (full node)
JUDGE_ROUTER_DP_SIZE="${JUDGE_ROUTER_DP_SIZE:-4}"  # 4 replicas (matches reference: GENRM_ROUTER_DP_SIZE=4)
JUDGE_NUM_NODES="${JUDGE_NUM_NODES:-4}"      # 4 nodes (32 GPUs) for 235B (matches reference: NUM_GENRM_NODES=4)
JUDGE_GPU_UTIL="${JUDGE_GPU_UTIL:-0.9}"      # Higher util with FP8
JUDGE_MAX_LEN="${JUDGE_MAX_LEN:-32768}"      # 32K context for evaluations
JUDGE_MAX_NUM_SEQS="${JUDGE_MAX_NUM_SEQS:-256}"  # Limit concurrent seqs to avoid OOM during sampler warmup
JUDGE_ENFORCE_EAGER="${JUDGE_ENFORCE_EAGER:-true}"  # Disable CUDA graphs to save ~3 GiB for forward pass activations
JUDGE_ENABLE_EP="${JUDGE_ENABLE_EP:-true}"   # Expert parallelism for MoE
JUDGE_MULTITHREAD_LOAD="${JUDGE_MULTITHREAD_LOAD:-true}"  # Fast model loading

# Hugging Face configuration
HF_HOME_PATH="${HF_HOME:-${HSH:-$HOME}/.cache/huggingface}"
HF_TOKEN_VALUE="${HF_TOKEN:-}"

# Weights & Biases configuration
WANDB_API_KEY_VALUE="${WANDB_API_KEY:-${WANDB_TOKEN:-}}"
WANDB_PROJECT="${WANDB_PROJECT:-nemo-rl-turing-vif-inverse-if}"

# VLLM configuration - use precompiled wheel for faster startup
VLLM_PRECOMPILED_WHEEL_LOCATION="${VLLM_PRECOMPILED_WHEEL_LOCATION:-https://github.com/vllm-project/vllm/releases/download/v0.13.0/vllm-0.13.0-cp38-abi3-manylinux_2_31_x86_64.whl}"
export VLLM_PRECOMPILED_WHEEL_LOCATION

# Code snapshot - disabled by default, set ENABLE_CODE_SNAPSHOT=true to enable
ENABLE_CODE_SNAPSHOT="${ENABLE_CODE_SNAPSHOT:-false}"

# Number of dependent SLURM jobs to chain (default: 1 = single job)
# When NUM_JOBS > 1, each subsequent job waits for the previous one to complete
# successfully (--dependency=afterok) and resumes training from its checkpoint.
# All jobs share the same checkpoint directory so resumption is automatic.
NUM_JOBS="${NUM_JOBS:-1}"

# ============================================================================
# DERIVED VARIABLES
# ============================================================================

# Ensure we're in the repo root
cd "$REPO_LOCATION"

# ============================================================================
# SUBMODULE INITIALIZATION
# ============================================================================
# Ensure the Gym submodule is initialized. Only check Gym (not vllm, which may
# not exist as a submodule in the public repo).
if [ ! -f "${REPO_LOCATION}/3rdparty/Gym-workspace/Gym/.git" ] && [ ! -d "${REPO_LOCATION}/3rdparty/Gym-workspace/Gym/.git" ]; then
    echo "Initializing Gym submodule..."
    git submodule update --init --recursive -- 3rdparty/Gym-workspace/Gym
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to initialize Gym submodule. Please run manually:"
        echo "  git submodule update --init --recursive -- 3rdparty/Gym-workspace/Gym"
        exit 1
    fi
    echo "Gym submodule initialized successfully."
fi

# ============================================================================
# UV LOCK FILE CHECK
# ============================================================================
# NOTE: We no longer run uv lock on the login node since it may not have the
# proper environment (e.g., missing CUDA, wrong Python version). Instead,
# the uv lock check is performed inside the container before launching training.
# Set UV_SKIP_LOCK=true to skip the in-container lock check entirely.
UV_LOCK_NEEDED="false"
if [ "${UV_SKIP_LOCK:-false}" != "true" ]; then
    if [ ! -f "${REPO_LOCATION}/uv.lock" ] || [ "${REPO_LOCATION}/pyproject.toml" -nt "${REPO_LOCATION}/uv.lock" ]; then
        echo "NOTE: uv.lock may need updating (pyproject.toml is newer or lock file missing)."
        echo "      This will be handled inside the container before training starts."
        UV_LOCK_NEEDED="true"
    fi
fi

# Create code snapshot for reproducibility (only copies git-tracked files)
# Enable with: ENABLE_CODE_SNAPSHOT=true bash launch_turing_vif_inverse_if_training.sh
if [ "$ENABLE_CODE_SNAPSHOT" = "true" ]; then
    SNAPSHOT_DIR=$(bash ${REPO_LOCATION}/tools/code_snapshot.sh ${EXP_NAME} 2>/dev/null || echo "")
    if [ -n "$SNAPSHOT_DIR" ] && [ -d "$SNAPSHOT_DIR" ]; then
        echo "Created code snapshot at: ${SNAPSHOT_DIR}"
        # Symlink 3rdparty/vllm submodule if not already present
        if [ -d "${REPO_LOCATION}/3rdparty/vllm" ] && [ ! -e "${SNAPSHOT_DIR}/3rdparty/vllm" ]; then
            echo "Symlinking 3rdparty/vllm to snapshot..."
            mkdir -p ${SNAPSHOT_DIR}/3rdparty
            ln -s ${REPO_LOCATION}/3rdparty/vllm ${SNAPSHOT_DIR}/3rdparty/vllm
        fi
        # Use snapshot directory for execution
        EXEC_DIR="$SNAPSHOT_DIR"
    else
        echo "Warning: Code snapshot not created, using repo directly"
        EXEC_DIR="$REPO_LOCATION"
    fi
else
    EXEC_DIR="$REPO_LOCATION"
fi

# Create descriptive log directory: logs/<exp_name>/<date>_<time>_<jobid>
# The job ID will be appended by ray.sub as: $BASE_LOG_DIR/$SLURM_JOB_ID-logs
LOG_BASE_DIR="${LOG_BASE_DIR:-logs/${EXP_NAME}}"
mkdir -p "$LOG_BASE_DIR"

# Shared checkpoint directory across all chained jobs.
# When NUM_JOBS > 1, all jobs read/write checkpoints here so each job
# automatically resumes from the last checkpoint saved by the previous job.
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${LOG_BASE_DIR}/checkpoints}"

# ============================================================================
# BUILD THE COMMAND
# ============================================================================

# Build judge configuration arguments
JUDGE_ARGS=""
if [ "$USE_SEPARATE_JUDGE" = "true" ]; then
    if [ -z "$JUDGE_MODEL" ]; then
        echo "ERROR: USE_SEPARATE_JUDGE=true but JUDGE_MODEL is not set"
        exit 1
    fi
    
    # Calculate total nodes (policy + judge)
    TOTAL_NODES=$((NUM_NODES + JUDGE_NUM_NODES))
    
    # Base judge arguments (local_vllm_model spins up its own vLLM instance)
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
    ++env.nemo_gym.judge_model.responses_api_models.local_vllm_model.vllm_serve_kwargs.max_num_seqs=${JUDGE_MAX_NUM_SEQS} \\
    ++env.nemo_gym.judge_model.responses_api_models.local_vllm_model.vllm_serve_kwargs.enforce_eager=${JUDGE_ENFORCE_EAGER} \\
    ++env.nemo_gym.judge_model.responses_api_models.local_vllm_model.vllm_serve_env_vars.VLLM_RAY_DP_PACK_STRATEGY=strict"

    # Add expert parallelism for large MoE models (e.g., Qwen3-235B)
    if [ "$JUDGE_ENABLE_EP" = "true" ]; then
        JUDGE_ARGS="${JUDGE_ARGS} \\
    ++env.nemo_gym.judge_model.responses_api_models.local_vllm_model.vllm_serve_kwargs.enable_expert_parallel=true"
    fi

    # Add multithread loading for faster startup with large models
    if [ "$JUDGE_MULTITHREAD_LOAD" = "true" ]; then
        JUDGE_ARGS="${JUDGE_ARGS} \\
    ++env.nemo_gym.judge_model.responses_api_models.local_vllm_model.vllm_serve_kwargs.model_loader_extra_config.enable_multithread_load=true \\
    ++env.nemo_gym.judge_model.responses_api_models.local_vllm_model.vllm_serve_kwargs.model_loader_extra_config.num_threads=112"
    fi
else
    TOTAL_NODES=$NUM_NODES
fi

# Construct the training command
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
export NRL_IGNORE_TP_ACCURACY_CHECK=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export NRL_FORCE_REBUILD_VENVS=true
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export UV_HTTP_TIMEOUT=300
export VLLM_USE_PRECOMPILED=1
export VLLM_PRECOMPILED_WHEEL_LOCATION=${VLLM_PRECOMPILED_WHEEL_LOCATION}

# Increase refit buffer ratio to ensure large embedding weights can be transferred
# NemotronH has ~700MB embedding that needs ~1.5GB buffer for ping-pong transfer
export NRL_REFIT_BUFFER_MEMORY_RATIO=0.9

# Update uv.lock inside the container if needed
# This runs on the head node only (SLURM_NODEID=0) to avoid race conditions
if [ "${UV_LOCK_NEEDED}" = "true" ] && [ "\${SLURM_NODEID:-0}" = "0" ]; then
    echo "Updating uv.lock inside container..."
    if command -v uv &> /dev/null; then
        uv lock --refresh 2>&1 || {
            echo "WARNING: uv lock failed inside container. Continuing anyway..."
        }
    else
        echo "WARNING: uv not found in container, skipping lock file update"
    fi
fi

# Run GRPO training with NeMo-Gym (Turing VIF Inverse IF)
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
echo "Launching Turing VIF Inverse IF GRPO Training"
echo "============================================================================"
echo "Experiment Name: ${EXP_NAME}"
echo "Policy Model:    ${POLICY_MODEL}"
echo "Policy Nodes:    ${NUM_NODES}"
if [ "$USE_SEPARATE_JUDGE" = "true" ]; then
echo "Judge Model:     ${JUDGE_MODEL}"
echo "Judge Config:    ${JUDGE_NUM_NODES} nodes, TP=${JUDGE_TP}, replicas=${JUDGE_ROUTER_DP_SIZE}, max_len=${JUDGE_MAX_LEN}"
if [ "$JUDGE_ENABLE_EP" = "true" ]; then
echo "                 Expert Parallel: enabled"
fi
if [ "$JUDGE_MULTITHREAD_LOAD" = "true" ]; then
echo "                 Multithread Load: enabled (112 threads)"
fi
echo "Total Nodes:     ${TOTAL_NODES}"
else
echo "Judge Model:     (using policy model)"
echo "Total Nodes:     ${TOTAL_NODES}"
fi
echo "Container:       ${CONTAINER_IMAGE}"
echo "SLURM Account:   ${SLURM_ACCOUNT}"
echo "SLURM Partition: ${SLURM_PARTITION}"
echo "Time Limit:      ${SLURM_TIME}"
echo "Num Jobs:        ${NUM_JOBS} (chained with afterok dependency)"
echo "Log Directory:   ${LOG_BASE_DIR}/<job_id>-logs/"
echo "Checkpoints:     ${CHECKPOINT_DIR}/"
echo "============================================================================"
echo ""
echo "Training Command:"
echo "$COMMAND"
echo ""
echo "============================================================================"

# Mount paths - mount the filesystem and overlay 3rdparty directories into /opt/nemo-rl
# IMPORTANT: Don't override /opt/nemo-rl completely to preserve flashinfer-workspace in container
#
# NOTE: On LAX, /lustre is a symlink to /scratch, so we mount /scratch directly.
# On DFW, we mount /lustre directly.
# Mount the specific subdirectories (Gym, Megatron-LM, vllm) to match the container paths.
# Also mount the execution directory (snapshot or repo) for the code to run from.
MOUNT_STRING="${EXEC_DIR}:${EXEC_DIR},${EXEC_DIR}/3rdparty/Gym-workspace/Gym:/opt/nemo-rl/3rdparty/Gym-workspace/Gym,${GPFS}/3rdparty/Megatron-LM-workspace/Megatron-LM:/opt/nemo-rl/3rdparty/Megatron-LM-workspace/Megatron-LM"

# Mount the shared filesystem based on cluster
if [ "$CLUSTER" = "lax" ]; then
    # On LAX, /lustre is a symlink to /scratch on the host.
    # Mount both so that paths using either /scratch or /lustre work inside the container.
    # This is critical because HF cache and model paths may use /lustre/... which needs to resolve.
    MOUNT_STRING="/scratch:/scratch,/lustre:/lustre,${MOUNT_STRING}"
else
    # On DFW and other clusters, mount /lustre directly
    # This is required for HF_HOME and model checkpoints to be accessible across nodes
    MOUNT_STRING="/lustre:/lustre,${MOUNT_STRING}"
fi

# Mount vllm submodule if it exists, otherwise use container's built-in vllm
if [ -d "${GPFS}/3rdparty/vllm" ]; then
    MOUNT_STRING="${MOUNT_STRING},${GPFS}/3rdparty/vllm:/opt/nemo-rl/3rdparty/vllm"
fi

# Submit the SLURM job(s)
# Note: CPUS_PER_WORKER is set to match the cluster config (112 for interactive partition)
# ray.sub defaults to 8*16=128 which may exceed node capacity
# --exclusive: ensures full node allocation for GPU jobs
# --dependency: singleton for first job, afterok:PREV_JOBID for subsequent jobs
# BASE_LOG_DIR: sets where logs are stored (ray.sub creates $BASE_LOG_DIR/$SLURM_JOB_ID-logs)
#
# When NUM_JOBS > 1, jobs are chained so each waits for the previous to finish
# successfully. All jobs share CHECKPOINT_DIR, so nemo-rl's CheckpointManager
# automatically finds the latest checkpoint and resumes training from there.
PREV_JOB_ID=""
SUBMITTED_JOBS=()

for JOB_NUM in $(seq 1 "${NUM_JOBS}"); do
    # First job uses singleton dependency (prevents collisions with same job name).
    # Subsequent jobs use afterok (wait for previous job to succeed).
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
if [ "${NUM_JOBS}" -gt 1 ]; then
echo ""
echo "   Jobs are chained: each resumes from the previous job's checkpoint."
echo "   Monitor chain:  squeue -u \$USER --name=${EXP_NAME}"
fi
