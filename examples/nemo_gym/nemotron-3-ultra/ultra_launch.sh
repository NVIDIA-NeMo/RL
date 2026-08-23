#!/bin/bash
set -euo pipefail

# =============================================================================
# ultra_launch.sh
#
# Public launcher for Nemotron 3 Ultra post-training stages on a SLURM cluster.
#
# Each training stage (Student RLVR, teacher RLVR/RLHF stages, MOPD) has a
# matching YAML config under examples/nemo_gym/nemotron-3-ultra/. The stage-specific
# hyperparameters (batch size, advantage clip, MoE parallelism, etc.) live
# in the YAML; this launcher only handles orchestration: SLURM submission,
# code snapshotting, persistent cache management, container mounts, and the
# Hydra overrides that vary per run (data paths, model checkpoint, judge
# endpoints, log directories).
#
# Usage:
#
#   EXP_NAME=ultra-student-rlvr-001 \
#   CONFIG_PATH=examples/nemo_gym/nemotron-3-ultra/student_rlvr1.yaml \
#   MODEL_PATH=/path/to/sft_checkpoint \
#   TRAIN_PATH=/path/to/train.jsonl \
#   VAL_PATH=/path/to/val.jsonl \
#   CONTAINER=nvcr.io/nvidia/nemo-rl:<tag> \
#   SANDBOX_CONTAINER=/path/to/nemo-skills-sandbox.sqsh \
#   PERSISTENT_CACHE=/path/to/persistent/cache \
#   SLURM_PARTITION=batch \
#   SLURM_ACCOUNT=your_account \
#   GENRM_MODEL=<HF repo id or local path>      # Or set GENRM_BASE_URL to use a remote service
#   NL2BASH_JUDGE_MODEL=Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 \   # Or NL2BASH_BASE_URL
#   EXTERNAL_JUDGES=1                           # Serve GenRM and NL2Bash outside the training Ray cluster
#   SAFETY_JUDGE_MODEL=/path/to/safety_checkpoint \
#   bash examples/nemo_gym/nemotron-3-ultra/ultra_launch.sh
#
# Optional knobs:
#   WALLTIME=4:00:00                       Slurm --time
#   SLURM_QOS=                             Slurm --qos; defaults to short when
#                                          WALLTIME is under two hours
#   SLURM_RESERVATION=                     Slurm --reservation
#   SLURM_DEPENDENCY=                      Extra Slurm dependency, merged with
#                                          singleton (e.g. afterany:<jobid>)
#   EXCLUDE_NODES=                         Slurm --exclude
#   NUM_TRAIN_NODES=64                     Training (Megatron) nodes
#   NUM_GEN_NODES=172                      vLLM generation nodes
#   NUM_GYM_NODES=20                       NeMo Gym (judge) nodes
#   EXTERNAL_JUDGES=0                      1 to serve GenRM and NL2Bash as
#                                          external vLLM pools in a second
#                                          Slurm hetgroup instead of inside Gym
#   GENRM_REPLICAS=4                       External GenRM servers (DP=1 each)
#   GENRM_TENSOR_PARALLEL_SIZE=4           TP per external GenRM server
#   GENRM_REASONING_PARSER=                Reasoning-parser plugin .py; only
#                                          needed for a parser vLLM lacks
#   GENRM_REASONING_PARSER_NAME=nemotron_v3
#   NL2BASH_REPLICAS=4                     External NL2Bash servers (DP=1 each)
#   NL2BASH_TENSOR_PARALLEL_SIZE=4         TP per external NL2Bash server
#   EXTERNAL_VLLM_SEGMENT_SIZE=4           Segment size for the external hetgroup
#   BATCH_SCRIPT=ray.sub                   Slurm entrypoint; external services
#                                          wrap ray.sub
#   ENABLE_MTP_INFERENCE=0                 1 to enable MTP speculative decoding
#   NUM_SPECULATIVE_TOKENS=5               MTP speculative tokens
#   MAX_NUM_BATCHED_TOKENS=8480            vLLM max batched tokens (MTP)
#   NRL_MAX_STEPS=                         Override grpo.max_num_steps
#   EXTRA_MOUNTS=                          Comma-separated host:container pairs
#   USE_SNAPSHOT=1                         Snapshot source tree at submission
#   DRY_RUN=0                              1 to print TRAIN_CMD and exit
#   INTERACTIVE=0                          1 to bring up Ray and idle for attach
#                                          (no training driver) for debugging
#   INTERACTIVE_WAIT=1                     0 to submit and return immediately
#   INTERACTIVE_WALLTIME=                  override WALLTIME for the interactive alloc
#   HF_HOME=                               HuggingFace cache root (recommended)
#   HF_TOKEN=                              HuggingFace API token
#   WANDB_API_KEY=                         Weights & Biases API key
#   WANDB_PROJ=nemotron-3-ultra            W&B project
#   WANDB_ENTITY=                          W&B entity
#
# Hydra overrides are forwarded verbatim as positional arguments:
#   bash examples/nemo_gym/nemotron-3-ultra/ultra_launch.sh policy.megatron_cfg.optimizer.lr=1e-6 grpo.val_period=50
#
# GB200 NVL72 nodes have 4 GPUs each. SLURM total = NUM_TRAIN + NUM_GEN + NUM_GYM
# and must be a multiple of SEGMENT_SIZE (default 16, one NVLink domain group).
# =============================================================================

# =============================================================================
# Required environment
# =============================================================================
: "${EXP_NAME:?EXP_NAME is required (used for job name, W&B run, checkpoint/log dirs)}"
: "${CONFIG_PATH:?CONFIG_PATH is required (e.g. examples/nemo_gym/nemotron-3-ultra/student_rlvr1.yaml)}"
: "${MODEL_PATH:?MODEL_PATH is required (initial policy checkpoint, HF repo id or local path)}"
: "${TRAIN_PATH:?TRAIN_PATH is required (training data jsonl path)}"
: "${VAL_PATH:?VAL_PATH is required (validation data jsonl path)}"
: "${CONTAINER:?CONTAINER is required (NGC image URI or .sqsh path)}"
: "${SANDBOX_CONTAINER:?SANDBOX_CONTAINER is required (nemo-skills sandbox image)}"
: "${PERSISTENT_CACHE:?PERSISTENT_CACHE is required (Lustre dir for vLLM/Triton/Inductor caches)}"
: "${SLURM_PARTITION:?SLURM_PARTITION is required}"
: "${SLURM_ACCOUNT:?SLURM_ACCOUNT is required}"
# Judge models are recipe-specific. Most teachers (student RLVR, IFBench, RLHF,
# Reasoning) need all three (GenRM, NL2Bash, Safety). The SWE teacher uses
# code-execution rewards and needs none of them. Set per recipe; unset vars
# skip the corresponding override.
NL2BASH_JUDGE_MODEL="${NL2BASH_JUDGE_MODEL:-}"
NL2BASH_BASE_URL="${NL2BASH_BASE_URL:-}"
NL2BASH_API_MODEL_NAME="${NL2BASH_API_MODEL_NAME:-}"
SAFETY_JUDGE_MODEL="${SAFETY_JUDGE_MODEL:-}"
GENRM_BASE_URL="${GENRM_BASE_URL:-}"
GENRM_MODEL="${GENRM_MODEL:-}"
GENRM_API_MODEL_NAME="${GENRM_API_MODEL_NAME:-}"

# SIF_DIR: for the SWE teacher recipe — directory containing apptainer .sif
# images for SWE-Bench / SWE-Gym / R2E-Gym instances. The yaml's
# container_formatter uses `${sif_dir}/...` paths. Unset for non-SWE recipes.
SIF_DIR="${SIF_DIR:-}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "ERROR: CONFIG_PATH does not exist: ${CONFIG_PATH}" >&2
  exit 1
fi

# The SWE teacher recipe interpolates `${sif_dir}/...` paths at runtime. The
# exemplar config carries only a placeholder, so hard-require SIF_DIR whenever
# the selected config actually uses it (mirrors the mopd teacher-path guard).
if grep -q '${sif_dir}' "${CONFIG_PATH}"; then
  : "${SIF_DIR:?SIF_DIR is required for the SWE recipe (directory of apptainer .sif images)}"
fi

# =============================================================================
# Project root and code root
# =============================================================================
PROJECT_ROOT=$(realpath "$PWD")
cd "${PROJECT_ROOT}"

# =============================================================================
# External judge services (EXTERNAL_JUDGES=1)
#
# GenRM and NL2Bash are served as fixed-model vLLM pools in a second Slurm
# heterogeneous component, outside the training Ray cluster. The wrapper starts
# every replica plus one load balancer per pool, then substitutes each pool's
# URL placeholder in COMMAND before starting ray.sub on hetgroup 0.
#
# Replica defaults mirror the in-Gym deployment in student_rlvr1.yaml
# (TP=4, DP=4 per judge). Phase 2 typically runs a larger GenRM pool
# (GENRM_REPLICAS=16), matching student_rlvr2.yaml.
# =============================================================================
EXTERNAL_JUDGES="${EXTERNAL_JUDGES:-0}"
NUM_EXTERNAL_SERVICE_NODES=0
# Pool sizing divides replica GPUs by GPUS_PER_NODE, so resolve it before
# registration (re-exported with the other container settings below).
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
if [[ "${EXTERNAL_JUDGES}" == "1" ]]; then
  # Stages declare only the judges they use: rlhf_teacher has no NL2Bash block,
  # reasoning_teacher has no GenRM block, swe_teacher has neither. Each pool is
  # registered only when its model is set, so a stage never receives an override
  # for a judge its config does not declare.
  if [[ -z "${GENRM_MODEL}" && -z "${NL2BASH_JUDGE_MODEL}" ]]; then
    echo "ERROR: EXTERNAL_JUDGES=1 requires GENRM_MODEL, NL2BASH_JUDGE_MODEL, or both." >&2
    exit 1
  fi
  if [[ -n "${GENRM_BASE_URL}" ]]; then
    echo "ERROR: GENRM_BASE_URL and EXTERNAL_JUDGES=1 are mutually exclusive." >&2
    exit 1
  fi
  if [[ -n "${NL2BASH_BASE_URL}" ]]; then
    echo "ERROR: NL2BASH_BASE_URL and EXTERNAL_JUDGES=1 are mutually exclusive." >&2
    exit 1
  fi

  # Deployment-specific service definitions stay in this launcher; the
  # allocation wrapper consumes only pools registered through this interface.
  source "${PROJECT_ROOT}/tools/external_gym_vllm/pool_config.sh"
  EXTERNAL_VLLM_POOLS=""
  EXTERNAL_VLLM_TOOLS_DIR_HOST="${EXTERNAL_VLLM_TOOLS_DIR_HOST:-${PROJECT_ROOT}/tools/external_gym_vllm}"
  EXTERNAL_VLLM_LB_PYTHON="${EXTERNAL_VLLM_LB_PYTHON:-/opt/nemo_rl_venv/bin/python}"
fi

if [[ "${EXTERNAL_JUDGES}" == "1" && -n "${GENRM_MODEL}" ]]; then
  GENRM_BASE_URL="__GENRM_BASE_URL__"
  GENRM_REPLICAS="${GENRM_REPLICAS:-4}"
  GENRM_TENSOR_PARALLEL_SIZE="${GENRM_TENSOR_PARALLEL_SIZE:-4}"
  GENRM_SERVED_MODEL_NAME="${GENRM_SERVED_MODEL_NAME:-model}"
  GENRM_API_MODEL_NAME="${GENRM_API_MODEL_NAME:-${GENRM_SERVED_MODEL_NAME}}"
  GENRM_VLLM_PORT="${GENRM_VLLM_PORT:-8000}"
  GENRM_LB_PORT="${GENRM_LB_PORT:-9213}"
  GENRM_STARTUP_TIMEOUT="${GENRM_STARTUP_TIMEOUT:-3600}"
  GENRM_CONTAINER="${GENRM_CONTAINER:-${CONTAINER}}"
  GENRM_VLLM_PYTHON="${GENRM_VLLM_PYTHON:-/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python}"
  # nemotron_v3 is built into vLLM and parses the <think>/</think> format the
  # GenRM emits. Set GENRM_REASONING_PARSER only for a checkpoint that needs a
  # parser vLLM does not ship.
  GENRM_REASONING_PARSER="${GENRM_REASONING_PARSER:-}"
  GENRM_REASONING_PARSER_NAME="${GENRM_REASONING_PARSER_NAME:-nemotron_v3}"
  GENRM_TOOL_CALL_PARSER="${GENRM_TOOL_CALL_PARSER:-qwen3_coder}"
  GENRM_ENABLE_EXPERT_PARALLEL="${GENRM_ENABLE_EXPERT_PARALLEL:-1}"
  GENRM_COMPILATION_CONFIG="${GENRM_COMPILATION_CONFIG:-{\"pass_config\":{\"fuse_allreduce_rms\":false}}}"
  GENRM_MODEL_LOADER_EXTRA_CONFIG="${GENRM_MODEL_LOADER_EXTRA_CONFIG:-{\"enable_multithread_load\":true,\"num_threads\":96}}"

fi

if [[ "${EXTERNAL_JUDGES}" == "1" && -n "${NL2BASH_JUDGE_MODEL}" ]]; then
  NL2BASH_BASE_URL="__NL2BASH_BASE_URL__"
  NL2BASH_REPLICAS="${NL2BASH_REPLICAS:-4}"
  NL2BASH_TENSOR_PARALLEL_SIZE="${NL2BASH_TENSOR_PARALLEL_SIZE:-4}"
  NL2BASH_SERVED_MODEL_NAME="${NL2BASH_SERVED_MODEL_NAME:-model}"
  NL2BASH_API_MODEL_NAME="${NL2BASH_API_MODEL_NAME:-${NL2BASH_SERVED_MODEL_NAME}}"
  NL2BASH_VLLM_PORT="${NL2BASH_VLLM_PORT:-8000}"
  NL2BASH_LB_PORT="${NL2BASH_LB_PORT:-9214}"
  NL2BASH_STARTUP_TIMEOUT="${NL2BASH_STARTUP_TIMEOUT:-3600}"
  # GenRM's values when that pool is also registered, else the shared defaults.
  NL2BASH_CONTAINER="${NL2BASH_CONTAINER:-${GENRM_CONTAINER:-${CONTAINER}}}"
  NL2BASH_VLLM_PYTHON="${NL2BASH_VLLM_PYTHON:-${GENRM_VLLM_PYTHON:-/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python}}"
  NL2BASH_TOOL_CALL_PARSER="${NL2BASH_TOOL_CALL_PARSER:-hermes}"
  NL2BASH_ENABLE_EXPERT_PARALLEL="${NL2BASH_ENABLE_EXPERT_PARALLEL:-1}"
  NL2BASH_ATTENTION_BACKEND="${NL2BASH_ATTENTION_BACKEND:-TRITON_ATTN}"
  NL2BASH_MAX_MODEL_LEN="${NL2BASH_MAX_MODEL_LEN:-131072}"
  NL2BASH_COMPILATION_CONFIG="${NL2BASH_COMPILATION_CONFIG:-{\"cudagraph_capture_sizes\":[1,2,4,8,16,32,64,128,256]}}"
  NL2BASH_MODEL_LOADER_EXTRA_CONFIG="${NL2BASH_MODEL_LOADER_EXTRA_CONFIG:-{\"enable_multithread_load\":true,\"num_threads\":112}}"
fi

if [[ "${EXTERNAL_JUDGES}" == "1" && -n "${GENRM_MODEL}" ]]; then
  register_external_vllm_pool GENRM \
    --display-name GenRM \
    --model "${GENRM_MODEL}" \
    --container "${GENRM_CONTAINER}" \
    --python "${GENRM_VLLM_PYTHON}" \
    --replicas "${GENRM_REPLICAS}" \
    --tensor-parallel-size "${GENRM_TENSOR_PARALLEL_SIZE}" \
    --served-model-name "${GENRM_SERVED_MODEL_NAME}" \
    --vllm-port "${GENRM_VLLM_PORT}" \
    --lb-port "${GENRM_LB_PORT}" \
    --startup-timeout "${GENRM_STARTUP_TIMEOUT}" \
    --url-placeholder "${GENRM_BASE_URL}" \
    ${GENRM_REASONING_PARSER:+--shared-path "${GENRM_REASONING_PARSER}"}
  external_vllm_pool_env GENRM \
    "FLASHINFER_WORKSPACE_BASE=/tmp" \
    "VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm" \
    "VLLM_ALLREDUCE_USE_SYMM_MEM=0"
  genrm_vllm_args=(
    --trust-remote-code
    --dtype bfloat16
    --kv-cache-dtype fp8
    --max-num-seqs 256
    --gpu-memory-utilization 0.95
    --enable-prefix-caching
    --reasoning-parser "${GENRM_REASONING_PARSER_NAME}"
    --enable-auto-tool-choice
    --tool-call-parser "${GENRM_TOOL_CALL_PARSER}"
    --compilation-config "${GENRM_COMPILATION_CONFIG}"
    --model-loader-extra-config "${GENRM_MODEL_LOADER_EXTRA_CONFIG}"
  )
  [[ -n "${GENRM_REASONING_PARSER}" ]] && genrm_vllm_args+=(--reasoning-parser-plugin "${GENRM_REASONING_PARSER}")
  [[ "${GENRM_ENABLE_EXPERT_PARALLEL}" == "1" ]] && genrm_vllm_args+=(--enable-expert-parallel)
  external_vllm_pool_args GENRM "${genrm_vllm_args[@]}"
fi

if [[ "${EXTERNAL_JUDGES}" == "1" && -n "${NL2BASH_JUDGE_MODEL}" ]]; then
  register_external_vllm_pool NL2BASH \
    --display-name NL2Bash \
    --model "${NL2BASH_JUDGE_MODEL}" \
    --container "${NL2BASH_CONTAINER}" \
    --python "${NL2BASH_VLLM_PYTHON}" \
    --replicas "${NL2BASH_REPLICAS}" \
    --tensor-parallel-size "${NL2BASH_TENSOR_PARALLEL_SIZE}" \
    --served-model-name "${NL2BASH_SERVED_MODEL_NAME}" \
    --vllm-port "${NL2BASH_VLLM_PORT}" \
    --lb-port "${NL2BASH_LB_PORT}" \
    --startup-timeout "${NL2BASH_STARTUP_TIMEOUT}" \
    --url-placeholder "${NL2BASH_BASE_URL}"
  external_vllm_pool_env NL2BASH \
    "FLASHINFER_WORKSPACE_BASE=/tmp" \
    "VLLM_USE_FLASHINFER_MOE_FP16=0" \
    "VLLM_USE_FLASHINFER_MOE_FP8=0" \
    "VLLM_USE_DEEP_GEMM=0" \
    "VLLM_MOE_USE_DEEP_GEMM=0" \
    "NCCL_MNNVL_ENABLE=1"
  nl2bash_vllm_args=(
    --dtype bfloat16
    --pipeline-parallel-size 1
    --max-model-len "${NL2BASH_MAX_MODEL_LEN}"
    --max-num-seqs 256
    --gpu-memory-utilization 0.85
    --enable-prefix-caching
    --enable-chunked-prefill
    --enable-auto-tool-choice
    --tool-call-parser "${NL2BASH_TOOL_CALL_PARSER}"
    --attention-backend "${NL2BASH_ATTENTION_BACKEND}"
    --compilation-config "${NL2BASH_COMPILATION_CONFIG}"
    --model-loader-extra-config "${NL2BASH_MODEL_LOADER_EXTRA_CONFIG}"
  )
  [[ "${NL2BASH_ENABLE_EXPERT_PARALLEL}" == "1" ]] && nl2bash_vllm_args+=(--enable-expert-parallel)
  external_vllm_pool_args NL2BASH "${nl2bash_vllm_args[@]}"
fi

if [[ "${EXTERNAL_JUDGES}" == "1" ]]; then
  NUM_EXTERNAL_SERVICE_NODES="${EXTERNAL_VLLM_NUM_NODES}"
  export EXTERNAL_VLLM_LB_PYTHON EXTERNAL_VLLM_POOLS EXTERNAL_VLLM_TOOLS_DIR_HOST
fi

# Judge endpoints: a base_url routes the Gym server at a remote service and
# skips the local deployment; otherwise Gym serves the model itself.
GENRM_OVERRIDE=""
if [[ -n "${GENRM_BASE_URL}" ]]; then
  # Gym requires a model name even when it only proxies to base_url, so fall
  # back to the checkpoint when the served name was not given explicitly.
  GENRM_API_MODEL_NAME="${GENRM_API_MODEL_NAME:-${GENRM_MODEL}}"
  GENRM_OVERRIDE="++env.nemo_gym.genrm_model.responses_api_models.genrm_model.base_url=${GENRM_BASE_URL}"
  if [[ -n "${GENRM_API_MODEL_NAME}" ]]; then
    GENRM_OVERRIDE="${GENRM_OVERRIDE} env.nemo_gym.genrm_model.responses_api_models.genrm_model.model=${GENRM_API_MODEL_NAME}"
  fi
elif [[ -n "${GENRM_MODEL}" ]]; then
  GENRM_OVERRIDE="env.nemo_gym.genrm_model.responses_api_models.genrm_model.model=${GENRM_MODEL}"
fi
NL2BASH_OVERRIDE=""
if [[ -n "${NL2BASH_BASE_URL}" ]]; then
  NL2BASH_API_MODEL_NAME="${NL2BASH_API_MODEL_NAME:-${NL2BASH_JUDGE_MODEL}}"
  NL2BASH_OVERRIDE="++env.nemo_gym.nl2bash_judge_model.responses_api_models.local_vllm_model.base_url=${NL2BASH_BASE_URL}"
  if [[ -n "${NL2BASH_API_MODEL_NAME}" ]]; then
    NL2BASH_OVERRIDE="${NL2BASH_OVERRIDE} env.nemo_gym.nl2bash_judge_model.responses_api_models.local_vllm_model.model=${NL2BASH_API_MODEL_NAME}"
  fi
elif [[ -n "${NL2BASH_JUDGE_MODEL}" ]]; then
  NL2BASH_OVERRIDE="env.nemo_gym.nl2bash_judge_model.responses_api_models.local_vllm_model.model=${NL2BASH_JUDGE_MODEL}"
fi

# =============================================================================
# Job identity — fixed name for singleton.
# Slurm --dependency=singleton serialises queued submissions with the same name
# so a resubmission after preemption resumes from the latest checkpoint instead
# of running in parallel.
# =============================================================================
JOB_NAME="${EXP_NAME}"

# =============================================================================
# Output directories
# =============================================================================
# Resolved to absolute paths: the driver runs from ${CODE_ROOT} inside the
# container, and the external-service wrapper rejects relative paths.
RESULTS_DIR="$(realpath -m -s "${RESULTS_DIR:-results/${EXP_NAME}}")"
CHECKPOINT_DIR="$(realpath -m -s "${CHECKPOINT_DIR:-${RESULTS_DIR}/checkpoints}")"

# Per-submission dirs for logs and Slurm output (timestamped for history).
RUN_DIR="${RESULTS_DIR}/runs/$(date +%Y%m%d-%H%M)"
LOG_DIR="${RUN_DIR}/logs"
SLURM_LOG_DIR="${RUN_DIR}/slurm"
BASE_LOG_DIR="$(realpath -m -s "${BASE_LOG_DIR:-${RESULTS_DIR}/ray_logs}")"

# The wrapper bind-mounts EXTERNAL_VLLM_SHARED_ROOT into the service containers,
# so both paths must resolve under it. Check before creating any directories.
if (( NUM_EXTERNAL_SERVICE_NODES > 0 )); then
  _shared_root="${EXTERNAL_VLLM_SHARED_ROOT:-/lustre}"
  for _var in BASE_LOG_DIR EXTERNAL_VLLM_TOOLS_DIR_HOST; do
    _path="$(realpath -m -s "${!_var}")"
    if [[ "${_path}" != "${_shared_root}" && "${_path}" != "${_shared_root}"/* ]]; then
      echo "ERROR: ${_var}=${_path} must be under EXTERNAL_VLLM_SHARED_ROOT=${_shared_root} when EXTERNAL_JUDGES=1." >&2
      echo "  Set ${_var} (or RESULTS_DIR) to a path on the shared filesystem, or set EXTERNAL_VLLM_SHARED_ROOT." >&2
      exit 1
    fi
    printf -v "${_var}" '%s' "${_path}"
  done
  export EXTERNAL_VLLM_TOOLS_DIR_HOST
fi

mkdir -p "${CHECKPOINT_DIR}" "${LOG_DIR}" "${SLURM_LOG_DIR}"
ln -sfn "${RUN_DIR}" "${RESULTS_DIR}/runs/latest"

# ray.sub reads BASE_LOG_DIR and creates $BASE_LOG_DIR/$SLURM_JOB_ID-logs/ for
# ray infrastructure logs (ray-head.log, ray-driver.log, ray-worker-*.log,
# topology probes, attach scripts, etc.).
export BASE_LOG_DIR

# =============================================================================
# SLURM configuration
# =============================================================================
WALLTIME="${WALLTIME:-4:00:00}"
SLURM_QOS="${SLURM_QOS:-}"
SLURM_RESERVATION="${SLURM_RESERVATION:-}"
EXCLUDE_NODES="${EXCLUDE_NODES:-}"

slurm_walltime_seconds() {
  local value="$1"
  local days=0
  local -a fields

  if [[ "${value}" == *-* ]]; then
    days="${value%%-*}"
    value="${value#*-}"
  fi
  [[ "${days}" =~ ^[0-9]+$ ]] || return 1

  IFS=: read -r -a fields <<< "${value}"
  for field in "${fields[@]}"; do
    [[ "${field}" =~ ^[0-9]+$ ]] || return 1
  done

  case "${#fields[@]}" in
    1)
      if (( days > 0 )); then
        echo $((10#${days} * 86400 + 10#${fields[0]} * 3600))
      else
        echo $((10#${fields[0]} * 60))
      fi
      ;;
    2)
      if (( days > 0 )); then
        echo $((10#${days} * 86400 + 10#${fields[0]} * 3600 + 10#${fields[1]} * 60))
      else
        echo $((10#${fields[0]} * 60 + 10#${fields[1]}))
      fi
      ;;
    3)
      echo $((10#${days} * 86400 + 10#${fields[0]} * 3600 + 10#${fields[1]} * 60 + 10#${fields[2]}))
      ;;
    *) return 1 ;;
  esac
}

if [[ -z "${SLURM_QOS}" ]]; then
  if WALLTIME_SECONDS="$(slurm_walltime_seconds "${WALLTIME}")"; then
    if (( WALLTIME_SECONDS < 2 * 60 * 60 )); then
      SLURM_QOS=short
    fi
  else
    echo "[WARN] Could not parse WALLTIME=${WALLTIME}; leaving SLURM_QOS unset." >&2
  fi
fi
# INTERACTIVE=1 brings up the Ray cluster and idles for attachment (no training
# driver), so you can run/debug the recipe by hand. INTERACTIVE_WAIT=1 (default)
# blocks until Ray is ready; INTERACTIVE_WALLTIME overrides WALLTIME for the alloc.
INTERACTIVE="${INTERACTIVE:-0}"
INTERACTIVE_WAIT="${INTERACTIVE_WAIT:-1}"
# If set (format DD:HH:MM:SS), training stops early to reserve time for a final
# checkpoint save before walltime. Unset to use the YAML's default and let
# slurm walltime end the job naturally — fine when each step checkpoints.
CHECKPOINTING_SAVE_BY="${CHECKPOINTING_SAVE_BY:-}"

# =============================================================================
# Container & mounts
# =============================================================================
export CONTAINER
MOUNTS="${MOUNTS:-}"

# GB200 NVL72 defaults to 4 GPUs/node. Allow H100 smoke configs to request
# their native 8-GPU node shape through the launch environment.
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export CPUS_PER_WORKER="${CPUS_PER_WORKER:-144}"

# =============================================================================
# HuggingFace configuration
# =============================================================================
if [[ -n "${HF_HOME:-}" ]]; then
  export HF_HOME
  export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/hub}"
else
  echo "[WARN] HF_HOME is not set — HuggingFace will use the default cache (~/.cache/huggingface) per-node." >&2
fi

# =============================================================================
# W&B configuration
# =============================================================================
WANDB_PROJ="${WANDB_PROJ:-nemotron-3-ultra}"
WANDB_NAME="${EXP_NAME}"
WANDB_ENABLED=False
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  export WANDB_API_KEY
  WANDB_ENABLED=True
  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    export WANDB_ENTITY
  fi
else
  echo "[WARN] WANDB_API_KEY is not set — W&B logging will be disabled." >&2
fi

# =============================================================================
# Training overrides
# =============================================================================
NRL_MAX_STEPS="${NRL_MAX_STEPS:-}"

# =============================================================================
# MTP speculative decoding (optional)
# =============================================================================
ENABLE_MTP_INFERENCE="${ENABLE_MTP_INFERENCE:-0}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-5}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8480}"
MTP_EXTRA_ARGS=""
if [[ "${ENABLE_MTP_INFERENCE}" == "1" ]]; then
  MTP_EXTRA_ARGS="\
++policy.generation.vllm_cfg.enable_prefix_caching=true \
++policy.generation.vllm_kwargs.enable_chunked_prefill=true \
++policy.generation.vllm_kwargs.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS} \
++policy.generation.vllm_kwargs.mamba_cache_mode=align \
~policy.generation.vllm_kwargs.compilation_config.cudagraph_capture_sizes \
++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=${NUM_SPECULATIVE_TOKENS} \
++policy.generation.vllm_kwargs.speculative_config.method=mtp"
  echo "MTP speculative decoding ENABLED (num_speculative_tokens=${NUM_SPECULATIVE_TOKENS})"
fi

# =============================================================================
# Job shape — defaults match the 256-node student_rlvr1.yaml
#
#   Training:  64 nodes ( 256 GPUs) — Megatron training backend
#   vLLM:     172 nodes ( 688 GPUs) — async generation, EP=8 instances at TP=8
#   Gym:       20 nodes (  80 GPUs) — judges (GenRM, NL2Bash, Safety)
#
# Override via NUM_TRAIN_NODES / NUM_GEN_NODES / NUM_GYM_NODES.
#
# For STAGE_TYPE=mopd, additional teacher nodes are allocated for the
# non-colocated teacher panel: NUM_UNIQUE_TEACHERS × NUM_NODES_PER_TEACHER.
# =============================================================================
NUM_TRAIN_NODES="${NUM_TRAIN_NODES:-64}"
NUM_GEN_NODES="${NUM_GEN_NODES:-172}"
NUM_GYM_NODES="${NUM_GYM_NODES:-20}"

STAGE_TYPE="${STAGE_TYPE:-grpo}"
NUM_TEACHER_NODES=0
MOPD_OVERRIDES=""
if [[ "${STAGE_TYPE}" == "mopd" ]]; then
  : "${NRL_GENERAL_TEACHER_PATH:?NRL_GENERAL_TEACHER_PATH is required for STAGE_TYPE=mopd (path to the Student RLVR output checkpoint)}"
  NUM_UNIQUE_TEACHERS="${NUM_UNIQUE_TEACHERS:-5}"
  NUM_NODES_PER_TEACHER="${NUM_NODES_PER_TEACHER:-4}"
  NUM_TEACHER_NODES=$((NUM_UNIQUE_TEACHERS * NUM_NODES_PER_TEACHER))
  TEACHER_TP="${TEACHER_TP:-8}"
  TEACHER_CP="${TEACHER_CP:-2}"
  TEACHER_PP="${TEACHER_PP:-1}"
  TEACHER_EP="${TEACHER_EP:-16}"

  # _teachers.general is required; other slots fall back via the YAML's
  # interpolation. Pass only the slots the user explicitly set.
  MOPD_OVERRIDES="_teachers.general=${NRL_GENERAL_TEACHER_PATH}"
  for _slot in RLHF IFBENCH REASONING SWE; do
    _var="NRL_${_slot}_TEACHER_PATH"
    _val="${!_var:-}"
    if [[ -n "${_val}" ]]; then
      MOPD_OVERRIDES="${MOPD_OVERRIDES} _teachers.$(echo ${_slot} | tr A-Z a-z)=${_val}"
    fi
  done

  # Teacher parallelism + per-teacher node count
  MOPD_OVERRIDES="${MOPD_OVERRIDES} \
on_policy_distillation.non_colocated_teachers.default_teacher_cfg.tensor_model_parallel_size=${TEACHER_TP} \
on_policy_distillation.non_colocated_teachers.default_teacher_cfg.context_parallel_size=${TEACHER_CP} \
on_policy_distillation.non_colocated_teachers.default_teacher_cfg.pipeline_model_parallel_size=${TEACHER_PP} \
on_policy_distillation.non_colocated_teachers.default_teacher_cfg.expert_model_parallel_size=${TEACHER_EP} \
on_policy_distillation.non_colocated_teachers.default_teacher_cfg.num_nodes=${NUM_NODES_PER_TEACHER}"

  echo "MOPD: ${NUM_UNIQUE_TEACHERS} teacher pools × ${NUM_NODES_PER_TEACHER} nodes = ${NUM_TEACHER_NODES} teacher nodes"
fi

NUM_ACTOR_NODES=$((NUM_TRAIN_NODES + NUM_GEN_NODES + NUM_TEACHER_NODES))
NUM_RAY_NODES=$((NUM_ACTOR_NODES + NUM_GYM_NODES))
NUM_TOTAL_NODES=$((NUM_RAY_NODES + NUM_EXTERNAL_SERVICE_NODES))

if (( NUM_TRAIN_NODES <= 0 )); then
  echo "ERROR: NUM_TRAIN_NODES must be > 0 (got ${NUM_TRAIN_NODES})" >&2; exit 1
fi
if (( NUM_GEN_NODES <= 0 )); then
  echo "ERROR: NUM_GEN_NODES must be > 0 (got ${NUM_GEN_NODES})" >&2; exit 1
fi
if (( NUM_GYM_NODES < 0 )); then
  echo "ERROR: NUM_GYM_NODES must be >= 0 (got ${NUM_GYM_NODES})" >&2; exit 1
fi

# GB200 NVL72 topology: 18 nodes per NVLink domain, allocate in groups of 16.
# With external judges, Slurm schedules two heterogeneous components, so the
# NeMo RL nodes and the external-service nodes are validated separately.
SEGMENT_SIZE="${SEGMENT_SIZE:-16}"
EXTERNAL_VLLM_SEGMENT_SIZE="${EXTERNAL_VLLM_SEGMENT_SIZE:-4}"
if (( NUM_RAY_NODES < SEGMENT_SIZE )); then
  echo "ERROR: NUM_RAY_NODES=${NUM_RAY_NODES} < SEGMENT_SIZE=${SEGMENT_SIZE}" >&2
  exit 1
fi
if (( NUM_RAY_NODES % SEGMENT_SIZE != 0 )); then
  echo "ERROR: NeMo RL nodes=${NUM_RAY_NODES} is not divisible by SEGMENT_SIZE=${SEGMENT_SIZE}." >&2
  echo "  Training=${NUM_TRAIN_NODES} + Generation=${NUM_GEN_NODES} + Gym=${NUM_GYM_NODES} + Teachers=${NUM_TEACHER_NODES} = ${NUM_RAY_NODES}" >&2
  echo "  Adjust node counts so the total is a multiple of ${SEGMENT_SIZE}." >&2
  exit 1
fi
if (( NUM_EXTERNAL_SERVICE_NODES > 0 )); then
  if (( EXTERNAL_VLLM_SEGMENT_SIZE <= 0 )); then
    echo "ERROR: EXTERNAL_VLLM_SEGMENT_SIZE must be > 0." >&2
    exit 1
  fi
  if (( NUM_EXTERNAL_SERVICE_NODES % EXTERNAL_VLLM_SEGMENT_SIZE != 0 )); then
    echo "ERROR: External service nodes=${NUM_EXTERNAL_SERVICE_NODES} is not divisible by EXTERNAL_VLLM_SEGMENT_SIZE=${EXTERNAL_VLLM_SEGMENT_SIZE}." >&2
    echo "  Registered pools: ${EXTERNAL_VLLM_POOLS}" >&2
    exit 1
  fi
fi

# =============================================================================
# NeMo Skills sandbox (for math_formal_lean, ns_tools, etc.)
# =============================================================================
export SANDBOX_CONTAINER
export SANDBOX_COMMAND="${SANDBOX_COMMAND:-/start-with-nginx.sh}"
export NEMO_SKILLS_SANDBOX_PORT="${NEMO_SKILLS_SANDBOX_PORT:-6000}"

# =============================================================================
# Ray log sync
# =============================================================================
export RAY_LOG_SYNC_FREQUENCY="${RAY_LOG_SYNC_FREQUENCY:-60}"

CODE_ROOT="/opt/nemo-rl"

# =============================================================================
# Persistent cache directories
# =============================================================================
# Lustre holds the warm persistent cache. At job start, SETUP_COMMAND clears
# stale /tmp caches then seeds node-local /tmp from Lustre. JIT writes go to
# /tmp to avoid Lustre metadata contention from parallel compilation.
_vllm_cache_precision="bf16"
CACHE_READ_DIR="${PERSISTENT_CACHE}/cache_read"
CACHE_WRITE_DIR="${PERSISTENT_CACHE}/cache_write"
LUSTRE_VLLM_CACHE="${CACHE_WRITE_DIR}/vllm_compile_cache_${_vllm_cache_precision}"
LUSTRE_FLASHINFER_CUBIN_CACHE="${PERSISTENT_CACHE}/flashinfer_cubins"
FLASHINFER_CUBIN_CACHE="/tmp/nemo_rl_flashinfer_cubins"
FLASHINFER_WS_BASE="${PERSISTENT_CACHE}/flashinfer_workspace"
LUSTRE_INDUCTOR_CACHE="${PERSISTENT_CACHE}/inductor_cache"
LUSTRE_TRITON_CACHE="${PERSISTENT_CACHE}/triton_cache"
NRL_VLLM_LOCAL_CACHE_DIR="/tmp/nemo_rl_vllm_cache"
NRL_VLLM_CACHE_SEED_DIR="/tmp/nemo_rl_vllm_cache_warm"
INDUCTOR_CACHE_DIR="/tmp/nemo_rl_inductor_cache"
TRITON_CACHE_DIR="/tmp/nemo_rl_triton_cache"
CACHE_SYNC_FREQUENCY="${CACHE_SYNC_FREQUENCY:-0}"

export LUSTRE_VLLM_CACHE
export LUSTRE_INDUCTOR_CACHE
export LUSTRE_TRITON_CACHE
export CACHE_READ_DIR
export CACHE_WRITE_DIR
export NRL_VLLM_LOCAL_CACHE_DIR
export INDUCTOR_CACHE_DIR
export TRITON_CACHE_DIR
export CACHE_SYNC_FREQUENCY

mkdir -p "${LUSTRE_FLASHINFER_CUBIN_CACHE}" "${FLASHINFER_WS_BASE}" \
  "${LUSTRE_INDUCTOR_CACHE}" "${LUSTRE_TRITON_CACHE}" \
  "${CACHE_READ_DIR}" "${CACHE_WRITE_DIR}"

# Read path  : cache_read/*.tar.zst   — compute nodes extract tarballs (hundreds of concurrent reads)
# Write path : cache_write/*/        — sidecar rsyncs individual files (one sequential writer)
# Splitting reads (tarball) from writes (directory) avoids Lustre MDT invalidation storms
# and lets rsync accumulate the union of all roles' kernels across jobs.
for _name in inductor_cache triton_cache; do
  _write_dir="${CACHE_WRITE_DIR}/${_name}"
  _old_dir="${PERSISTENT_CACHE}/${_name}"

  # One-time migration: move legacy dir → cache_write/ (instant rename, same FS)
  if ([ ! -d "$_write_dir" ] || [ -z "$(ls -A "$_write_dir" 2>/dev/null)" ]) \
     && [ -d "$_old_dir" ] && [ -n "$(ls -A "$_old_dir" 2>/dev/null)" ]; then
    [ -d "$_write_dir" ] && rmdir "$_write_dir" 2>/dev/null
    mv "$_old_dir" "$_write_dir" 2>/dev/null \
      && echo "[CACHE] Moved legacy ${_name}/ → cache_write/${_name}/" \
      || echo "[CACHE] Failed to move legacy ${_name}/"
  fi
done

# vLLM: migrate the most recent legacy seed dir → cache_write/ (one-time, instant rename)
_vllm_write="${CACHE_WRITE_DIR}/vllm_compile_cache_${_vllm_cache_precision}"
_vllm_read_tar="${CACHE_READ_DIR}/vllm_compile_cache_${_vllm_cache_precision}.tar.zst"

if [ ! -d "$_vllm_write" ] || [ -z "$(ls -A "$_vllm_write" 2>/dev/null)" ]; then
  _best="$(ls -1dt \
      "${PERSISTENT_CACHE}/vllm_compile_cache_${_vllm_cache_precision}" \
      "${PERSISTENT_CACHE}/vllm_compile_cache_${_vllm_cache_precision}_"* \
    2>/dev/null \
    | while IFS= read -r d; do
        [ -d "$d" ] && [ -n "$(ls -A "$d" 2>/dev/null)" ] && echo "$d" && break
      done
  )" || true
  if [ -n "$_best" ]; then
    [ -d "$_vllm_write" ] && rmdir "$_vllm_write" 2>/dev/null || true
    mv "$_best" "$_vllm_write" 2>/dev/null \
      && echo "[CACHE] Moved $(basename "$_best") → cache_write/vllm_compile_cache_${_vllm_cache_precision}/" \
      || echo "[CACHE] Failed to move vLLM cache"
  fi
fi

# Purge redundant legacy vLLM cache directories.
# The old sidecar wrote every vLLM seed as a separate directory on Lustre
# (e.g. vllm_compile_cache_bf16_2058, _3072, ...). With cache_write/ + tarball,
# only cache_write/vllm_compile_cache_{precision}/ matters. All seed copies are
# content-addressed duplicates — safe to remove after migration.
_purge_count=0
for _d in "${PERSISTENT_CACHE}/vllm_compile_cache_${_vllm_cache_precision}" \
          "${PERSISTENT_CACHE}/vllm_compile_cache_${_vllm_cache_precision}_"*; do
  [ -d "$_d" ] || continue
  rm -rf "$_d" 2>/dev/null && (( _purge_count++ )) || true
done
for _d in "${PERSISTENT_CACHE}"/vllm_compile_cache_[0-9]*/; do
  [ -d "$_d" ] || continue
  rm -rf "$_d" 2>/dev/null && (( _purge_count++ )) || true
done
for _d in "${PERSISTENT_CACHE}/vllm_compile_cache" \
          "${PERSISTENT_CACHE}/vllm_compile_cache_warm"; do
  [ -d "$_d" ] || continue
  rm -rf "$_d" 2>/dev/null && (( _purge_count++ )) || true
done
if (( _purge_count > 0 )); then
  echo "[CACHE] Purged ${_purge_count} redundant legacy vLLM cache directories from ${PERSISTENT_CACHE}/"
fi

# =============================================================================
# Code snapshot
# =============================================================================
# Snapshot the git-tracked source tree so the code is frozen at submission time.
# This guarantees we know exactly which code was used for a given experiment.
# Set USE_SNAPSHOT=0 to skip (runs from container built-in or live checkout).
# Interactive mode defaults to the live checkout for fast iteration; batch snapshots.
if [[ "${INTERACTIVE}" == "1" ]]; then
  USE_SNAPSHOT="${USE_SNAPSHOT:-0}"
else
  USE_SNAPSHOT="${USE_SNAPSHOT:-1}"
fi

if [[ "${USE_SNAPSHOT}" == "1" ]]; then
  if [[ ! -f "${PROJECT_ROOT}/tools/code_snapshot.sh" ]]; then
    echo "ERROR: tools/code_snapshot.sh not found at ${PROJECT_ROOT}/tools/code_snapshot.sh" >&2
    echo "  Set USE_SNAPSHOT=0 to run from the live checkout instead." >&2
    exit 1
  fi
  SNAPSHOT_DIR=$(bash "${PROJECT_ROOT}/tools/code_snapshot.sh" "${JOB_NAME}")

  echo "Code snapshot: ${SNAPSHOT_DIR}"
  OVERLAY_SOURCE="${SNAPSHOT_DIR}"
else
  OVERLAY_SOURCE="${PROJECT_ROOT}"
fi

# =============================================================================
# Container mounts
# =============================================================================
# By default, nemo_rl (Python package) and examples/configs (YAML configs) from
# the code snapshot are overlaid into the container. Everything else uses the
# container's built-in code at /opt/nemo-rl.
#
# To overlay additional components (e.g. a local Megatron-LM checkout), pass
# EXTRA_MOUNTS as a comma-separated list of host:container pairs:
#
#   EXTRA_MOUNTS="/path/to/Megatron-LM:/opt/nemo-rl/3rdparty/Megatron-LM-workspace/Megatron-LM" bash examples/nemo_gym/nemotron-3-ultra/ultra_launch.sh
#
# Container paths for reference:
#   /opt/nemo-rl/nemo_rl                                              — Python package
#   /opt/nemo-rl/examples/configs                                     — YAML configs
#   /opt/nemo-rl/3rdparty/Megatron-LM-workspace/Megatron-LM           — Megatron-LM
#   /opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge   — Megatron-Bridge
#   /opt/nemo-rl/3rdparty/Gym-workspace/Gym                           — NeMo-Gym
# =============================================================================
_append_mount() {
  if [[ -z "${MOUNTS}" ]]; then
    MOUNTS="$1"
  else
    MOUNTS="${MOUNTS},$1"
  fi
}

if [[ -d "${OVERLAY_SOURCE}/nemo_rl" ]]; then
  _append_mount "${OVERLAY_SOURCE}/nemo_rl:/opt/nemo-rl/nemo_rl"
  echo "  Mount: nemo_rl → /opt/nemo-rl/nemo_rl"
fi
if [[ -d "${OVERLAY_SOURCE}/examples/configs" ]]; then
  _append_mount "${OVERLAY_SOURCE}/examples/configs:/opt/nemo-rl/examples/configs"
  echo "  Mount: configs → /opt/nemo-rl/examples/configs"
fi
if [[ -d "${OVERLAY_SOURCE}/3rdparty/Gym-workspace/Gym" ]]; then
  _append_mount "${OVERLAY_SOURCE}/3rdparty/Gym-workspace/Gym:/opt/nemo-rl/3rdparty/Gym-workspace/Gym"
  echo "  Mount: Gym → /opt/nemo-rl/3rdparty/Gym-workspace/Gym"
fi

if [[ "${USE_SNAPSHOT}" == "1" ]]; then
  _append_mount "${SNAPSHOT_DIR}:${SNAPSHOT_DIR}"
fi

if [[ -n "${EXTRA_MOUNTS:-}" ]]; then
  _append_mount "${EXTRA_MOUNTS}"
  echo "  Extra mounts: ${EXTRA_MOUNTS}"
fi

export MOUNTS

# =============================================================================
# Resolve ray.sub
# =============================================================================
RAY_SUB="${RAY_SUB:-${PROJECT_ROOT}/ray.sub}"
if [[ ! -f "${RAY_SUB}" ]]; then
  echo "ERROR: ray.sub not found at ${RAY_SUB}" >&2
  exit 1
fi
export RAY_SUB
# External judges wrap ray.sub so the services come up before training starts.
if (( NUM_EXTERNAL_SERVICE_NODES > 0 )); then
  BATCH_SCRIPT="${BATCH_SCRIPT:-${PROJECT_ROOT}/tools/external_gym_vllm/run_in_allocation.sh}"
else
  BATCH_SCRIPT="${BATCH_SCRIPT:-${RAY_SUB}}"
fi
if [[ ! -f "${BATCH_SCRIPT}" ]]; then
  echo "ERROR: batch script not found at ${BATCH_SCRIPT}" >&2
  exit 1
fi

# =============================================================================
# Per-node cache seeding (SETUP_COMMAND)
# =============================================================================
# Triton, Inductor, and FlashInfer cubins compile/download to node-local /tmp to
# avoid Lustre race conditions and file lock contention during concurrent JIT
# compilation. To avoid cold-start penalties, we seed /tmp from a warm Lustre
# cache before Ray starts.
#
# IMPORTANT: Stale /tmp caches from previous jobs can cause hangs (e.g. the
# Triton bundler skipping non-empty temp dirs). We rm -rf /tmp caches first,
# then seed fresh from Lustre.
# =============================================================================
read -r -d '' SETUP_COMMAND <<SETUPEOF || true
command -v zstd >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq zstd; } 2>/dev/null || true
echo "[CACHE SEED] Clearing stale /tmp caches and seeding from Lustre..."
WARM_SEED="${NRL_VLLM_CACHE_SEED_DIR}"
LOCAL_IND="${INDUCTOR_CACHE_DIR}"
LOCAL_TRI="${TRITON_CACHE_DIR}"
CACHE_READ="${CACHE_READ_DIR}"

# vLLM caches are per-instance (VLLM_CACHE_ROOT_{seed}). Clear ALL from prior jobs.
rm -rf /tmp/nemo_rl_vllm_cache /tmp/nemo_rl_vllm_cache_*
rm -rf "\$LOCAL_IND" "\$LOCAL_TRI"
mkdir -p "\$LOCAL_IND" "\$LOCAL_TRI"

_seed_cache() {
  local tarball="\$1" local_dir="\$2" name="\$3"
  if [ -f "\$tarball" ]; then
    tar --zstd -xf "\$tarball" -C "\$local_dir" \
      && echo "[CACHE SEED] \$name: seeded from tarball (\$(du -sh "\$local_dir" 2>/dev/null | cut -f1))" \
      || echo "[CACHE SEED] \$name: tarball extract failed (non-fatal)"
  else
    echo "[CACHE SEED] \$name: no warm cache on Lustre yet"
  fi
}

# Seed vLLM compile cache from cache_read/ tarball (one per precision).
rm -rf "\$WARM_SEED"
_vllm_tar="\$CACHE_READ/vllm_compile_cache_${_vllm_cache_precision}.tar.zst"
if [ -f "\$_vllm_tar" ]; then
  mkdir -p "\$WARM_SEED"
  tar --zstd -xf "\$_vllm_tar" -C "\$WARM_SEED" \
    && echo "[CACHE SEED] vLLM (${_vllm_cache_precision}): seeded from tarball (\$(du -sh "\$WARM_SEED" 2>/dev/null | cut -f1))" \
    || echo "[CACHE SEED] vLLM: tarball extract failed (non-fatal)"
else
  echo "[CACHE SEED] vLLM: no warm cache on Lustre yet"
fi

_seed_cache "\$CACHE_READ/inductor_cache.tar.zst" "\$LOCAL_IND" "Inductor"
_seed_cache "\$CACHE_READ/triton_cache.tar.zst" "\$LOCAL_TRI" "Triton"

echo "[CACHE SEED] Done."
SETUPEOF
export SETUP_COMMAND

# =============================================================================
# Build the training command
# =============================================================================
# Stage-specific hyperparameters (batch sizes, advantage clip, MoE parallelism,
# learning rate, etc.) live in CONFIG_PATH. The launcher only passes the
# per-run overrides: cluster shape, paths, judge endpoints, logging.
# =============================================================================
TRAIN_CMD="cd ${CODE_ROOT} && date ; \
OMP_NUM_THREADS=16 \
RAY_DEDUP_LOGS=1 \
WANDB_INIT_TIMEOUT=300 \
VLLM_CACHE_ROOT=${NRL_VLLM_LOCAL_CACHE_DIR} \
NRL_VLLM_CACHE_SEED_DIR=${NRL_VLLM_CACHE_SEED_DIR} \
DG_JIT_CACHE_DIR=${NRL_VLLM_LOCAL_CACHE_DIR}/deep_gemm \
TORCHINDUCTOR_CACHE_DIR=${INDUCTOR_CACHE_DIR} \
TRITON_CACHE_DIR=${TRITON_CACHE_DIR} \
UV_CACHE_DIR=/tmp/nemo-gym-uv-cache-\${SLURM_JOB_ID:-default} \
UV_LOCK_TIMEOUT=1800 \
RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
UV_HTTP_TIMEOUT=10 \
VLLM_USE_FLASHINFER_MOE_FP8=1 \
VLLM_FLASHINFER_MOE_BACKEND=latency \
NRL_VLLM_ASYNC_TIMEOUT_SECONDS=1800 \
NRL_WG_USE_RAY_REF=1 \
HF_HOME=${HF_HOME:-} \
HF_TOKEN=${HF_TOKEN:-} \
NRL_USE_FASTOKENS=${NRL_USE_FASTOKENS:-1} \
uv run ./examples/nemo_gym/run_grpo_nemo_gym.py \
--config ${CONFIG_PATH} \
policy.model_name=${MODEL_PATH} \
cluster.num_nodes=${NUM_ACTOR_NODES} \
policy.generation.colocated.resources.num_nodes=${NUM_GEN_NODES} \
env.nemo_gym.num_gpu_nodes=${NUM_GYM_NODES} \
checkpointing.checkpoint_dir=${CHECKPOINT_DIR} \
${CHECKPOINTING_SAVE_BY:+checkpointing.checkpoint_must_save_by=${CHECKPOINTING_SAVE_BY}} \
data.train.data_path=${TRAIN_PATH} \
data.validation.data_path=${VAL_PATH} \
${GENRM_OVERRIDE:+${GENRM_OVERRIDE}} \
${NL2BASH_OVERRIDE:+${NL2BASH_OVERRIDE}} \
${SAFETY_JUDGE_MODEL:+env.nemo_gym.safety_judge_model.responses_api_models.local_vllm_model.model=${SAFETY_JUDGE_MODEL}} \
${SIF_DIR:+sif_dir=${SIF_DIR}} \
env.nemo_gym.nemo_gym_log_dir=${LOG_DIR}/nemo_gym \
logger.log_dir=${LOG_DIR} \
logger.wandb_enabled=${WANDB_ENABLED} \
logger.wandb.name=${WANDB_NAME} \
logger.wandb.project=${WANDB_PROJ} \
${NRL_MAX_STEPS:+grpo.max_num_steps=${NRL_MAX_STEPS}} \
${MTP_EXTRA_ARGS} \
${MOPD_OVERRIDES} \
${*}"

export COMMAND="${TRAIN_CMD}"
if (( NUM_EXTERNAL_SERVICE_NODES > 0 )); then
  validate_external_vllm_submission "${COMMAND}" "${NUM_EXTERNAL_SERVICE_NODES}"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "================================================================"
echo "  Nemotron 3 Ultra — ${EXP_NAME} (${NUM_TOTAL_NODES}-node)"
echo "================================================================"
echo "  Job name:    ${JOB_NAME}  (singleton — only one runs at a time)"
echo "  Config:      ${CONFIG_PATH}"
echo "  Nodes:       ${NUM_TOTAL_NODES} total"
if (( NUM_EXTERNAL_SERVICE_NODES > 0 )); then
echo "    Hetgroup 0: ${NUM_RAY_NODES} NeMo RL nodes  (segment=${SEGMENT_SIZE})"
fi
echo "    Training:  ${NUM_TRAIN_NODES}  ($((NUM_TRAIN_NODES * GPUS_PER_NODE)) GPUs)"
echo "    vLLM gen:  ${NUM_GEN_NODES}  ($((NUM_GEN_NODES * GPUS_PER_NODE)) GPUs)"
echo "    Gym:       ${NUM_GYM_NODES}  ($((NUM_GYM_NODES * GPUS_PER_NODE)) GPUs)"
if (( NUM_TEACHER_NODES > 0 )); then
echo "    Teachers:  ${NUM_TEACHER_NODES}  ($((NUM_TEACHER_NODES * GPUS_PER_NODE)) GPUs)"
fi
if (( NUM_EXTERNAL_SERVICE_NODES > 0 )); then
echo "    Hetgroup 1: ${NUM_EXTERNAL_SERVICE_NODES} external-service nodes  (segment=${EXTERNAL_VLLM_SEGMENT_SIZE})"
for _pool in ${EXTERNAL_VLLM_POOLS}; do
  _replicas="${_pool}_REPLICAS"; _tp="${_pool}_TENSOR_PARALLEL_SIZE"; _lb="${_pool}_LB_PORT"
  _label="${_pool}_DISPLAY_NAME"
  echo "      ${!_label}: ${!_replicas} independent TP=${!_tp}, DP=1 servers; LB port=${!_lb}"
done
fi
echo "  Walltime:    ${WALLTIME}"
echo "  Batch script: ${BATCH_SCRIPT}"
echo ""
echo "  Checkpoints: ${CHECKPOINT_DIR}  (stable — auto-resumes across jobs)"
echo "  Run dir:     ${RUN_DIR}"
echo "  Logs:        ${LOG_DIR}"
echo "  Slurm logs:  ${SLURM_LOG_DIR}"
echo "  W&B:         ${WANDB_PROJ} / ${WANDB_NAME} (enabled=${WANDB_ENABLED})"
echo ""
echo "  Model:       ${MODEL_PATH}"
echo "  Train data:  ${TRAIN_PATH}"
echo "  Val data:    ${VAL_PATH}"
echo "  Container:   ${CONTAINER}"
echo "  Sandbox:     ${SANDBOX_CONTAINER}"
if [[ "${USE_SNAPSHOT}" == "1" ]]; then
echo "  Snapshot:    ${SNAPSHOT_DIR}"
fi
echo ""
echo "  Monitor:  squeue -u \$USER -n ${JOB_NAME}"
echo "  Logs:     tail -f ${SLURM_LOG_DIR}/*.out"
echo "  Latest:   ls -la ${RESULTS_DIR}/runs/latest"
echo ""
echo "================================================================"
echo ""

# =============================================================================
# Record code provenance in the run directory
# =============================================================================
{
  echo "timestamp: $(date -Iseconds)"
  echo "branch: $(git -C "${PROJECT_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  echo "commit: $(git -C "${PROJECT_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "dirty: $(git -C "${PROJECT_ROOT}" status --porcelain 2>/dev/null | head -20)"
  echo "snapshot: ${USE_SNAPSHOT}"
  if [[ "${USE_SNAPSHOT}" == "1" ]]; then
    echo "snapshot_dir: ${SNAPSHOT_DIR}"
  fi
  echo "container: ${CONTAINER}"
  echo "config: ${CONFIG_PATH}"
  echo "command: ${TRAIN_CMD}"
} > "${RUN_DIR}/provenance.txt"

# =============================================================================
# Dry-run mode: print everything, don't submit
# =============================================================================
DRY_RUN="${DRY_RUN:-0}"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1 — printing TRAIN_CMD and exiting without submission."
  echo ""
  echo "--- TRAIN_CMD ---"
  echo "${TRAIN_CMD}"
  echo "--- end ---"
  exit 0
fi

# =============================================================================
# Interactive mode: bring up Ray and idle for attachment (no training driver)
# =============================================================================
# With COMMAND empty, ray.sub starts the Ray cluster, writes <jobid>-attach.sh,
# then idles. We save the driver command to <jobid>-run-cmd.sh so you can attach
# and run it by hand, edit it, and re-run without requeueing.
if [[ "${INTERACTIVE}" == "1" ]]; then
  if (( NUM_EXTERNAL_SERVICE_NODES > 0 )); then
    echo "ERROR: INTERACTIVE=1 is not supported with external judge services." >&2
    echo "  Use DRY_RUN=1 to inspect the command or submit the batch job normally." >&2
    exit 1
  fi
  unset COMMAND 2>/dev/null || true   # empty COMMAND -> ray.sub idle/interactive mode
  WALLTIME="${INTERACTIVE_WALLTIME:-${WALLTIME}}"

  echo ""
  echo "================================================================"
  echo "  INTERACTIVE MODE — ${NUM_TOTAL_NODES}-node allocation (walltime ${WALLTIME})"
  echo "  Ray will start and idle until you attach."
  echo "================================================================"

  SBATCH_OUTPUT=$(sbatch \
    --nodes="${NUM_TOTAL_NODES}" \
    --account="${SLURM_ACCOUNT}" \
    --job-name="interactive-${JOB_NAME}" \
    --partition="${SLURM_PARTITION}" \
    --time="${WALLTIME}" \
    --gres=gpu:${GPUS_PER_NODE} \
    --exclusive \
    --mem=0 \
    --segment="${SEGMENT_SIZE}" \
    --output="${SLURM_LOG_DIR}/%j.out" \
    --error="${SLURM_LOG_DIR}/%j.err" \
    ${SLURM_QOS:+--qos="${SLURM_QOS}"} \
    ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES}"} \
    ${SLURM_RESERVATION:+--reservation="${SLURM_RESERVATION}"} \
    --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"60","reason":"interactive","description":"interactive debugging"}}' \
    "${RAY_SUB}")
  echo "${SBATCH_OUTPUT}"
  JOB_ID=$(echo "${SBATCH_OUTPUT}" | grep -oP '\d+$')
  [[ -z "${JOB_ID}" ]] && { echo "ERROR: could not parse job ID from sbatch output." >&2; exit 1; }

  LAUNCH_DIR="$(pwd)"
  ATTACH_SCRIPT="${LAUNCH_DIR}/${JOB_ID}-attach.sh"
  CMD_FILE="${LAUNCH_DIR}/${JOB_ID}-run-cmd.sh"
  cat > "${CMD_FILE}" <<CMDEOF
${TRAIN_CMD}
CMDEOF
  chmod +x "${CMD_FILE}"

  echo ""
  echo "  Driver command saved to:  ${CMD_FILE}"
  echo "  When Ray is up:"
  echo "    bash ${ATTACH_SCRIPT}                          # shell on the head node (Ray already up)"
  echo "    source ${CMD_FILE}                             # run the recipe inside that shell"
  echo "    # or non-interactively: COMMAND=\"\$(cat ${CMD_FILE})\" bash ${ATTACH_SCRIPT}"
  echo "  Edit ${CMD_FILE} and re-source to iterate without requeueing.  Cancel: scancel ${JOB_ID}"

  if [[ "${INTERACTIVE_WAIT}" == "1" ]]; then
    echo ""
    echo "  Waiting for Ray (Ctrl+C to stop waiting; the job keeps running)..."
    prev_state=""
    while [[ ! -f "${ATTACH_SCRIPT}" ]]; do
      state=$(squeue -j "${JOB_ID}" -h -o "%T" 2>/dev/null || true)
      [[ -z "${state}" ]] && { echo "  Job ${JOB_ID} left the queue. Check: sacct -j ${JOB_ID}"; exit 1; }
      [[ "${state}" != "${prev_state}" ]] && { echo "  [$(date +%H:%M:%S)] state: ${state}"; prev_state="${state}"; }
      sleep 15
    done
    echo ""
    echo "  Ray is ready — attach: bash ${ATTACH_SCRIPT}"
  fi
  exit 0
fi

# =============================================================================
# Submit
# =============================================================================
# Always serialise same-name submissions via singleton; optionally chain after
# another job with SLURM_DEPENDENCY (e.g. "afterany:3044848" or "afterok:JOBID").
SLURM_DEPENDENCY="${SLURM_DEPENDENCY:-}"
DEPENDENCY="singleton"
[[ -n "${SLURM_DEPENDENCY}" ]] && DEPENDENCY="singleton,${SLURM_DEPENDENCY}"

if (( NUM_EXTERNAL_SERVICE_NODES > 0 )); then
  SBATCH_OUTPUT=$(sbatch \
    --nodes="${NUM_RAY_NODES}" \
    --account="${SLURM_ACCOUNT}" \
    --job-name="${JOB_NAME}" \
    --partition="${SLURM_PARTITION}" \
    --time="${WALLTIME}" \
    --gres=gpu:${GPUS_PER_NODE} \
    --exclusive \
    --mem=0 \
    --dependency="${DEPENDENCY}" \
    --segment="${SEGMENT_SIZE}" \
    --output="${SLURM_LOG_DIR}/%j.out" \
    --error="${SLURM_LOG_DIR}/%j.err" \
    ${SLURM_QOS:+--qos="${SLURM_QOS}"} \
    ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES}"} \
    ${SLURM_RESERVATION:+--reservation="${SLURM_RESERVATION}"} \
    --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"60","reason":"other","description":"batch training run"}}' \
    : \
    --nodes="${NUM_EXTERNAL_SERVICE_NODES}" \
    --account="${SLURM_ACCOUNT}" \
    --job-name="${JOB_NAME}-services" \
    --partition="${SLURM_PARTITION}" \
    --time="${WALLTIME}" \
    --gres=gpu:${GPUS_PER_NODE} \
    --exclusive \
    --mem=0 \
    --segment="${EXTERNAL_VLLM_SEGMENT_SIZE}" \
    ${SLURM_QOS:+--qos="${SLURM_QOS}"} \
    ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES}"} \
    ${SLURM_RESERVATION:+--reservation="${SLURM_RESERVATION}"} \
    "${BATCH_SCRIPT}")
else
  SBATCH_OUTPUT=$(sbatch \
    --nodes="${NUM_TOTAL_NODES}" \
    --account="${SLURM_ACCOUNT}" \
    --job-name="${JOB_NAME}" \
    --partition="${SLURM_PARTITION}" \
    --time="${WALLTIME}" \
    --gres=gpu:${GPUS_PER_NODE} \
    --exclusive \
    --mem=0 \
    --dependency="${DEPENDENCY}" \
    --segment="${SEGMENT_SIZE}" \
    --output="${SLURM_LOG_DIR}/%j.out" \
    --error="${SLURM_LOG_DIR}/%j.err" \
    ${SLURM_QOS:+--qos="${SLURM_QOS}"} \
    ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES}"} \
    ${SLURM_RESERVATION:+--reservation="${SLURM_RESERVATION}"} \
    --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"60","reason":"other","description":"batch training run"}}' \
    "${BATCH_SCRIPT}")
fi

echo "${SBATCH_OUTPUT}"
JOB_ID=$(echo "${SBATCH_OUTPUT}" | grep -oP '\d+$')

if [[ -n "${JOB_ID}" ]]; then
  echo ""
  echo "  Ray logs:    ${BASE_LOG_DIR}/${JOB_ID}-logs/"
  echo ""
fi
