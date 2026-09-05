#!/usr/bin/env bash

set -euo pipefail

ACTION=${ACTION:-render}
MODEL=${MODEL:-qwen30}
PRECISION_MODE=${PRECISION_MODE:-mxfp8}
FP8_PARAM=${FP8_PARAM:-true}
MAX_STEPS=${MAX_STEPS:-2}
NATIVE_REFIT_AUDIT="${NATIVE_REFIT_AUDIT:-0}"
PROFILE_MODE=${PROFILE_MODE:-none}
NTRACE_INSTALL_TARGET=${NTRACE_INSTALL_TARGET:-/opt/ntrace-runtime}
NTRACE_RANKS=${NTRACE_RANKS:-0}
NTRACE_CAPTURE_ITER=${NTRACE_CAPTURE_ITER:-2}
NTRACE_NUM_ITERS=${NTRACE_NUM_ITERS:-3}
RUN_GROUP=${RUN_GROUP:-$(date +%Y%m%d-%H%M%S)}
WALLTIME=${WALLTIME:-04:00:00}
PARTITION=${PARTITION:-batch}
LOCAL_SCRATCH=${LOCAL_SCRATCH:-/raid/scratch/${USER}}
SLURM_HELPER_CANDIDATE_DIRS=${SLURM_HELPER_CANDIDATE_DIRS:-/usr/local/bin:/usr/bin:/bin}
SLURM_HELPER_RESOLVER_CANDIDATES=${SLURM_HELPER_RESOLVER_CANDIDATES:-/usr/bin/readlink:/bin/readlink:/usr/bin/realpath:/bin/realpath}

case "${ACTION}" in
  render|test-only|submit) ;;
  *)
    echo "ACTION must be render, test-only, or submit" >&2
    exit 2
    ;;
esac

case "${MODEL}" in
  qwen30|qwen235|nano) ;;
  *)
    echo "MODEL must be qwen30, qwen235, or nano" >&2
    exit 2
    ;;
esac

case "${PRECISION_MODE}" in
  bf16|mxfp8) ;;
  *)
    echo "PRECISION_MODE must be bf16 or mxfp8" >&2
    exit 2
    ;;
esac

case "${FP8_PARAM}" in
  true|false) ;;
  *)
    echo "FP8_PARAM must be true or false" >&2
    exit 2
    ;;
esac

case "${NATIVE_REFIT_AUDIT}" in
  0|1) ;;
  *)
    echo "NATIVE_REFIT_AUDIT must be 0 or 1" >&2
    exit 2
    ;;
esac

case "${PROFILE_MODE}" in
  none|ntrace) ;;
  *)
    echo "PROFILE_MODE must be none or ntrace" >&2
    exit 2
    ;;
esac

if [[ "${NATIVE_REFIT_AUDIT}" == 1 ]] \
  && [[ "${MODEL}:${PRECISION_MODE}:${FP8_PARAM}" != qwen30:mxfp8:true ]]; then
  echo "runtime audit requires MODEL=qwen30 PRECISION_MODE=mxfp8 FP8_PARAM=true" >&2
  exit 2
fi

if ! [[ "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_STEPS must be a positive integer" >&2
  exit 2
fi

for value_name in NTRACE_CAPTURE_ITER NTRACE_NUM_ITERS; do
  value=${!value_name}
  if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
    echo "${value_name} must be a non-negative integer" >&2
    exit 2
  fi
done
if [[ "${PROFILE_MODE}" == ntrace ]]; then
  if (( NTRACE_NUM_ITERS < 1 )); then
    echo "NTRACE_NUM_ITERS must be positive" >&2
    exit 2
  fi
  if (( MAX_STEPS < NTRACE_CAPTURE_ITER + NTRACE_NUM_ITERS )); then
    echo "MAX_STEPS must complete every requested ntrace iteration" >&2
    exit 2
  fi
fi

case "${MODEL}:${PRECISION_MODE}:${FP8_PARAM}" in
  qwen30:bf16:false)
    CONFIG=experiments/native_mxfp8_source_refit/qwen30-bf16.yaml
    NUM_NODES=4
    SEGMENT_SIZE=2
    MODEL_CACHE_PATHS='hub/models--Qwen--Qwen3-30B-A3B'
    ;;
  qwen30:mxfp8:false)
    CONFIG=experiments/native_mxfp8_source_refit/qwen30-fp8param-false.yaml
    NUM_NODES=4
    SEGMENT_SIZE=2
    MODEL_CACHE_PATHS='hub/models--Qwen--Qwen3-30B-A3B'
    ;;
  qwen30:mxfp8:true)
    CONFIG=experiments/native_mxfp8_source_refit/qwen30-fp8param-true.yaml
    NUM_NODES=4
    SEGMENT_SIZE=2
    MODEL_CACHE_PATHS='hub/models--Qwen--Qwen3-30B-A3B'
    ;;
  qwen235:mxfp8:true)
    CONFIG=experiments/native_mxfp8_source_refit/qwen235-fp8param-true.yaml
    NUM_NODES=32
    SEGMENT_SIZE=16
    MODEL_CACHE_PATHS='hub/models--Qwen--Qwen3-235B-A22B'
    ;;
  nano:mxfp8:false)
    CONFIG=experiments/native_mxfp8_source_refit/nano-fp8param-false.yaml
    NUM_NODES=8
    SEGMENT_SIZE=4
    MODEL_CACHE_PATHS='hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16 hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16'
    ;;
  nano:mxfp8:true)
    CONFIG=experiments/native_mxfp8_source_refit/nano-fp8param-true.yaml
    NUM_NODES=8
    SEGMENT_SIZE=4
    MODEL_CACHE_PATHS='hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16 hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16'
    ;;
  nano:bf16:false)
    CONFIG=experiments/native_mxfp8_source_refit/nano-bf16-train-mxfp8-rollout.yaml
    NUM_NODES=8
    SEGMENT_SIZE=4
    MODEL_CACHE_PATHS='hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16 hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16'
    ;;
  *)
    echo "Unsupported MODEL/PRECISION_MODE/FP8_PARAM combination: ${MODEL}/${PRECISION_MODE}/${FP8_PARAM}" >&2
    exit 2
    ;;
esac

SOURCE_SHA=unknown
if [[ -n "${REPO:-}" ]] && git -C "${REPO}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  SOURCE_SHA=$(git -C "${REPO}" rev-parse HEAD)
fi

if [[ "${ACTION}" == render ]]; then
  printf 'model=%s\nprecision_mode=%s\nfp8_param=%s\nnative_refit_audit=%s\nprofile_mode=%s\nntrace_ranks=%s\nntrace_capture_iter=%s\nntrace_num_iters=%s\nconfig=%s\nnodes=%s\nsegment_size=%s\nsteps=%s\nsource_sha=%s\n' \
    "${MODEL}" "${PRECISION_MODE}" "${FP8_PARAM}" "${NATIVE_REFIT_AUDIT}" "${PROFILE_MODE}" "${NTRACE_RANKS}" "${NTRACE_CAPTURE_ITER}" "${NTRACE_NUM_ITERS}" "${CONFIG}" "${NUM_NODES}" "${SEGMENT_SIZE}" "${MAX_STEPS}" "${SOURCE_SHA}"
  exit 0
fi

require_prefix() {
  local path=$1 prefix=$2 label=$3
  case "${path}" in
    "${prefix}"/*) ;;
    *)
      echo "${label} must be under ${prefix}: ${path}" >&2
      exit 2
      ;;
  esac
}

: "${REPO:?Set REPO to the branch checkout under /home}"
: "${CONTAINER:?Set CONTAINER to an immutable NeMo-RL image under /lustre}"
: "${HF_HOME:?Set HF_HOME to the immutable Hugging Face cache under /lustre}"
: "${WANDB_HOME:?Set WANDB_HOME to a directory containing .netrc under /home}"
: "${RESULT_ROOT:?Set RESULT_ROOT to durable output under /lustre}"
: "${SLURM_ACCOUNT:?Set SLURM_ACCOUNT}"

require_prefix "${REPO}" /home REPO
require_prefix "${CONTAINER}" /lustre CONTAINER
require_prefix "${HF_HOME}" /lustre HF_HOME
require_prefix "${WANDB_HOME}" /home WANDB_HOME
require_prefix "${RESULT_ROOT}" /lustre RESULT_ROOT
require_prefix "${LOCAL_SCRATCH}" /raid/scratch LOCAL_SCRATCH

resolve_helper_path() {
  local helper=$1
  local helper_path candidate_dirs=()
  helper_path=$(command -v "${helper}" 2>/dev/null || true)
  if [[ -z "${helper_path}" ]]; then
    local candidate_dir
    IFS=: read -r -a candidate_dirs <<< "${SLURM_HELPER_CANDIDATE_DIRS}"
    for candidate_dir in "${candidate_dirs[@]}"; do
      [[ -n "${candidate_dir}" ]] || continue
      if [[ -x "${candidate_dir}/${helper}" ]]; then
        helper_path="${candidate_dir}/${helper}"
        break
      fi
    done
  fi
  if [[ -z "${helper_path}" ]]; then
    echo "Required Slurm helper ${helper} not found on PATH or SLURM_HELPER_CANDIDATE_DIRS" >&2
    exit 2
  fi
  if [[ "${helper_path}" != /* ]]; then
    echo "Required Slurm helper ${helper} resolved to a non-absolute path: ${helper_path}" >&2
    exit 2
  fi

  local resolved_path
  resolved_path=$(resolve_real_path "${helper_path}")
  if [[ ! -x "${resolved_path}" ]]; then
    echo "Required Slurm helper ${helper} is not executable: ${resolved_path}" >&2
    exit 2
  fi
  local helper_dir=${resolved_path%/*}
  if [[ -z "${helper_dir}" || "${helper_dir}" == "${resolved_path}" ]]; then
    helper_dir=/
  fi
  printf '%s\n' "${helper_dir}"
}

resolve_real_path() {
  local path=$1
  local resolver resolver_candidates=() resolved_path
  IFS=: read -r -a resolver_candidates <<< "${SLURM_HELPER_RESOLVER_CANDIDATES}"
  for resolver in "${resolver_candidates[@]}"; do
    [[ -n "${resolver}" && -x "${resolver}" ]] || continue
    case "${resolver}" in
      */readlink)
        if resolved_path=$("${resolver}" -f "${path}" 2>/dev/null) && [[ -n "${resolved_path}" ]]; then
          printf '%s\n' "${resolved_path}"
          return
        fi
        ;;
      */realpath)
        if resolved_path=$("${resolver}" "${path}" 2>/dev/null) && [[ -n "${resolved_path}" ]]; then
          printf '%s\n' "${resolved_path}"
          return
        fi
        ;;
    esac
  done
  printf '%s\n' "${path}"
}

resolve_slurm_helper_path() {
  local helper helper_dir joined=
  for helper in sinfo scontrol srun; do
    helper_dir=$(resolve_helper_path "${helper}")
    case ":${joined}:" in
      *":${helper_dir}:"*) ;;
      *) joined="${joined:+${joined}:}${helper_dir}" ;;
    esac
  done
  printf '%s\n' "${joined}"
}

SLURM_HELPER_PATH=$(resolve_slurm_helper_path)
export SLURM_HELPER_PATH

for path in "${REPO}/${CONFIG}" "${REPO}/ray.sub" "${CONTAINER}" "${HF_HOME}" "${WANDB_HOME}/.netrc"; do
  test -e "${path}"
done

if [[ "${ACTION}" == submit ]]; then
  git -C "${REPO}" pull --ff-only
  git -C "${REPO}" submodule update --init --recursive --checkout
  test -z "$(git -C "${REPO}" status --porcelain --untracked-files=no --ignore-submodules=all)"
  if git -C "${REPO}" submodule status --recursive | grep -q '^[+-U]'; then
    echo "All submodules must be initialized at pinned revisions" >&2
    exit 2
  fi
fi
SOURCE_SHA=$(git -C "${REPO}" rev-parse HEAD)
MEGATRON_CHECKPOINT_ROOT=${MEGATRON_CHECKPOINT_ROOT:-${RESULT_ROOT}/pretrained-checkpoints/${SOURCE_SHA}}
DATASET_ROOT=${DATASET_ROOT:-${RESULT_ROOT}/datasets/${SOURCE_SHA}}
require_prefix "${MEGATRON_CHECKPOINT_ROOT}" /lustre MEGATRON_CHECKPOINT_ROOT
require_prefix "${DATASET_ROOT}" /lustre DATASET_ROOT

CACHE_ARM="${PRECISION_MODE}-fp8param-${FP8_PARAM}"
RUN_NAME="native-mxfp8-${MODEL}-${CACHE_ARM}-${RUN_GROUP}"
RUN_ROOT="${RESULT_ROOT}/${RUN_NAME}"
if [[ "${ACTION}" == submit ]]; then
  mkdir -p "${RUN_ROOT}/logs"
  mkdir -p "${MEGATRON_CHECKPOINT_ROOT}/${MODEL}"
  mkdir -p "${DATASET_ROOT}"
fi

NTRACE_COMMAND="unset NRL_POLICY_PROFILER_CLASS"
PYTHONPATH_VALUE=${REPO}
if [[ "${PROFILE_MODE}" == ntrace ]]; then
  NTRACE_OUTPUT_DIR="${RUN_ROOT}/ntrace/policy"
  PYTHONPATH_VALUE=${NTRACE_INSTALL_TARGET}:${REPO}
  NTRACE_COMMAND=$(cat <<EOF
export NRL_POLICY_PROFILER_CLASS=ntrace.NemoRLTraceController
export NTRACE_CUPTI_BACKEND=cpp
export NTRACE_RANKS=${NTRACE_RANKS}
export NTRACE_OUTPUT_DIR=${NTRACE_OUTPUT_DIR}
export NTRACE_CAPTURE_ITER=${NTRACE_CAPTURE_ITER}
export NTRACE_NUM_ITERS=${NTRACE_NUM_ITERS}
unset NTRACE_MAX_STACK_DEPTH
export NTRACE_INCLUDE_STACK_TRACES=1
export NTRACE_STACK_CAPTURE_SCOPE=all
export NTRACE_STACK_TIMING_SAMPLE_INTERVAL=32
export NTRACE_INCLUDE_NVTX_RANGES=1
export NTRACE_ENABLE_MEGATRON_NVTX=1
export NTRACE_INCLUDE_MEMOPS=0
export NTRACE_SAVE_CPU_NVTX=0
export NTRACE_GRAPH_CAPTURE=iteration
export NTRACE_RECORD_WATERMARK_MB=0
export NTRACE_CUPTI_FINALIZE_ON_PAUSE=1
export NTRACE_MEMORY_DIAGNOSTICS=0
mkdir -p ${NTRACE_OUTPUT_DIR}
/opt/nemo_rl_venv/bin/python -c 'import ntrace, pyarrow; from ntrace.backends import get_backend, selected_backend_name; assert selected_backend_name() == "cpp"; get_backend()'
EOF
)
fi

NATIVE_OVERRIDES=()
if [[ "${PRECISION_MODE}" == mxfp8 && "${FP8_PARAM}" == true ]]; then
  NATIVE_OVERRIDES=(
    policy.megatron_cfg.fp8_cfg.enabled=true
    policy.megatron_cfg.fp8_cfg.fp8=e4m3
    policy.megatron_cfg.fp8_cfg.fp8_recipe=mxfp8
    policy.megatron_cfg.fp8_cfg.fp8_param=true
    policy.megatron_cfg.distributed_data_parallel_config.overlap_param_gather=true
    policy.megatron_cfg.distributed_data_parallel_config.overlap_grad_reduce=true
    policy.generation.refit_transport=nccl_reshard
    policy.generation.vllm_cfg.precision=fp8
    policy.generation.vllm_cfg.is_mx=true
    policy.generation.vllm_kwargs.moe_backend=flashinfer_trtllm
  )
fi

NATIVE_AUDIT_COMMAND="unset NRL_NATIVE_MXFP8_REFIT_AUDIT"
if [[ "${NATIVE_REFIT_AUDIT}" == 1 ]]; then
  NATIVE_AUDIT_COMMAND="export NRL_NATIVE_MXFP8_REFIT_AUDIT=require-second-change"
fi

COMMAND=$(cat <<EOF
set -euo pipefail
cd ${REPO}
export PYTHONPATH=${PYTHONPATH_VALUE}
${NATIVE_AUDIT_COMMAND}
${NTRACE_COMMAND}
export HOME=/root
export HF_HOME_SOURCE=${HF_HOME}
export HF_HOME=${LOCAL_SCRATCH}/hf-cache/${MODEL}
export HF_DATASETS_CACHE=${DATASET_ROOT}
export HUGGINGFACE_HUB_CACHE=\${HF_HOME}/hub
export NRL_MEGATRON_CHECKPOINT_DIR=${MEGATRON_CHECKPOINT_ROOT}/${MODEL}
export NEMO_RL_VENV_DIR=${LOCAL_SCRATCH}/nemo-rl-worker-cache/${SOURCE_SHA}
export VLLM_CACHE_ROOT=${LOCAL_SCRATCH}/vllm-cache/${SOURCE_SHA}/${CACHE_ARM}
export TORCHINDUCTOR_CACHE_DIR=${LOCAL_SCRATCH}/inductor-cache/${SOURCE_SHA}/${CACHE_ARM}
export TRITON_CACHE_DIR=${LOCAL_SCRATCH}/triton-cache/${SOURCE_SHA}/${CACHE_ARM}
export UV_CACHE_DIR=${LOCAL_SCRATCH}/uv-cache
export UV_PYTHON_INSTALL_DIR=${LOCAL_SCRATCH}/uv-python
export UV_LOCK_TIMEOUT=7200
unset UV_PROJECT_ENVIRONMENT WANDB_API_KEY
mkdir -p "\${HF_HOME}"
/opt/nemo_rl_venv/bin/python examples/run_grpo.py \\
  --config ${CONFIG} \\
  grpo.max_num_steps=${MAX_STEPS} \\
  grpo.val_at_start=false \\
  ++grpo.val_at_end=false \\
  checkpointing.enabled=false \\
  policy.generation.vllm_cfg.use_tqdm=false \\
  logger.log_dir=${RUN_ROOT}/logs \\
  logger.wandb_enabled=true \\
  logger.wandb.project=nemo-rl-mxfp8-training \\
  logger.wandb.name=${RUN_NAME} \\
  logger.tensorboard_enabled=true \\
  logger.monitor_gpus=true \\
  ${NATIVE_OVERRIDES[*]}
EOF
)

SETUP_COMMAND=$(cat <<EOF
set -euo pipefail
LOCAL_HF_HOME=${LOCAL_SCRATCH}/hf-cache/${MODEL}
mkdir -p "${LOCAL_SCRATCH}/nemo-rl-worker-cache/${SOURCE_SHA}" \\
  "${LOCAL_SCRATCH}/vllm-cache/${SOURCE_SHA}/${CACHE_ARM}" \\
  "${LOCAL_SCRATCH}/inductor-cache/${SOURCE_SHA}/${CACHE_ARM}" \\
  "${LOCAL_SCRATCH}/triton-cache/${SOURCE_SHA}/${CACHE_ARM}" \\
  "${LOCAL_SCRATCH}/uv-cache" \\
  "${LOCAL_SCRATCH}/uv-python" \\
  "${LOCAL_SCRATCH}/ray" \\
  "\${LOCAL_HF_HOME}"
for relative_path in ${MODEL_CACHE_PATHS}; do
  source_path=${HF_HOME}/\${relative_path}
  destination_path=\${LOCAL_HF_HOME}/\${relative_path}
  test -d "\${source_path}"
  mkdir -p "\${destination_path}"
  rsync -a --ignore-existing "\${source_path}/" "\${destination_path}/"
done
EOF
)

export CONTAINER
export MOUNTS="/lustre:/lustre,/home:/home,/raid/scratch:/raid/scratch,${WANDB_HOME}/.netrc:/root/.netrc"
export CONTAINER_REMAP_ROOT=1
export RAY_TMPDIR_ROOT="${LOCAL_SCRATCH}/ray"
export COMMAND
export SETUP_COMMAND
export GPUS_PER_NODE=4
export CPUS_PER_WORKER=${CPUS_PER_WORKER:-144}
export BASE_LOG_DIR="${RUN_ROOT}"

SBATCH_ACTION=()
if [[ "${ACTION}" == test-only ]]; then
  SBATCH_ACTION=(--test-only)
fi

SBATCH_ARGS=(
  --nodes="${NUM_NODES}"
  --gres=gpu:4
  --exclusive
  --account="${SLURM_ACCOUNT}"
  --partition="${PARTITION}"
  --time="${WALLTIME}"
  --segment="${SEGMENT_SIZE}"
  --job-name="${SLURM_ACCOUNT}.${RUN_NAME}"
  --output="${RUN_ROOT}/slurm-%j.out"
  --export="ALL,SLURM_HELPER_PATH=${SLURM_HELPER_PATH}"
  --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"120","reason":"model_loading","description":"native MXFP8 source refit"}}'
)

printf 'repo=%s\nsha=%s\nconfig=%s\nresult=%s\n' "${REPO}" "${SOURCE_SHA}" "${CONFIG}" "${RUN_ROOT}"
exec sbatch "${SBATCH_ACTION[@]}" "${SBATCH_ARGS[@]}" "${REPO}/ray.sub"
