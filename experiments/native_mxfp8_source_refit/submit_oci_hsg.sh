#!/usr/bin/env bash

set -euo pipefail

ACTION=${ACTION:-render}
MODEL=${MODEL:-qwen30}
FP8_PARAM=${FP8_PARAM:-true}
MAX_STEPS=${MAX_STEPS:-2}
RUN_GROUP=${RUN_GROUP:-$(date +%Y%m%d-%H%M%S)}
WALLTIME=${WALLTIME:-04:00:00}
PARTITION=${PARTITION:-batch}
LOCAL_SCRATCH=${LOCAL_SCRATCH:-/raid/scratch/${USER}}

case "${ACTION}" in
  render|test-only|submit) ;;
  *)
    echo "ACTION must be render, test-only, or submit" >&2
    exit 2
    ;;
esac

case "${MODEL}" in
  qwen30|nano) ;;
  *)
    echo "MODEL must be qwen30 or nano" >&2
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

if ! [[ "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_STEPS must be a positive integer" >&2
  exit 2
fi

case "${MODEL}:${FP8_PARAM}" in
  qwen30:false)
    CONFIG=experiments/native_mxfp8_source_refit/qwen30-fp8param-false.yaml
    NUM_NODES=4
    SEGMENT_SIZE=2
    MODEL_CACHE_PATHS='hub/models--Qwen--Qwen3-30B-A3B'
    ;;
  qwen30:true)
    CONFIG=experiments/native_mxfp8_source_refit/qwen30-fp8param-true.yaml
    NUM_NODES=4
    SEGMENT_SIZE=2
    MODEL_CACHE_PATHS='hub/models--Qwen--Qwen3-30B-A3B'
    ;;
  nano:false)
    CONFIG=experiments/native_mxfp8_source_refit/nano-fp8param-false.yaml
    NUM_NODES=8
    SEGMENT_SIZE=4
    MODEL_CACHE_PATHS='hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16 hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16'
    ;;
  nano:true)
    CONFIG=experiments/native_mxfp8_source_refit/nano-fp8param-true.yaml
    NUM_NODES=8
    SEGMENT_SIZE=4
    MODEL_CACHE_PATHS='hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16 hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16'
    ;;
esac

SOURCE_SHA=unknown
if [[ -n "${REPO:-}" ]] && git -C "${REPO}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  SOURCE_SHA=$(git -C "${REPO}" rev-parse HEAD)
fi

if [[ "${ACTION}" == render ]]; then
  printf 'model=%s\nfp8_param=%s\nconfig=%s\nnodes=%s\nsegment_size=%s\nsteps=%s\nsource_sha=%s\n' \
    "${MODEL}" "${FP8_PARAM}" "${CONFIG}" "${NUM_NODES}" "${SEGMENT_SIZE}" "${MAX_STEPS}" "${SOURCE_SHA}"
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

RUN_NAME="native-mxfp8-${MODEL}-fp8param-${FP8_PARAM}-${RUN_GROUP}"
RUN_ROOT="${RESULT_ROOT}/${RUN_NAME}"
if [[ "${ACTION}" == submit ]]; then
  mkdir -p "${RUN_ROOT}/logs"
fi

NATIVE_OVERRIDES=()
if [[ "${FP8_PARAM}" == true ]]; then
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

COMMAND=$(cat <<EOF
set -euo pipefail
cd ${REPO}
export HOME=/root
export HF_HOME_SOURCE=${HF_HOME}
export HF_HOME=${LOCAL_SCRATCH}/hf-cache/${MODEL}
export HF_DATASETS_CACHE=\${HF_HOME}/datasets
export HUGGINGFACE_HUB_CACHE=\${HF_HOME}/hub
export NEMO_RL_VENV_DIR=${LOCAL_SCRATCH}/nemo-rl-worker-cache/${SOURCE_SHA}
export VLLM_CACHE_ROOT=${LOCAL_SCRATCH}/vllm-cache/${SOURCE_SHA}/fp8param-${FP8_PARAM}
export TORCHINDUCTOR_CACHE_DIR=${LOCAL_SCRATCH}/inductor-cache/${SOURCE_SHA}/fp8param-${FP8_PARAM}
export TRITON_CACHE_DIR=${LOCAL_SCRATCH}/triton-cache/${SOURCE_SHA}/fp8param-${FP8_PARAM}
export UV_CACHE_DIR=${LOCAL_SCRATCH}/uv-cache
export UV_PYTHON_INSTALL_DIR=${LOCAL_SCRATCH}/uv-python
export UV_LOCK_TIMEOUT=7200
export PYTHONPATH=${REPO}
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
  "${LOCAL_SCRATCH}/vllm-cache/${SOURCE_SHA}/fp8param-${FP8_PARAM}" \\
  "${LOCAL_SCRATCH}/inductor-cache/${SOURCE_SHA}/fp8param-${FP8_PARAM}" \\
  "${LOCAL_SCRATCH}/triton-cache/${SOURCE_SHA}/fp8param-${FP8_PARAM}" \\
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
  --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"120","reason":"model_loading","description":"native MXFP8 source refit"}}'
)

printf 'repo=%s\nsha=%s\nconfig=%s\nresult=%s\n' "${REPO}" "${SOURCE_SHA}" "${CONFIG}" "${RUN_ROOT}"
exec sbatch "${SBATCH_ACTION[@]}" "${SBATCH_ARGS[@]}" "${REPO}/ray.sub"
