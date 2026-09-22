#!/bin/bash
set -euo pipefail

# Submit one arm of the matched Dynamic-CP versus static-CP comparison.
#
# Usage:
#   tools/launch_dynamic_cp_comparison.sh qwen30 dynamic
#   tools/launch_dynamic_cp_comparison.sh qwen30 static
#
# "static" means Dynamic CP is disabled, but CP is still greater than one. It
# is deliberately not a no-CP (CP1) baseline.

usage() {
  cat <<'EOF'
Usage: launch_dynamic_cp_comparison.sh MODEL MODE

MODEL: qwen30 | qwen32 | nano
MODE:  dynamic | static

Useful environment variables:
  COMPARISON_NAME          Common name for all six runs (required for pairing)
  NRL_MAX_STEPS            Training steps; default: 3
  DYNAMIC_CP_RESULTS_ROOT  Shared output root on /lustre
  CONTAINER                NeMo-RL squashfs/image
  SLURM_ACCOUNT            Default: coreai_dlalgo_nemorl
  SLURM_PARTITION          Default: batch
  WALLTIME                 Default: 1:59:00
  ENABLE_WANDB             0 (default) or 1
  DRY_RUN                  1 prints the submission without calling sbatch
EOF
}

if [[ $# -ne 2 ]]; then
  usage >&2
  exit 2
fi

MODEL="$1"
MODE="$2"
case "${MODE}" in
  dynamic|static) ;;
  *)
    echo "ERROR: MODE must be 'dynamic' or 'static' (got '${MODE}')." >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd -L -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -L)"
NEMO_RL_ROOT="${NEMO_RL_ROOT:-$(cd -L -- "${SCRIPT_DIR}/.." && pwd -L)}"

case "${MODEL}" in
  qwen30)
    NODES=4
    SEGMENT_SIZE=2
    GBS=512
    MAX_SEQUENCE_LENGTH=8192
    STATIC_CP=2
    DYNAMIC_CP_RANGE="1-2"
    DYNAMIC_RECIPE="examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-async-1off-megatron-dynamiccp-10step.yaml"
    STATIC_RECIPE="examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-async-1off-megatron-staticcp-10step.yaml"
    ;;
  qwen32)
    NODES=4
    SEGMENT_SIZE=2
    GBS=512
    MAX_SEQUENCE_LENGTH=16384
    STATIC_CP=4
    DYNAMIC_CP_RANGE="1-4"
    DYNAMIC_RECIPE="examples/configs/recipes/llm/performance/grpo-qwen3-32b-4n4g-async-1off-megatron-dynamiccp-10step.yaml"
    STATIC_RECIPE="examples/configs/recipes/llm/performance/grpo-qwen3-32b-4n4g-async-1off-megatron-staticcp-10step.yaml"
    ;;
  nano)
    NODES=8
    # The Nano recipe does not set cluster.segment_size, so do not claim an
    # allocation-side segment that the runtime config does not also express.
    SEGMENT_SIZE=""
    GBS=64
    MAX_SEQUENCE_LENGTH=8192
    STATIC_CP=4
    # Configured min=1; TP2*CP must contain EP8, so the effective minimum is 4.
    DYNAMIC_CP_RANGE="4-16 (effective; configured 1-16)"
    DYNAMIC_RECIPE="examples/configs/recipes/llm/performance/grpo-nemotron3-nano-30ba3b-8n4g-megatron-dynamiccp-quick.yaml"
    STATIC_RECIPE="examples/configs/recipes/llm/performance/grpo-nemotron3-nano-30ba3b-8n4g-megatron-staticcp-quick.yaml"
    ;;
  *)
    echo "ERROR: MODEL must be qwen30, qwen32, or nano (got '${MODEL}')." >&2
    exit 2
    ;;
esac

if [[ "${MODE}" == "dynamic" ]]; then
  RECIPE="${DYNAMIC_RECIPE}"
  CP_DESCRIPTION="Dynamic CP ${DYNAMIC_CP_RANGE}"
else
  RECIPE="${STATIC_RECIPE}"
  CP_DESCRIPTION="static CP${STATIC_CP} (Dynamic CP disabled)"
fi

if [[ ! -f "${NEMO_RL_ROOT}/${RECIPE}" ]]; then
  echo "ERROR: recipe does not exist: ${NEMO_RL_ROOT}/${RECIPE}" >&2
  echo "The comparison recipes are local to the Dynamic-CP checkout." >&2
  exit 1
fi

CONTAINER="${CONTAINER:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_ci/nemo_rl_ci/sqsh_files/rl.nightly.sqsh}"
if [[ "${CONTAINER}" == /* && ! -e "${CONTAINER}" ]]; then
  echo "ERROR: container does not exist: ${CONTAINER}" >&2
  echo "Set CONTAINER to a NeMo-RL image available on the compute nodes." >&2
  exit 1
fi

if [[ -z "${COMPARISON_NAME:-}" ]]; then
  echo "ERROR: set COMPARISON_NAME once and reuse it for all six runs." >&2
  echo 'Example: export COMPARISON_NAME="cp-smoke-$(date +%Y%m%d-%H%M%S)"' >&2
  exit 2
fi
NRL_MAX_STEPS="${NRL_MAX_STEPS:-3}"
if ! [[ "${NRL_MAX_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: NRL_MAX_STEPS must be a positive integer." >&2
  exit 2
fi

DYNAMIC_CP_RESULTS_ROOT="${DYNAMIC_CP_RESULTS_ROOT:-/lustre/fsw/portfolios/coreai/users/${USER}/projects/nemo-rl-workspace/dynamic-cp-comparisons}"
RUN_DIR="${DYNAMIC_CP_RESULTS_ROOT}/${COMPARISON_NAME}/${MODEL}/${MODE}"
METRICS_DIR="${RUN_DIR}/metrics"
BASE_LOG_DIR="${RUN_DIR}/slurm"

ENABLE_WANDB="${ENABLE_WANDB:-0}"
case "${ENABLE_WANDB}" in
  0) WANDB_ENABLED=false ;;
  1)
    if [[ -z "${WANDB_API_KEY:-}" ]]; then
      echo "ERROR: ENABLE_WANDB=1 requires WANDB_API_KEY." >&2
      exit 2
    fi
    WANDB_ENABLED=true
    ;;
  *)
    echo "ERROR: ENABLE_WANDB must be 0 or 1." >&2
    exit 2
    ;;
esac

if [[ "${DRY_RUN:-0}" != "1" ]] && [[ -d "${RUN_DIR}" ]] && \
   [[ -n "$(find "${RUN_DIR}" -mindepth 1 -print -quit)" ]]; then
  echo "ERROR: run directory is not empty: ${RUN_DIR}" >&2
  echo "Use a new COMPARISON_NAME so old and new measurements are not mixed." >&2
  exit 1
fi
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  mkdir -p "${RUN_DIR}" "${BASE_LOG_DIR}"
fi

COMMAND_PARTS=(
  uv run python examples/run_grpo.py
  --config "${RECIPE}"
  "grpo.max_num_steps=${NRL_MAX_STEPS}"
  checkpointing.enabled=false
  "logger.log_dir=${METRICS_DIR}"
  "logger.wandb_enabled=${WANDB_ENABLED}"
  logger.tensorboard_enabled=true
  logger.wandb.project=nemo-rl-cp-comparison
  "logger.wandb.name=${COMPARISON_NAME}-${MODEL}-${MODE}"
)
printf -v COMMAND '%q ' "${COMMAND_PARTS[@]}"

REQUIRED_MOUNTS="/lustre:/lustre,${NEMO_RL_ROOT}:${NEMO_RL_ROOT}"
MOUNTS="${MOUNTS:-${REQUIRED_MOUNTS}}"
if [[ -n "${EXTRA_MOUNTS:-}" ]]; then
  MOUNTS="${MOUNTS},${EXTRA_MOUNTS}"
fi

export CONTAINER MOUNTS COMMAND BASE_LOG_DIR
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export RAY_LOG_SYNC_FREQUENCY="${RAY_LOG_SYNC_FREQUENCY:-30}"
export HF_HOME="${HF_HOME:-/lustre/fsw/portfolios/coreai/users/${USER}/hf_home}"

SLURM_ACCOUNT="${SLURM_ACCOUNT:-coreai_dlalgo_nemorl}"
SLURM_PARTITION="${SLURM_PARTITION:-batch}"
WALLTIME="${WALLTIME:-1:59:00}"
JOB_NAME="dcp-${MODEL}-${MODE}"
SBATCH_ARGS=(
  --nodes="${NODES}"
  --account="${SLURM_ACCOUNT}"
  --partition="${SLURM_PARTITION}"
  --time="${WALLTIME}"
  --gres="gpu:${GPUS_PER_NODE}"
  --exclusive
  --mem=0
  --job-name="${JOB_NAME}"
  --output="${RUN_DIR}/slurm-%j.out"
)
if [[ -n "${SEGMENT_SIZE}" ]]; then
  SBATCH_ARGS+=(--segment="${SEGMENT_SIZE}")
fi
if [[ -n "${SLURM_QOS:-}" ]]; then
  SBATCH_ARGS+=(--qos="${SLURM_QOS}")
fi

cat <<EOF
Comparison : ${COMPARISON_NAME}
Run        : ${MODEL}/${MODE}
Recipe     : ${RECIPE}
Shape      : ${NODES} nodes x ${GPUS_PER_NODE} GPUs, GBS=${GBS}, max_seq=${MAX_SEQUENCE_LENGTH}
CP mode    : ${CP_DESCRIPTION}
Steps      : ${NRL_MAX_STEPS}
Output     : ${RUN_DIR}
EOF

cd "${NEMO_RL_ROOT}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'CONTAINER=%q MOUNTS=%q BASE_LOG_DIR=%q COMMAND=%q sbatch' \
    "${CONTAINER}" "${MOUNTS}" "${BASE_LOG_DIR}" "${COMMAND}"
  printf ' %q' "${SBATCH_ARGS[@]}" "${NEMO_RL_ROOT}/ray.sub"
  printf '\n'
  exit 0
fi

sbatch "${SBATCH_ARGS[@]}" "${NEMO_RL_ROOT}/ray.sub"
