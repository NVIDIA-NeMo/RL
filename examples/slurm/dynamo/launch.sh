#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-one-node}
ROOT=${ROOT:-/lustre/fsw/portfolios/coreai/users/jothomson/nemo-rl-dynamo-slurm-new}
REPO=${REPO:-${ROOT}/RL}
ACCOUNT=${ACCOUNT:-coreai_dlalgo_nemorl}
CONTAINER=${CONTAINER:-${ROOT}/images/nemo-rl-dynamo-slurm.sqsh}
HF_HOME=${HF_HOME:-${ROOT}/cache/huggingface}
UV_CACHE_DIR_OVERRIDE=${UV_CACHE_DIR_OVERRIDE:-${ROOT}/cache/uv}
BASE_LOG_DIR=${BASE_LOG_DIR:-${ROOT}/logs}
RESULTS_DIR=${RESULTS_DIR:-${ROOT}/results/${MODE}}
CONFIG=examples/slurm/dynamo/grpo_math_1b_dynamo_ray.yaml

mkdir -p "${HF_HOME}" "${UV_CACHE_DIR_OVERRIDE}" "${BASE_LOG_DIR}" "${RESULTS_DIR}"

case "${MODE}" in
  one-node)
    NODES=1
    PARTITION=${PARTITION:-interactive}
    TIME_LIMIT=${TIME_LIMIT:-01:00:00}
    OVERRIDES=""
    RUNNER="/opt/nemo_rl_venv/bin/python -u examples/run_grpo.py --config ${CONFIG}"
    ;;
  two-node)
    NODES=2
    PARTITION=${PARTITION:-batch_short}
    TIME_LIMIT=${TIME_LIMIT:-02:00:00}
    OVERRIDES="cluster.num_nodes=2 cluster.gpus_per_node=8 policy.generation.colocated.resources.gpus_per_node=8 policy.generation.colocated.resources.num_nodes=1 grpo.num_prompts_per_step=4"
    RUNNER="/opt/nemo_rl_venv/bin/python -u examples/run_grpo.py --config ${CONFIG}"
    ;;
  refit-verifier)
    NODES=1
    PARTITION=${PARTITION:-interactive}
    TIME_LIMIT=${TIME_LIMIT:-01:00:00}
    OVERRIDES=""
    RUNNER="/opt/nemo_rl_venv/bin/python -u tools/refit_verifier.py --dynamo-config ${CONFIG}"
    ;;
  *)
    echo "usage: $0 {one-node|two-node|refit-verifier}" >&2
    exit 2
    ;;
esac

if [[ ! -f "${CONTAINER}" ]]; then
  if [[ -n "${SBATCH_DEPENDENCY:-}" ]]; then
    echo "Container is pending dependency ${SBATCH_DEPENDENCY}: ${CONTAINER}"
  else
    echo "Container squashfs not found: ${CONTAINER}" >&2
    exit 1
  fi
fi

export HF_HOME UV_CACHE_DIR_OVERRIDE BASE_LOG_DIR RESULTS_DIR
export DYNAMO_PYTHON=/opt/dynamo_venv/bin/python
# The Slurm image materializes the locked all-groups NeMo-RL environment at
# build time. Reuse it for Ray actors instead of relying on uv cache links that
# are not portable across Enroot runtime mounts.
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1
export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export GPUS_PER_NODE=8
REAL_ROOT=$(readlink -f "${ROOT}")
export MOUNTS="${ROOT}:${ROOT}"
if [[ "${REAL_ROOT}" != "${ROOT}" ]]; then
  MOUNTS+=",${REAL_ROOT}:${REAL_ROOT}"
fi
export CONTAINER
export COMMAND="cd ${REPO} && ${RUNNER} ${OVERRIDES}"

cd "${REPO}"
SBATCH_EXTRA_ARGS=()
if [[ -n "${SBATCH_DEPENDENCY:-}" ]]; then
  SBATCH_EXTRA_ARGS+=(--dependency="${SBATCH_DEPENDENCY}")
fi
sbatch \
  "${SBATCH_EXTRA_ARGS[@]}" \
  --nodes="${NODES}" \
  --account="${ACCOUNT}" \
  --job-name="nemo-rl-dynamo-${MODE}" \
  --partition="${PARTITION}" \
  --time="${TIME_LIMIT}" \
  --gres=gpu:8 \
  --output="${BASE_LOG_DIR}/slurm-${MODE}-%j.out" \
  ray.sub
