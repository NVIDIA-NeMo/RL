#!/bin/bash
# Nemotron-Omni GRPO training on CLEVR-CoGenT — multi-node Slurm submission
#
# Usage:
#   bash yuekai_scripts/run_nemotron_omni_grpo_slurm.sh
#   NUM_NODES=4 bash yuekai_scripts/run_nemotron_omni_grpo_slurm.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEMORL="${NEMORL:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

# --- Training configuration ---
CONFIG_PATH="${CONFIG_PATH:-yuekai_scripts/configs/vlm_grpo_nemotron_omni_multinode.yaml}"
NUM_NODES="${NUM_NODES:-4}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
JOB_NAME="${JOB_NAME:-grpo-nemotron-omni-clevr}"
RESULTS_DIR="${RESULTS_DIR:-${NEMORL}/results/${JOB_NAME}}"

# --- Slurm settings ---
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-coreai_dlalgo_nemorl}"
# SBATCH_PARTITION="${SBATCH_PARTITION:-batch_short}"
# SBATCH_TIME="${SBATCH_TIME:-2:00:00}"
SBATCH_PARTITION="${SBATCH_PARTITION:-batch}"
SBATCH_TIME="${SBATCH_TIME:-4:00:00}"
SBATCH_DEPENDENCY="${SBATCH_DEPENDENCY:-singleton}"
# --- Container & environment ---
export CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/coreai/users/yuekaiz/containers/nemo_rl.0413.vllm.0.19.sqsh}"

USER_ROOT="/lustre/fsw/portfolios/coreai/users/${USER:-yuekaiz}"
export HF_HOME="${HF_HOME:-${USER_ROOT}/.cache/huggingface}"
export TMPDIR="${TMPDIR:-/tmp/nrl-${USER:-yuekaiz}}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NRL_IGNORE_VERSION_MISMATCH="${NRL_IGNORE_VERSION_MISMATCH:-1}"

# --- Validate ---
if [[ ! -f "${NEMORL}/ray.sub" ]]; then
  echo "ray.sub not found under NEMORL=${NEMORL}" >&2
  exit 1
fi

if [[ ! -f "${NEMORL}/${CONFIG_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

# --- Build training command ---
export COMMAND="\
export PYTHONPATH=\${PYTHONPATH:-}:/lustre/fsw/portfolios/coreai/users/yuekaiz/omni/automodel-omni && \
mkdir -p ${HF_HOME} ${TMPDIR} ${RESULTS_DIR} && \
uv run examples/run_vlm_grpo.py --config ${CONFIG_PATH} \
    cluster.num_nodes=${NUM_NODES} \
    cluster.gpus_per_node=${GPUS_PER_NODE} \
    checkpointing.checkpoint_dir='${RESULTS_DIR}' \
    logger.wandb.name='${JOB_NAME}'"

# --- Submit ---
cd "${NEMORL}"

MOUNTS="${MOUNTS:-/lustre:/lustre}" \
sbatch \
    --nodes="${NUM_NODES}" \
    --account="${SBATCH_ACCOUNT}" \
    --job-name="nemo-rl-${JOB_NAME}" \
    --partition="${SBATCH_PARTITION}" \
    --time="${SBATCH_TIME}" \
    --dependency="${SBATCH_DEPENDENCY}" \
    --gres="gpu:${GPUS_PER_NODE}" \
    ray.sub

echo "Submitted: ${NUM_NODES} nodes, ${GPUS_PER_NODE} GPUs/node"
echo "Results: ${RESULTS_DIR}"
