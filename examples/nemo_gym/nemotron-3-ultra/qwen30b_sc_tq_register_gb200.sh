#!/bin/bash
# =============================================================================
# qwen30b_sc_tq_register_gb200.sh — Qwen3-30B-A3B SingleController + TQ register
# mode on 6 GB200 nodes.
#
# Verification vehicle for data_plane.backend=transfer_engine: a math GRPO run
# (no SWE sandbox, no .sif assets) through run_grpo_single_controller.py, which
# is the only entrypoint that honours the TQ data plane on the async path.
#
# Submits ray.sub directly rather than going through ultra_launch.sh, which
# hard-requires a nemo-skills SANDBOX_CONTAINER this run never uses.
#
#   bash examples/nemo_gym/nemotron-3-ultra/qwen30b_sc_tq_register_gb200.sh
#       DRY_RUN=1 (default): print the sbatch + driver command and exit
#   DRY_RUN=0 bash .../qwen30b_sc_tq_register_gb200.sh
#   DRY_RUN=0 bash .../qwen30b_sc_tq_register_gb200.sh grpo.max_num_steps=2
#
# Extra args are appended to the driver command as config overrides.
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${CODE_DIR:-$(cd "${HERE}/../../.." && pwd)}"

# Secrets carry HF_TOKEN (removes the download rate limit), WANDB_API_KEY and
# HF_HOME. Sourced, never inlined — the values must not land in this file.
NRL_SECRETS_FILE="${NRL_SECRETS_FILE:-/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_pre/users/zhiyul/secrets.sh}"
if [[ -r "${NRL_SECRETS_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${NRL_SECRETS_FILE}"
  set +a
  echo "[SECRETS] sourced ${NRL_SECRETS_FILE} (wandb=${WANDB_API_KEY:+set} hf=${HF_TOKEN:+set} HF_HOME=${HF_HOME:-unset})"
else
  echo "[WARN] no secrets file at ${NRL_SECRETS_FILE} — W&B off, HF downloads unauthenticated." >&2
fi

EXP_NAME="${EXP_NAME:-qwen30b-sc-tq-register}"
CONFIG_PATH="${CONFIG_PATH:-${CODE_DIR}/examples/configs/recipes/llm/grpo-qwen3-30ba3b-6n4g-gb200-single-controller-tq_register.yaml}"

# --- cluster shape -----------------------------------------------------------
# 4 train + 2 generation nodes; the split itself lives in the config's
# policy.generation.colocated.resources block.
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"     # GB200 NVL72
NUM_NODES="${NUM_NODES:-6}"
SEGMENT_SIZE="${SEGMENT_SIZE:-6}"              # keep the allocation on one rack
WALLTIME="${WALLTIME:-3:59:00}"
SLURM_PARTITION="${SLURM_PARTITION:-batch}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-nemotron_sw_pre}"

# --- container + caches ------------------------------------------------------
export CONTAINER="${CONTAINER:-/scratch/fsw/portfolios/nemotron/users/zhiyul/enroot-images/nvcr.io+nvidian+nemo-rl+nightly-gym.2026-08-17.squashfs}"
# HF_HOME normally comes from the secrets file; this is only the fallback.
export HF_HOME="${HF_HOME:-/lustre/fsw/portfolios/nemotron/users/zhiyul/hf_cache}"
export PERSISTENT_CACHE="${PERSISTENT_CACHE:-/scratch/fsw/portfolios/nemotron/users/zhiyul/persistent_cache}"
export NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR:-${PERSISTENT_CACHE}/megatron_ckpt_cache}"
export NRL_FORCE_REBUILD_VENVS="${NRL_FORCE_REBUILD_VENVS:-false}"

# Ray actors import nemo_rl from the image, so overlay this checkout's copy —
# the driver itself runs from ${CODE_DIR} (ray.sub sets --container-workdir to
# the submit dir), and /scratch is mounted through for the config path.
export MOUNTS="${MOUNTS:-/scratch:/scratch,/lustre:/lustre,${CODE_DIR}/nemo_rl:/opt/nemo-rl/nemo_rl,${CODE_DIR}/examples:/opt/nemo-rl/examples}"

RESULTS_DIR="${RESULTS_DIR:-${CODE_DIR}/workspace/results/${EXP_NAME}}"
SLURM_LOG_DIR="${RESULTS_DIR}/slurm"
mkdir -p "${SLURM_LOG_DIR}"
export BASE_LOG_DIR="${BASE_LOG_DIR:-${CODE_DIR}/workspace/ray_logs/${EXP_NAME}}"

WANDB_ENABLED=False
[[ -n "${WANDB_API_KEY:-}" ]] && WANDB_ENABLED=True

export COMMAND="uv run examples/run_grpo_single_controller.py \
--config ${CONFIG_PATH} \
logger.log_dir=${RESULTS_DIR}/logs \
logger.wandb_enabled=${WANDB_ENABLED} \
logger.wandb.project=${WANDB_PROJ:-nemo-rl-data-plane} \
logger.wandb.name=${EXP_NAME} \
logger.tensorboard_enabled=True \
logger.monitor_gpus=True \
checkpointing.enabled=False \
$*"

DRY_RUN="${DRY_RUN:-1}"
echo "=============================================================="
echo "  ${EXP_NAME}"
echo "  Nodes:      ${NUM_NODES} x ${GPUS_PER_NODE} GPU (segment ${SEGMENT_SIZE})"
echo "  Config:     ${CONFIG_PATH}"
echo "  Container:  ${CONTAINER}"
echo "  Code:       ${CODE_DIR}"
echo "  Logs:       ${SLURM_LOG_DIR}"
echo "  Driver command:"
echo "    ${COMMAND}"
echo "=============================================================="

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[DRY_RUN] not submitting. Re-run with DRY_RUN=0 to launch."
  exit 0
fi

cd "${CODE_DIR}"
sbatch \
  --nodes="${NUM_NODES}" \
  --account="${SLURM_ACCOUNT}" \
  --partition="${SLURM_PARTITION}" \
  --job-name="${SLURM_ACCOUNT}-${EXP_NAME}" \
  --time="${WALLTIME}" \
  --gres=gpu:"${GPUS_PER_NODE}" \
  --exclusive \
  --mem=0 \
  --segment="${SEGMENT_SIZE}" \
  --output="${SLURM_LOG_DIR}/%j.out" \
  --error="${SLURM_LOG_DIR}/%j.err" \
  "${CODE_DIR}/ray.sub"
