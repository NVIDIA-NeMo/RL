#!/bin/bash
set -euo pipefail

# =============================================================================
# launch_mopd_nanov3_4n8g_smoke.sh
#
# MOPD smoke check on cw-dfw H100: 4 nodes x 8 GPUs (32 GPUs total).
# Non-colocated layout (yaml-defined):
#   * 2 policy training nodes (DP=2, TP=2 PP=2 CP=1 EP=2)
#   * 1 vLLM generation node     (TP=4 -> 2 replicas)
#   * 1 non-colocated teacher    (TP=2 PP=2 CP=1 EP=2)
#
# Static config: examples/nemo_gym/mopd_nanov3_4n8g_smoke.yaml
#
# Defaults to teacher == student (both = trained Nano v3 30B-A3B BF16) so the
# OPD loss should land at ~0 within a couple of steps. Last verified green
# run (student==teacher): train/loss ≈ -3e-3, teacher_student_logprob_gap ≈
# 7e-5, token_mult_prob_error ≈ 1.025. Override to run the real distillation:
#   NRL_MODEL_PATH=<base-model-path> ./launch_mopd_nanov3_4n8g_smoke.sh
#
# (NRL_TEACHER_MODEL_PATH defaults to NRL_MODEL_PATH; override separately
# only when student != teacher.)
#
# Externalized: NRL_GYM_DIR and NRL_MEGATRON_CHECKPOINT_DIR — export those
# before running if you want to point at a patched gym checkout or a
# pre-built mcore-conversion cache, otherwise the container defaults apply.
#
# Usage:
#   ./launch_mopd_nanov3_4n8g_smoke.sh                              # batch SLURM
#   NRL_MAX_STEPS=2 ./launch_mopd_nanov3_4n8g_smoke.sh              # even shorter
#   USE_WORKTREE=1 ./launch_mopd_nanov3_4n8g_smoke.sh               # overlay local code
#
# Interactive debugging (reuse allocation across runs):
#   INTERACTIVE=1 ./launch_mopd_nanov3_4n8g_smoke.sh
#   INTERACTIVE=1 INTERACTIVE_WAIT=0 ./launch_mopd_nanov3_4n8g_smoke.sh
# =============================================================================

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=${SCRIPT_DIR}
cd "${PROJECT_ROOT}"

USE_WORKTREE="${USE_WORKTREE:-0}"
INTERACTIVE="${INTERACTIVE:-0}"
INTERACTIVE_WAIT="${INTERACTIVE_WAIT:-1}"

# ---------- SLURM configuration ----------
SLURM_ACCOUNT="${SLURM_ACCOUNT:-coreai_dlalgo_nemorl}"
PARTITION="${PARTITION:-batch}"
SLURM_QOS="${SLURM_QOS:-}"
WALLTIME="${WALLTIME:-4:00:00}"

# ---------- Container & mounts ----------
export CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/coreai/users/terryk/enroot-images/nvcr.io/nvidian/nemo-rl:nightly.squashfs}"
MOUNTS="/lustre:/lustre"

# 4 nodes x 8 H100 GPUs = 32 GPUs total (2 policy + 2 teacher).
export GPUS_PER_NODE=8
export CPUS_PER_WORKER="${CPUS_PER_WORKER:-16}"

# ---------- HuggingFace ----------
export HF_HOME="${HF_HOME:-}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-}"

# ---------- Megatron HF->mcore conversion cache (externalized) ----------
# Set NRL_MEGATRON_CHECKPOINT_DIR in the calling environment to reuse an
# existing converted-mcore cache; otherwise the container default is used.
if [[ -n "${NRL_MEGATRON_CHECKPOINT_DIR:-}" ]]; then
  export NRL_MEGATRON_CHECKPOINT_DIR
  mkdir -p "${NRL_MEGATRON_CHECKPOINT_DIR}"
fi

# ---------- Training knobs ----------
NRL_MAX_STEPS="${NRL_MAX_STEPS:-}"

# ---------- Job shape ----------
# Yaml-defined layout: 1 teacher + 1 vLLM + 2 policy training = 4 total nodes.
# NUM_POLICY_NODES/NUM_TEACHER_NODES are kept for header-comment math; only
# NUM_TOTAL_NODES is used downstream by sbatch (Ray scheduler places workers
# per the yaml's cluster.num_nodes and on_policy_distillation.non_colocated_teachers.*).
NUM_POLICY_NODES="${NUM_POLICY_NODES:-2}"
NUM_TEACHER_NODES="${NUM_TEACHER_NODES:-1}"
NUM_VLLM_NODES="${NUM_VLLM_NODES:-1}"
NUM_TOTAL_NODES="${NUM_TOTAL_NODES:-$((NUM_POLICY_NODES + NUM_TEACHER_NODES + NUM_VLLM_NODES))}"

# ---------- W&B ----------
WANDB_PROJ="${WANDB_PROJ:-mopd-smoke}"
WANDB_NAME="${WANDB_NAME:-mopd-nanov3-${NUM_TOTAL_NODES}n${GPUS_PER_NODE}g-smoke-$(date +%Y%m%d-%H%M%S)}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

# ---------- Model and data paths ----------
# Trained Nano v3 BF16 — student (this run) AND teacher (always) by default.
# Points at the HF Hub snapshot under our HF_HOME (matches the green smoke).
# For the real distillation, set NRL_MODEL_PATH to the base checkpoint, e.g.
#   /lustre/fsw/portfolios/llmservice/users/igitman/hf_models/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16
NRL_MODEL_PATH="${NRL_MODEL_PATH:-/lustre/fs1/portfolios/coreai/users/yifuw/hf_home/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/snapshots/cbd3fa9f933d55ef16a84236559f4ee2a0526848}"
NRL_TEACHER_MODEL_PATH="${NRL_TEACHER_MODEL_PATH:-${NRL_MODEL_PATH}}"
NRL_TRAIN_PATH="${NRL_TRAIN_PATH:-/lustre/fs1/portfolios/coreai/users/yifuw/code/nano-v3-data/nano_data/data/train-split.jsonl}"
NRL_VAL_PATH="${NRL_VAL_PATH:-/lustre/fs1/portfolios/coreai/users/yifuw/code/nano-v3-data/nano_data/data/val-split.jsonl}"
CONFIG_FILE="${CONFIG_FILE:-examples/nemo_gym/mopd_nanov3_4n8g_smoke.yaml}"

EXP_SUFFIX="${EXP_SUFFIX:-mopd-nanov3-4n8g-smoke}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-results/${EXP_SUFFIX}}"
mkdir -p "${CHECKPOINT_DIR}"
CHECKPOINT_DIR="$(cd "${CHECKPOINT_DIR}" && pwd)"

# ---------- Code snapshot ----------
if [[ "${INTERACTIVE}" == "1" ]]; then
  USE_SNAPSHOT="${USE_SNAPSHOT:-0}"
else
  USE_SNAPSHOT="${USE_SNAPSHOT:-1}"
fi

if [[ "${USE_SNAPSHOT}" == "1" ]] && [[ -x "${PROJECT_ROOT}/tools/code_snapshot.sh" ]]; then
  SNAPSHOT_DIR=$(bash "${PROJECT_ROOT}/tools/code_snapshot.sh" "${EXP_SUFFIX}")
  echo "Code snapshot: ${SNAPSHOT_DIR}"
  OVERLAY_SOURCE="${SNAPSHOT_DIR}"
else
  USE_SNAPSHOT=0
  OVERLAY_SOURCE="${PROJECT_ROOT}"
fi

# ---------- Persistent cache directories ----------
if [[ -z "${PERSISTENT_CACHE:-}" ]]; then
  _access_group="${SLURM_ACCOUNT%%_*}"
  PERSISTENT_CACHE="/lustre/fs1/portfolios/${_access_group}/users/${USER}/.cache/nanov3"
fi
INDUCTOR_CACHE_DIR="/tmp/nemo_rl_inductor_cache"
TRITON_CACHE_DIR="/tmp/nemo_rl_triton_cache"
NRL_VLLM_LOCAL_CACHE_DIR="/tmp/nemo_rl_vllm_cache"

mkdir -p "${PERSISTENT_CACHE}/uv" "${PERSISTENT_CACHE}/gym_venvs"

export INDUCTOR_CACHE_DIR
export TRITON_CACHE_DIR
export NRL_VLLM_LOCAL_CACHE_DIR

# =============================================================================
# Validation
# =============================================================================
_walltime_secs() {
  local t="$1" h m s
  IFS=: read -r h m s <<< "${t}"
  echo $(( 10#${h} * 3600 + 10#${m} * 60 + 10#${s} ))
}

if (( $(_walltime_secs "${WALLTIME}") > 4 * 3600 )); then
  echo "WARNING: WALLTIME=${WALLTIME} exceeds 4h — many partitions cap at 4h."
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WARNING: WANDB_API_KEY is not set. W&B logging will be offline / disabled."
fi

for _p in "${NRL_MODEL_PATH}" "${NRL_TEACHER_MODEL_PATH}"; do
  if [[ "${_p}" =~ ^[a-zA-Z0-9_-]+/[a-zA-Z0-9_./-]+$ ]] && [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: ${_p} looks like a HF Hub ID but HF_TOKEN is not set." >&2
    exit 1
  fi
done

# =============================================================================
# Worktree setup (only when USE_WORKTREE=1)
# =============================================================================
if [[ "${USE_WORKTREE}" == "1" ]]; then
  WORKTREE_ROOT="$(cd "${SCRIPT_DIR}" && pwd)"
  MAIN_REPO_ROOT="${MAIN_REPO_ROOT:-$(git -C "${WORKTREE_ROOT}" worktree list --porcelain 2>/dev/null | awk '/^worktree /{print $2}' | grep -v '/.worktrees/' | head -n1)}"
  if [[ -z "${MAIN_REPO_ROOT}" || ! -d "${MAIN_REPO_ROOT}" ]]; then
    echo "Could not resolve MAIN_REPO_ROOT; set MAIN_REPO_ROOT explicitly." >&2
    exit 1
  fi
  echo "Worktree mode: overlaying ${WORKTREE_ROOT}"
fi

# =============================================================================
# Code root — container path or worktree
# =============================================================================
if [[ "${USE_WORKTREE}" == "1" ]]; then
  CODE_ROOT="${WORKTREE_ROOT}"
else
  CODE_ROOT="/opt/nemo-rl"
fi

echo "Nodes: ${NUM_TOTAL_NODES} x ${GPUS_PER_NODE} GPUs (policy=${NUM_POLICY_NODES}, vllm=${NUM_VLLM_NODES}, teacher=${NUM_TEACHER_NODES})"
echo "Config:  ${CONFIG_FILE}"
echo "Student: ${NRL_MODEL_PATH}"
echo "Teacher: ${NRL_TEACHER_MODEL_PATH}"
if [[ "${NRL_MODEL_PATH}" == "${NRL_TEACHER_MODEL_PATH}" ]]; then
  echo "  -> sanity-check mode (student == teacher); expect OPD loss ~= 0"
fi
echo "Train:   ${NRL_TRAIN_PATH}"
echo "Val:     ${NRL_VAL_PATH}"
echo "Code root: ${CODE_ROOT}"
echo "Persistent cache root: ${PERSISTENT_CACHE}"

# =============================================================================
# Build the training command
# =============================================================================
TRAIN_CMD="cd ${CODE_ROOT} && date ; \
OMP_NUM_THREADS=16 \
RAY_DEDUP_LOGS=1 \
NRL_VLLM_USE_V1=1 \
VLLM_CACHE_ROOT=${NRL_VLLM_LOCAL_CACHE_DIR} \
TORCHINDUCTOR_CACHE_DIR=${INDUCTOR_CACHE_DIR} \
TRITON_CACHE_DIR=${TRITON_CACHE_DIR} \
UV_CACHE_DIR=${PERSISTENT_CACHE}/uv \
RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
HF_HOME=${HF_HOME} \
HF_TOKEN=${HF_TOKEN:-} \
${NRL_MEGATRON_CHECKPOINT_DIR:+NRL_MEGATRON_CHECKPOINT_DIR=${NRL_MEGATRON_CHECKPOINT_DIR}} \
uv run ./examples/nemo_gym/run_grpo_nemo_gym.py \
--config ${CONFIG_FILE} \
policy.model_name=${NRL_MODEL_PATH} \
on_policy_distillation.teacher_model_by_agent_name.default_teacher=${NRL_TEACHER_MODEL_PATH} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
cluster.num_nodes=${NUM_TOTAL_NODES} \
data.train.data_path=${NRL_TRAIN_PATH} \
data.validation.data_path=${NRL_VAL_PATH} \
++env.nemo_gym.uv_venv_dir=${PERSISTENT_CACHE}/gym_venvs \
++env.nemo_gym.skip_venv_if_present=true \
checkpointing.checkpoint_dir=${CHECKPOINT_DIR}/checkpoints \
logger.log_dir=${CHECKPOINT_DIR}/logs \
logger.wandb_enabled=True \
logger.wandb.name=${WANDB_NAME} \
logger.wandb.project=${WANDB_PROJ} \
${NRL_MAX_STEPS:+grpo.max_num_steps=${NRL_MAX_STEPS}} \
${*}"

# =============================================================================
# Overlay mounts
# =============================================================================
NRL_NEMO_RL_DIR="${NRL_NEMO_RL_DIR:-${OVERLAY_SOURCE}/nemo_rl}"
NRL_CONFIGS_DIR="${NRL_CONFIGS_DIR:-${OVERLAY_SOURCE}/examples/configs}"
NRL_GYM_CONFIGS_DIR="${NRL_GYM_CONFIGS_DIR:-${OVERLAY_SOURCE}/examples/nemo_gym}"
NRL_MEGATRON_LM_DIR="${NRL_MEGATRON_LM_DIR:-}"
NRL_MEGATRON_BRIDGE_DIR="${NRL_MEGATRON_BRIDGE_DIR:-}"
NRL_GYM_DIR="${NRL_GYM_DIR:-}"

_maybe_mount() {
  local src="$1" dst="$2" label="$3"
  if [[ -z "${src}" ]]; then return; fi
  if [[ -d "${src}" ]]; then
    MOUNTS="${MOUNTS},${src}:${dst}"
    echo "  Mount: ${label} -> ${dst}"
  else
    echo "  Skip:  ${label} (${src} not found, using container built-in)"
  fi
}

echo ""
echo "Overlay mounts:"
_maybe_mount "${NRL_NEMO_RL_DIR}" "/opt/nemo-rl/nemo_rl" "nemo_rl"
_maybe_mount "${NRL_CONFIGS_DIR}" "/opt/nemo-rl/examples/configs" "configs"
_maybe_mount "${NRL_GYM_CONFIGS_DIR}" "/opt/nemo-rl/examples/nemo_gym" "nemo_gym configs"
_maybe_mount "${NRL_MEGATRON_LM_DIR}" "/opt/nemo-rl/3rdparty/Megatron-LM-workspace/Megatron-LM" "Megatron-LM"
_maybe_mount "${NRL_MEGATRON_BRIDGE_DIR}" "/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge" "Megatron-Bridge"
_maybe_mount "${NRL_GYM_DIR}" "/opt/nemo-rl/3rdparty/Gym-workspace/Gym" "NeMo-Gym"

if [[ "${USE_WORKTREE}" == "1" ]]; then
  MOUNTS="${MOUNTS},${WORKTREE_ROOT}:${WORKTREE_ROOT}"
fi
if [[ "${USE_SNAPSHOT}" == "1" ]]; then
  MOUNTS="${MOUNTS},${SNAPSHOT_DIR}:${SNAPSHOT_DIR}"
fi
if [[ -n "${EXTRA_MOUNTS:-}" ]]; then
  MOUNTS="${MOUNTS},${EXTRA_MOUNTS}"
fi
export MOUNTS

# Resolve ray.sub
if [[ "${USE_WORKTREE}" == "1" ]]; then
  RAY_SUB="${WORKTREE_ROOT}/ray.sub"
else
  RAY_SUB="${RAY_SUB:-${PROJECT_ROOT}/ray.sub}"
fi
if [[ ! -f "${RAY_SUB}" ]]; then
  echo "ERROR: ray.sub not found at ${RAY_SUB}" >&2
  exit 1
fi

# =============================================================================
# Per-node setup: clear stale /tmp caches before Ray starts.
# =============================================================================
read -r -d '' SETUP_COMMAND <<SETUPEOF || true
echo "[SETUP] Clearing stale /tmp caches..."
rm -rf /tmp/nemo_rl_vllm_cache /tmp/nemo_rl_vllm_cache_*
rm -rf "${INDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"
mkdir -p "${INDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"
echo "[SETUP] Done."
SETUPEOF
export SETUP_COMMAND

# =============================================================================
# Interactive mode
# =============================================================================
if [[ "${INTERACTIVE}" == "1" ]]; then
  unset COMMAND 2>/dev/null || true
  WALLTIME="${INTERACTIVE_WALLTIME:-${WALLTIME}}"

  echo ""
  echo "================================================================"
  echo "  INTERACTIVE MODE"
  echo "================================================================"
  echo "  Submitting ${NUM_TOTAL_NODES}-node allocation (walltime: ${WALLTIME})"
  echo ""

  submission_output=$(sbatch \
    --nodes="${NUM_TOTAL_NODES}" \
    --account="${SLURM_ACCOUNT}" \
    --job-name="interactive-${WANDB_NAME}" \
    --partition="${PARTITION}" \
    --time="${WALLTIME}" \
    --gres="gpu:${GPUS_PER_NODE}" \
    --exclusive \
    --mem=0 \
    --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"240","reason":"other","description":"debugging idle gpu"}}' \
    ${SLURM_QOS:+--qos="${SLURM_QOS}"} \
    "${RAY_SUB}")

  echo "${submission_output}"

  if [[ "${submission_output}" =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
    JOB_ID="${BASH_REMATCH[1]}"
  else
    echo "ERROR: Could not parse job ID from sbatch output." >&2
    exit 1
  fi

  LAUNCH_DIR="$(pwd)"
  ATTACH_SCRIPT="${LAUNCH_DIR}/${JOB_ID}-attach.sh"
  CMD_FILE="${LAUNCH_DIR}/${JOB_ID}-run-cmd.sh"

  cat > "${CMD_FILE}" <<CMDEOF
${TRAIN_CMD}
CMDEOF
  chmod +x "${CMD_FILE}"

  echo ""
  echo "  Saved training command to: ${CMD_FILE}"
  echo "  Attach when Ray is up:     bash ${ATTACH_SCRIPT}"
  echo "  Then run training:         source ${CMD_FILE}"
  echo "  Or non-interactively:      COMMAND=\"\$(cat ${CMD_FILE})\" bash ${ATTACH_SCRIPT}"
  echo "  Cancel:                    scancel ${JOB_ID}"

  if [[ "${INTERACTIVE_WAIT}" == "1" ]]; then
    echo ""
    echo "  Waiting for Ray to be ready (Ctrl+C to stop waiting)..."
    prev_state=""
    while [[ ! -f "${ATTACH_SCRIPT}" ]]; do
      state=$(squeue -j "${JOB_ID}" -h -o "%T" 2>/dev/null || true)
      if [[ -z "${state}" ]]; then
        echo "  Job ${JOB_ID} no longer in queue. Check: sacct -j ${JOB_ID}" >&2
        exit 1
      fi
      if [[ "${state}" != "${prev_state}" ]]; then
        echo "  [$(date +%H:%M:%S)] Job state: ${state}"
        prev_state="${state}"
      fi
      sleep 15
    done
    echo "  Ray ready! Attach: bash ${ATTACH_SCRIPT}"
  fi

  exit 0
fi

# =============================================================================
# Batch mode — submit
# =============================================================================
export COMMAND="${TRAIN_CMD}"

sbatch \
  --nodes="${NUM_TOTAL_NODES}" \
  --account="${SLURM_ACCOUNT}" \
  --job-name="${WANDB_NAME}" \
  --partition="${PARTITION}" \
  --time="${WALLTIME}" \
  --gres="gpu:${GPUS_PER_NODE}" \
  --exclusive \
  --mem=0 \
  --dependency=singleton \
  --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"240","reason":"other","description":"debugging idle gpu"}}' \
  ${SLURM_QOS:+--qos="${SLURM_QOS}"} \
  "${RAY_SUB}"
