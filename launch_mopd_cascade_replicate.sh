#!/bin/bash
set -euo pipefail

# =============================================================================
# launch_mopd_cascade_replicate.sh
#
# Replicate the Nemotron-Cascade MOPD run in THIS repo's MOPD implementation.
# Source recipe: /lustre/fs1/portfolios/coreai/users/yifuw/cascade/mopd_v2_ifwork.{sh,yaml}
# (ran on a different MOPD impl). See header of the config yaml for the porting
# deviations.
#
# cw-dfw H100: 6 nodes x 8 GPUs (48 GPUs):
#   * 2 policy training nodes (TP=2 PP=2 CP=1 EP=4 -> DP=4)
#   * 1 vLLM generation node     (TP=4 -> 2 replicas)
#   * 3 teacher nodes            (AceMath / nano-ifrl / nano-rlvr, TP=2 PP=2 EP=2 each)
#
# Config: examples/nemo_gym/mopd_cascade_replicate_6n8g.yaml
#
# Checkpoints + data are the copies under /lustre/.../cascade/{checkpoints,data}
# (already baked into the yaml). The opd_proxy_reward gym server (+ code_opd /
# rlhf_opd agents) is supplied by the patched gym at NRL_GYM_DIR (default below),
# built by applying patches/gym/opd_proxy_reward.patch to a gym source copy.
#
# Usage:
#   ./launch_mopd_cascade_replicate.sh
#   NRL_MAX_STEPS=3 ./launch_mopd_cascade_replicate.sh
#   INTERACTIVE=1 ./launch_mopd_cascade_replicate.sh
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

export GPUS_PER_NODE=8
export CPUS_PER_WORKER="${CPUS_PER_WORKER:-16}"

# ---------- HuggingFace ----------
export HF_HOME="${HF_HOME:-/lustre/fs1/portfolios/coreai/users/yifuw/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-}"

# ---------- Patched gym (supplies opd_proxy_reward + code_opd/rlhf_opd agents) ----------
# Built by: rsync the gym source (excluding .venv) to this dir, then
#   patch -p1 < <cascade>/code/nemo-rl-internal/patches/gym/opd_proxy_reward.patch
export NRL_GYM_DIR="${NRL_GYM_DIR:-/lustre/fs1/portfolios/coreai/users/yifuw/cascade/gym_patched}"

# ---------- Megatron HF->mcore conversion cache (externalized) ----------
if [[ -n "${NRL_MEGATRON_CHECKPOINT_DIR:-}" ]]; then
  export NRL_MEGATRON_CHECKPOINT_DIR
  mkdir -p "${NRL_MEGATRON_CHECKPOINT_DIR}"
fi

# ---------- Training knobs ----------
NRL_MAX_STEPS="${NRL_MAX_STEPS:-}"

# ---------- Job shape ----------
# 2 policy + 1 vLLM + 3 teachers = 6 total nodes.
NUM_POLICY_NODES="${NUM_POLICY_NODES:-2}"
NUM_VLLM_NODES="${NUM_VLLM_NODES:-1}"
NUM_TEACHER_NODES="${NUM_TEACHER_NODES:-3}"
NUM_TOTAL_NODES="${NUM_TOTAL_NODES:-$((NUM_POLICY_NODES + NUM_VLLM_NODES + NUM_TEACHER_NODES))}"

# ---------- W&B ----------
WANDB_PROJ="${WANDB_PROJ:-mopd-cascade-replicate}"
WANDB_NAME="${WANDB_NAME:-mopd-cascade-replicate-$(date +%Y%m%d-%H%M%S)}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

# ---------- Model and data paths (defaults baked in yaml; override here if needed) ----------
NRL_MODEL_PATH="${NRL_MODEL_PATH:-/lustre/fs1/portfolios/coreai/users/yifuw/cascade/checkpoints/nano_v3_oursft_33k_ifrl_stage1_dapo_step_180-workbench-stage2}"
NRL_TRAIN_PATH="${NRL_TRAIN_PATH:-/lustre/fs1/portfolios/coreai/users/yifuw/cascade/data/train_opd_multidomain_v2_comprehensive.jsonl}"
NRL_VAL_PATH="${NRL_VAL_PATH:-/lustre/fs1/portfolios/coreai/users/yifuw/cascade/data/math_val.jsonl}"
CONFIG_FILE="${CONFIG_FILE:-examples/nemo_gym/mopd_cascade_replicate_6n8g.yaml}"

EXP_SUFFIX="${EXP_SUFFIX:-mopd-cascade-replicate}"
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

if [[ ! -d "${NRL_GYM_DIR}/resources_servers/opd_proxy_reward" ]]; then
  echo "WARNING: ${NRL_GYM_DIR} has no opd_proxy_reward server — code_opd/rlhf_opd agents will fail." >&2
  echo "         Build it: rsync the gym source (no .venv) to NRL_GYM_DIR and apply opd_proxy_reward.patch." >&2
fi

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
echo "Teachers: AceMath-0033000 / nano-ifrl / nano-rlvr (from yaml)"
echo "Patched gym (NRL_GYM_DIR): ${NRL_GYM_DIR}"
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
${NRL_OPD_PACKED_TEACHERS:+NRL_OPD_PACKED_TEACHERS=${NRL_OPD_PACKED_TEACHERS}} \
uv run ./examples/nemo_gym/run_grpo_nemo_gym.py \
--config ${CONFIG_FILE} \
policy.model_name=${NRL_MODEL_PATH} \
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
_maybe_mount "${NRL_GYM_DIR}" "/opt/nemo-rl/3rdparty/Gym-workspace/Gym" "NeMo-Gym (patched: opd_proxy_reward)"

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
