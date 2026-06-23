#!/bin/bash
set -euo pipefail
# =============================================================================
# launch_mopd_cascade_98k.sh
#
# QUEUEABLE (non-interactive) launcher for the mopd-cascade-98k-16n-clean run.
# Just `bash launch_mopd_cascade_98k.sh` on the login node — it sbatch's ray.sub.
# No interactive attach needed.
#
# Reproduces the live invocation exactly (overrides captured from
#   results/mopd-cascade-98k-16n-clean/run-20260615-164025.log):
#   - config bakes: model, data, cluster.num_nodes=16, wandb.name, max_num_steps=52,
#     save_period=5  -> we do NOT re-override those.
#   - extra overrides: gym response-logging off, gym port range, clean ckpt/log dirs,
#     wandb project.
#
# AUTO-RESUME: checkpoint_dir points at the existing results dir, so NeMo-RL resumes
# from the latest checkpoint (currently step_10). Re-running just continues; the fixed
# job-name + --dependency=singleton keep a re-submit from running concurrently with an
# in-flight job (avoids two writers on the same checkpoint_dir).
#
# Usage:
#   bash launch_mopd_cascade_98k.sh                # submit one 4h batch job (resumes)
#   INTERACTIVE=1 bash launch_mopd_cascade_98k.sh  # idle cluster to debug by hand (attach + run)
#   DRYRUN=1 bash launch_mopd_cascade_98k.sh       # print the sbatch+command, don't submit
# =============================================================================

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
cd "${PROJECT_ROOT}"

# ---------- SLURM ----------
SLURM_ACCOUNT="${SLURM_ACCOUNT:-coreai_dlalgo_nemorl}"
PARTITION="${PARTITION:-batch}"
WALLTIME="${WALLTIME:-4:00:00}"
NUM_TOTAL_NODES="${NUM_TOTAL_NODES:-16}"     # recipe bakes cluster.num_nodes=16
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

# ---------- Container (user-specified: the gym-venv-baked ultra_gym squashfs) ----------
export CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/nemotron/projects/nemotron_n3_post/users/yifuw/nemo-rl:main_ultra_gym_20260609_b.squashfs}"

# ---------- Run identity / paths ----------
CONFIG_FILE="${CONFIG_FILE:-examples/nemo_gym/mopd_cascade_replicate_16n8g_98k.yaml}"
# Fresh run identity for the skip_verification gym change (new dir -> starts from step_0,
# does NOT resume the opd_proxy_reward checkpoints under .../mopd-cascade-98k-16n-clean).
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_ROOT}/results/mopd-cascade-98k-16n-skipverify}"
WANDB_NAME="${WANDB_NAME:-mopd-cascade-98k-16n-skipverify-20260616}"
WANDB_PROJ="${WANDB_PROJ:-mopd-cascade-replicate}"
mkdir -p "${RESULTS_DIR}"

# ---------- Gym source: the in-tree submodule (has opd_proxy_reward upstream) ----------
# Mounted over the container's gym; carries the skip_verification feature (gym 74d48fb).
export NRL_GYM_DIR="${NRL_GYM_DIR:-${PROJECT_ROOT}/3rdparty/Gym-workspace/Gym}"

# ---------- Caches / HF / W&B (pass through from your shell; no secrets baked here) ----------
export HF_HOME="${HF_HOME:-/lustre/fs1/portfolios/coreai/users/yifuw/hf_home}"
export HF_TOKEN="${HF_TOKEN:-}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
PERSISTENT_CACHE="${PERSISTENT_CACHE:-/lustre/fs1/portfolios/coreai/users/yifuw/.cache/nanov3}"
# super-style: persistent vLLM compile cache on Lustre (live read+write) so the ~30-min
# compile is paid once and reused across runs. inductor/triton stay on /tmp (as in super).
VLLM_CACHE_DIR="${VLLM_CACHE_DIR:-${PERSISTENT_CACHE}/vllm_compile_cache}"
FLASHINFER_CUBIN_CACHE="${FLASHINFER_CUBIN_CACHE:-${PERSISTENT_CACHE}/flashinfer_cubins}"
FLASHINFER_WS_BASE="${FLASHINFER_WS_BASE:-${PERSISTENT_CACHE}/flashinfer_workspace}"
mkdir -p "${PERSISTENT_CACHE}/uv" "${VLLM_CACHE_DIR}" "${FLASHINFER_CUBIN_CACHE}" "${FLASHINFER_WS_BASE}"
export CPUS_PER_WORKER="${CPUS_PER_WORKER:-16}"

[[ -z "${WANDB_API_KEY}" ]] && echo "WARNING: WANDB_API_KEY unset -> wandb will be offline. export it before submitting." >&2
# code_opd/rlhf_opd run as skip_verification simple_agents (gym commit 74d48fb); no custom env needed.
grep -q "skip_verification" "${NRL_GYM_DIR}/nemo_gym/base_responses_api_agent.py" 2>/dev/null || echo "WARNING: ${NRL_GYM_DIR} lacks the skip_verification feature (gym 74d48fb) — code_opd/rlhf_opd will fail." >&2

# =============================================================================
# Overlay mounts: ship THIS worktree's code over the container's /opt/nemo-rl.
# Direct worktree overlay (NOT a git snapshot) so the UNTRACKED recipe yaml and the
# modified nemo_rl/*.py (teacher_worker_group, trajectory_collector, opd, grpo) both ship.
# =============================================================================
MOUNTS="/lustre:/lustre"
# Do NOT mount a uv cache over /root/.cache/uv: the baked venv installs some packages
# (e.g. transformers) as SYMLINKS into the image's /root/.cache/uv. Shadowing it with a
# different cache makes those links dangle -> transformers becomes an empty namespace ->
# "cannot import name 'AutoProcessor' from 'transformers' (unknown location)". The baked
# cache is self-contained; ray.sub also unsets UV_CACHE_DIR for the same reason.
_mount() { [[ -d "$1" ]] && { MOUNTS="${MOUNTS},$1:$2"; echo "  mount: $1 -> $2"; } || echo "  skip:  $1 (absent)"; }
echo "Overlay mounts:"
_mount "${PROJECT_ROOT}/nemo_rl"            "/opt/nemo-rl/nemo_rl"
_mount "${PROJECT_ROOT}/examples/nemo_gym"  "/opt/nemo-rl/examples/nemo_gym"
_mount "${PROJECT_ROOT}/examples/configs"   "/opt/nemo-rl/examples/configs"
_mount "${NRL_GYM_DIR}"                     "/opt/nemo-rl/3rdparty/Gym-workspace/Gym"
# Claude CLI data dir, so you can start claude inside the (interactive) session.
_mount "/lustre/fs1/portfolios/coreai/users/yifuw/claude_bin" "/root/.local/share/claude"
export MOUNTS

# ---------- Per-node setup: clear stale /tmp caches before Ray starts ----------
export SETUP_COMMAND='echo "[SETUP] clearing stale /tmp caches"; rm -rf /tmp/nemo_rl_vllm_cache* /tmp/nemo_rl_inductor_cache /tmp/nemo_rl_triton_cache; mkdir -p /tmp/nemo_rl_inductor_cache /tmp/nemo_rl_triton_cache'

# =============================================================================
# Training command (run inside the container; matches the live overrides exactly).
# Built as TRAIN_CMD (not exported); BATCH mode exports it as COMMAND, INTERACTIVE
# mode writes it to a run-cmd file to source by hand.
# NRL_IGNORE_VERSION_MISMATCH=1: worktree code overlays a different baked version.
# =============================================================================
TRAIN_CMD="cd /opt/nemo-rl && date ; \
OMP_NUM_THREADS=16 RAY_DEDUP_LOGS=1 NRL_VLLM_USE_V1=1 \
VLLM_CACHE_ROOT=${VLLM_CACHE_DIR} FLASHINFER_CUBIN_DIR=${FLASHINFER_CUBIN_CACHE} FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WS_BASE} \
TORCHINDUCTOR_CACHE_DIR=/tmp/nemo_rl_inductor_cache TRITON_CACHE_DIR=/tmp/nemo_rl_triton_cache \
UV_CACHE_DIR=${PERSISTENT_CACHE}/uv RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
HF_HOME=${HF_HOME} HF_TOKEN=${HF_TOKEN} WANDB_API_KEY=${WANDB_API_KEY} \
NRL_IGNORE_VERSION_MISMATCH=1 \
uv run ./examples/nemo_gym/run_grpo_nemo_gym.py \
--config ${CONFIG_FILE} \
++env.should_log_nemo_gym_responses=false \
++env.nemo_gym.port_range_low=20100 \
++env.nemo_gym.port_range_high=24900 \
checkpointing.checkpoint_dir=${RESULTS_DIR}/checkpoints \
logger.log_dir=${RESULTS_DIR}/logs \
logger.wandb_enabled=true \
logger.wandb.name=${WANDB_NAME} \
logger.wandb.project=${WANDB_PROJ}"

# =============================================================================
# Submit one job (--dependency=singleton + shared job-name block a concurrent re-submit).
# =============================================================================
echo ""
echo "Container : ${CONTAINER}"
echo "Config    : ${CONFIG_FILE}"
echo "Results   : ${RESULTS_DIR}   (auto-resumes from latest checkpoint)"
echo "Nodes     : ${NUM_TOTAL_NODES} x ${GPUS_PER_NODE}   walltime ${WALLTIME}"
echo "W&B       : ${WANDB_PROJ}/${WANDB_NAME}"

if [[ "${DRYRUN:-0}" == "1" ]]; then
  echo ""; echo "----- DRYRUN: COMMAND -----"; echo "${TRAIN_CMD}"; echo "----- MOUNTS -----"; echo "${MOUNTS}"; exit 0
fi

# --------- INTERACTIVE: bring up an idle Ray cluster to debug by hand ---------
# Empty COMMAND -> ray.sub idles the cluster and writes <jobid>-attach.sh. We drop the
# training command into a run-cmd file to source (or edit) once attached to the head node.
if [[ "${INTERACTIVE:-0}" == "1" ]]; then
  unset COMMAND 2>/dev/null || true
  SBATCH_OUTPUT=$(sbatch \
    --nodes="${NUM_TOTAL_NODES}" \
    --account="${SLURM_ACCOUNT}" \
    --job-name="interactive-${WANDB_NAME}" \
    --partition="${PARTITION}" \
    --time="${INTERACTIVE_WALLTIME:-${WALLTIME}}" \
    --gres="gpu:${GPUS_PER_NODE}" \
    --exclusive --mem=0 \
    --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"240","reason":"other","description":"mopd cascade 98k interactive"}}' \
    "${PROJECT_ROOT}/ray.sub")
  echo "${SBATCH_OUTPUT}"
  JOB_ID=$(echo "${SBATCH_OUTPUT}" | grep -oP '\d+$')
  CMD_FILE="${PROJECT_ROOT}/${JOB_ID}-run-cmd.sh"
  printf '%s\n' "${TRAIN_CMD}" > "${CMD_FILE}"; chmod +x "${CMD_FILE}"
  echo ""
  echo "  INTERACTIVE (job ${JOB_ID}). Once it is RUNNING and ${JOB_ID}-attach.sh appears:"
  echo "    attach to head node:  bash ${PROJECT_ROOT}/${JOB_ID}-attach.sh"
  echo "    run training by hand: source ${CMD_FILE}      (edit it to debug)"
  echo "    watch / cancel:       squeue -j ${JOB_ID}   |   scancel ${JOB_ID}"
  exit 0
fi

# --------- BATCH: ray.sub runs COMMAND, logging to <jobid>-logs/ray-driver.log ---------
export COMMAND="${TRAIN_CMD}"
# Inlined sbatch (ultra-v3 style): capture the output, print it, extract the job id.
SBATCH_OUTPUT=$(sbatch \
  --nodes="${NUM_TOTAL_NODES}" \
  --account="${SLURM_ACCOUNT}" \
  --job-name="${WANDB_NAME}" \
  --partition="${PARTITION}" \
  --time="${WALLTIME}" \
  --gres="gpu:${GPUS_PER_NODE}" \
  --exclusive --mem=0 \
  --dependency=singleton \
  --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"240","reason":"other","description":"mopd cascade 98k"}}' \
  "${PROJECT_ROOT}/ray.sub")

echo "${SBATCH_OUTPUT}"
JOB_ID=$(echo "${SBATCH_OUTPUT}" | grep -oP '\d+$')
echo "Driver log: ${PROJECT_ROOT}/${JOB_ID}-logs/ray-driver.log"
