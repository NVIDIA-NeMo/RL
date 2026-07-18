#!/bin/bash
set -euo pipefail

# =============================================================================
# ultra_launch.sh — launch the Nemotron 3 Ultra SWE-teacher GRPO stage on SLURM.
# The recipe (examples/configs/ultra/swe_teacher_cp16.yaml) holds the stage
# hyperparameters; this launcher does orchestration only: SLURM submit, code
# snapshot, /tmp caches, container mounts, per-run Hydra overrides.
#
# Required env: EXP_NAME CONFIG_PATH MODEL_PATH TRAIN_PATH VAL_PATH CONTAINER
#               SANDBOX_CONTAINER PERSISTENT_CACHE SLURM_PARTITION SLURM_ACCOUNT
# Common knobs: WALLTIME SLURM_QOS SLURM_RESERVATION EXCLUDE_NODES SLURM_DEPENDENCY
#               NUM_TRAIN_NODES NUM_GEN_NODES NUM_GYM_NODES SEGMENT_SIZE SIF_DIR
#               USE_SNAPSHOT DRY_RUN NRL_MAX_STEPS EXTRA_MOUNTS HF_HOME HF_TOKEN
#               WANDB_API_KEY WANDB_PROJ WANDB_ENTITY
# Extra positional args pass through as Hydra overrides. GB200 NVL72 = 4 GPU/node;
# NUM_TRAIN+NUM_GEN+NUM_GYM must be a multiple of SEGMENT_SIZE.
# =============================================================================

# --- Required environment ---
: "${EXP_NAME:?EXP_NAME is required (job name, W&B run, checkpoint/log dirs)}"
: "${CONFIG_PATH:?CONFIG_PATH is required (e.g. examples/configs/ultra/swe_teacher_cp16.yaml)}"
: "${MODEL_PATH:?MODEL_PATH is required (initial policy checkpoint, HF repo id or local path)}"
: "${TRAIN_PATH:?TRAIN_PATH is required (training data jsonl path)}"
: "${VAL_PATH:?VAL_PATH is required (validation data jsonl path)}"
: "${CONTAINER:?CONTAINER is required (NGC image URI or .sqsh path)}"
: "${SANDBOX_CONTAINER:?SANDBOX_CONTAINER is required (nemo-skills sandbox image)}"
: "${PERSISTENT_CACHE:?PERSISTENT_CACHE is required (Lustre dir for the uv cache)}"
: "${SLURM_PARTITION:?SLURM_PARTITION is required}"
: "${SLURM_ACCOUNT:?SLURM_ACCOUNT is required}"
SIF_DIR="${SIF_DIR:-}"   # apptainer .sif dir for SWE instances; yaml container_formatter uses ${sif_dir}/...

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "ERROR: CONFIG_PATH does not exist: ${CONFIG_PATH}" >&2; exit 1
fi

# --- Project root. Do NOT realpath: users/<u> is a symlink to /lustre/fs1 (NOT
# mounted on GB200 compute); keep the fsw literal so pyxis workdir/mounts resolve. ---
PROJECT_ROOT="${PROJECT_ROOT_OVERRIDE:-$PWD}"
cd "${PROJECT_ROOT}"

# --- Job identity: fixed name so --dependency=singleton serialises resubmissions
# (a requeue after preemption resumes from the latest checkpoint). ---
JOB_NAME="${EXP_NAME}"

# --- Output dirs (per-submission, timestamped) ---
RESULTS_DIR="${RESULTS_DIR:-results/${EXP_NAME}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RESULTS_DIR}/checkpoints}"
RUN_DIR="${RESULTS_DIR}/runs/$(date +%Y%m%d-%H%M)"
LOG_DIR="${RUN_DIR}/logs"
SLURM_LOG_DIR="${RUN_DIR}/slurm"
mkdir -p "${CHECKPOINT_DIR}" "${LOG_DIR}" "${SLURM_LOG_DIR}"
ln -sfn "$(realpath "${RUN_DIR}")" "${RESULTS_DIR}/runs/latest"
export BASE_LOG_DIR="${BASE_LOG_DIR:-${RESULTS_DIR}/ray_logs}"   # ray.sub writes $BASE_LOG_DIR/$JOBID-logs/

# --- SLURM config ---
WALLTIME="${WALLTIME:-4:00:00}"
SLURM_QOS="${SLURM_QOS:-}"
SLURM_RESERVATION="${SLURM_RESERVATION:-}"
EXCLUDE_NODES="${EXCLUDE_NODES:-}"
# CHECKPOINTING_SAVE_BY (DD:HH:MM:SS): stop training early to save a final checkpoint
# before walltime. Unset to let slurm walltime end the job (fine when each step saves).
CHECKPOINTING_SAVE_BY="${CHECKPOINTING_SAVE_BY:-}"

# --- Container + GB200 fixed 4 GPU/node ---
export CONTAINER
MOUNTS="${MOUNTS:-}"
export GPUS_PER_NODE=4
export CPUS_PER_WORKER="${CPUS_PER_WORKER:-144}"

# --- HuggingFace ---
if [[ -n "${HF_HOME:-}" ]]; then
  export HF_HOME
  export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/hub}"
else
  echo "[WARN] HF_HOME unset — HuggingFace uses the per-node default cache." >&2
fi

# --- W&B ---
WANDB_PROJ="${WANDB_PROJ:-nemotron-3-ultra}"
WANDB_NAME="${EXP_NAME}"
WANDB_ENABLED=False
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  export WANDB_API_KEY
  WANDB_ENABLED=True
  [[ -n "${WANDB_ENTITY:-}" ]] && export WANDB_ENTITY
else
  echo "[WARN] WANDB_API_KEY unset — W&B logging disabled." >&2
fi

NRL_MAX_STEPS="${NRL_MAX_STEPS:-}"

# --- Job shape: colocated GRPO (train + vLLM gen + SWE gym) ---
NUM_TRAIN_NODES="${NUM_TRAIN_NODES:-64}"
NUM_GEN_NODES="${NUM_GEN_NODES:-172}"
NUM_GYM_NODES="${NUM_GYM_NODES:-20}"
NUM_ACTOR_NODES=$((NUM_TRAIN_NODES + NUM_GEN_NODES))
NUM_TOTAL_NODES=$((NUM_ACTOR_NODES + NUM_GYM_NODES))
(( NUM_TRAIN_NODES > 0 )) || { echo "ERROR: NUM_TRAIN_NODES must be > 0" >&2; exit 1; }
(( NUM_GEN_NODES   > 0 )) || { echo "ERROR: NUM_GEN_NODES must be > 0" >&2; exit 1; }
(( NUM_GYM_NODES  >= 0 )) || { echo "ERROR: NUM_GYM_NODES must be >= 0" >&2; exit 1; }

# GB200 NVL72: allocate in groups of SEGMENT_SIZE (one NVLink domain group).
SEGMENT_SIZE="${SEGMENT_SIZE:-16}"
if (( NUM_TOTAL_NODES < SEGMENT_SIZE || NUM_TOTAL_NODES % SEGMENT_SIZE != 0 )); then
  echo "ERROR: NUM_TOTAL_NODES=${NUM_TOTAL_NODES} (train ${NUM_TRAIN_NODES} + gen ${NUM_GEN_NODES} + gym ${NUM_GYM_NODES}) must be a positive multiple of SEGMENT_SIZE=${SEGMENT_SIZE}." >&2
  exit 1
fi

# --- NeMo Skills sandbox (SWE code-execution rewards) ---
export SANDBOX_CONTAINER
export SANDBOX_COMMAND="${SANDBOX_COMMAND:-/start-with-nginx.sh}"
export NEMO_SKILLS_SANDBOX_PORT="${NEMO_SKILLS_SANDBOX_PORT:-6000}"
export RAY_LOG_SYNC_FREQUENCY="${RAY_LOG_SYNC_FREQUENCY:-60}"

CODE_ROOT="/opt/nemo-rl"
# nemo-rl.env only exists in the custom-vLLM build; skip it on the standard nightly.
VLLM_ENV_SOURCE="{ [ -f /opt/nemo-rl/3rdparty/vllm/nemo-rl.env ] && source /opt/nemo-rl/3rdparty/vllm/nemo-rl.env || echo '[INFO] no custom vllm nemo-rl.env; using container default vllm'; } ; "

# --- /tmp JIT caches (node-local, avoids Lustre metadata contention). Consumed by TRAIN_CMD. ---
NRL_VLLM_LOCAL_CACHE_DIR="/tmp/nemo_rl_vllm_cache"
INDUCTOR_CACHE_DIR="/tmp/nemo_rl_inductor_cache"
TRITON_CACHE_DIR="/tmp/nemo_rl_triton_cache"
export NRL_VLLM_LOCAL_CACHE_DIR INDUCTOR_CACHE_DIR TRITON_CACHE_DIR

# --- Code snapshot: freeze the git tree at submission (USE_SNAPSHOT=0 = live checkout) ---
USE_SNAPSHOT="${USE_SNAPSHOT:-1}"
if [[ "${USE_SNAPSHOT}" == "1" ]]; then
  [[ -f "${PROJECT_ROOT}/tools/code_snapshot.sh" ]] || { echo "ERROR: tools/code_snapshot.sh not found (set USE_SNAPSHOT=0)" >&2; exit 1; }
  SNAPSHOT_DIR=$(bash "${PROJECT_ROOT}/tools/code_snapshot.sh" "${JOB_NAME}")
  if [[ -d "${PROJECT_ROOT}/3rdparty/vllm" && ! -e "${SNAPSHOT_DIR}/3rdparty/vllm" ]]; then
    mkdir -p "${SNAPSHOT_DIR}/3rdparty"
    ln -s "${PROJECT_ROOT}/3rdparty/vllm" "${SNAPSHOT_DIR}/3rdparty/vllm"
  fi
  echo "Code snapshot: ${SNAPSHOT_DIR}"
  OVERLAY_SOURCE="${SNAPSHOT_DIR}"
else
  OVERLAY_SOURCE="${PROJECT_ROOT}"
fi

# --- Container mounts: overlay nemo_rl + examples/configs (+ Gym if present) from the
# snapshot; everything else uses the container's /opt/nemo-rl. EXTRA_MOUNTS adds
# host:container pairs (e.g. a local Megatron-LM checkout). ---
_append_mount() { MOUNTS="${MOUNTS:+${MOUNTS},}$1"; }
[[ -d "${OVERLAY_SOURCE}/nemo_rl" ]] && _append_mount "${OVERLAY_SOURCE}/nemo_rl:/opt/nemo-rl/nemo_rl"
[[ -d "${OVERLAY_SOURCE}/examples/configs" ]] && _append_mount "${OVERLAY_SOURCE}/examples/configs:/opt/nemo-rl/examples/configs"
[[ -d "${OVERLAY_SOURCE}/3rdparty/Gym-workspace/Gym" ]] && _append_mount "${OVERLAY_SOURCE}/3rdparty/Gym-workspace/Gym:/opt/nemo-rl/3rdparty/Gym-workspace/Gym"
[[ "${USE_SNAPSHOT}" == "1" ]] && _append_mount "${SNAPSHOT_DIR}:${SNAPSHOT_DIR}"
[[ -n "${EXTRA_MOUNTS:-}" ]] && _append_mount "${EXTRA_MOUNTS}"
export MOUNTS

RAY_SUB="${RAY_SUB:-${PROJECT_ROOT}/ray.sub}"
[[ -f "${RAY_SUB}" ]] || { echo "ERROR: ray.sub not found at ${RAY_SUB}" >&2; exit 1; }

# --- Per-node /tmp cache reset: stale dirs from a prior job can hang the Triton bundler. ---
read -r -d '' SETUP_COMMAND <<SETUPEOF || true
rm -rf /tmp/nemo_rl_vllm_cache /tmp/nemo_rl_vllm_cache_* "${INDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"
mkdir -p "${INDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"
SETUPEOF
export SETUP_COMMAND

# --- Training command: recipe holds the hyperparameters; pass only per-run overrides. ---
TRAIN_CMD="cd ${CODE_ROOT} && date ; \
${VLLM_ENV_SOURCE}\
OMP_NUM_THREADS=16 \
RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-1} \
WANDB_INIT_TIMEOUT=300 \
VLLM_CACHE_ROOT=${NRL_VLLM_LOCAL_CACHE_DIR} \
DG_JIT_CACHE_DIR=${NRL_VLLM_LOCAL_CACHE_DIR}/deep_gemm \
TORCHINDUCTOR_CACHE_DIR=${INDUCTOR_CACHE_DIR} \
TRITON_CACHE_DIR=${TRITON_CACHE_DIR} \
UV_CACHE_DIR=${PERSISTENT_CACHE}/uv \
RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
UV_HTTP_TIMEOUT=10 \
${NRL_FORCE_FLASHINFER_MOE_FP8:+VLLM_USE_FLASHINFER_MOE_FP8=1 VLLM_FLASHINFER_MOE_BACKEND=latency} \
NRL_VLLM_ASYNC_TIMEOUT_SECONDS=1800 \
NRL_WG_USE_RAY_REF=1 \
HF_HOME=${HF_HOME:-} \
HF_TOKEN=${HF_TOKEN:-} \
NRL_USE_FASTOKENS=${NRL_USE_FASTOKENS:-1} \
${NRL_NAN_CAPTURE:+NRL_NAN_CAPTURE=${NRL_NAN_CAPTURE} NRL_NAN_CAPTURE_DIR=${NRL_NAN_CAPTURE_DIR:-${LOG_DIR}/nan_capture} NRL_NAN_GUARD=${NRL_NAN_GUARD:-1}} \
uv run ./examples/nemo_gym/run_grpo_nemo_gym.py \
--config ${CONFIG_PATH} \
policy.model_name=${MODEL_PATH} \
cluster.num_nodes=${NUM_ACTOR_NODES} \
policy.generation.colocated.resources.num_nodes=${NUM_GEN_NODES} \
checkpointing.checkpoint_dir=${CHECKPOINT_DIR} \
${CHECKPOINTING_SAVE_BY:+checkpointing.checkpoint_must_save_by=${CHECKPOINTING_SAVE_BY}} \
data.train.data_path=${TRAIN_PATH} \
data.validation.data_path=${VAL_PATH} \
${SIF_DIR:+sif_dir=${SIF_DIR}} \
env.nemo_gym.nemo_gym_log_dir=${LOG_DIR}/nemo_gym \
logger.log_dir=${LOG_DIR} \
logger.wandb_enabled=${WANDB_ENABLED} \
logger.wandb.name=${WANDB_NAME} \
logger.wandb.project=${WANDB_PROJ} \
${NRL_MAX_STEPS:+grpo.max_num_steps=${NRL_MAX_STEPS}} \
${*}"
export COMMAND="${TRAIN_CMD}"

# Redacted copy for printing so HF_TOKEN never lands in logs on a shared FS.
if [[ -n "${HF_TOKEN:-}" ]]; then
  TRAIN_CMD_SAFE="${TRAIN_CMD//${HF_TOKEN}/<redacted>}"
else
  TRAIN_CMD_SAFE="${TRAIN_CMD}"
fi

cat <<EOF

======== Nemotron 3 Ultra — ${EXP_NAME} (${NUM_TOTAL_NODES}-node) ========
  Job:      ${JOB_NAME} (singleton)   Config: ${CONFIG_PATH}
  Nodes:    ${NUM_TRAIN_NODES} train + ${NUM_GEN_NODES} gen + ${NUM_GYM_NODES} gym  (segment=${SEGMENT_SIZE})
  Walltime: ${WALLTIME}   W&B: ${WANDB_PROJ}/${WANDB_NAME} (enabled=${WANDB_ENABLED})
  Model:    ${MODEL_PATH}
  Ckpts:    ${CHECKPOINT_DIR} (auto-resumes)   Logs: ${SLURM_LOG_DIR}
  Monitor:  squeue -u \$USER -n ${JOB_NAME} ; tail -f ${SLURM_LOG_DIR}/*.out
EOF

# --- Dry-run: print the command and exit without submitting ---
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo ""; echo "DRY_RUN=1 — TRAIN_CMD:"; echo "${TRAIN_CMD_SAFE}"; exit 0
fi

# --- Submit. Singleton serialises same-name runs; SLURM_DEPENDENCY chains extra deps. ---
SLURM_DEPENDENCY="${SLURM_DEPENDENCY:-}"
DEPENDENCY="singleton${SLURM_DEPENDENCY:+,${SLURM_DEPENDENCY}}"
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
  "${RAY_SUB}")
echo "${SBATCH_OUTPUT}"
JOB_ID=$(echo "${SBATCH_OUTPUT}" | grep -oP '\d+$')
[[ -n "${JOB_ID}" ]] && echo "  Ray logs: ${BASE_LOG_DIR}/${JOB_ID}-logs/"
