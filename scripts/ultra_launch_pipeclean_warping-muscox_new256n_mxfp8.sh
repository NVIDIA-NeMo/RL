#!/bin/bash
set -euo pipefail

# =============================================================================
# launch_ultra_256n.sh
#
# GRPO Ultra V3 — 256-node GB200 NVL72 scale test with NeMo Gym
#
# Batch-only launch script. All logs, checkpoints, and slurm output are written under a shared Lustre directory.
# Uses --dependency=singleton with a fixed job name to prevent concurrent runs.
#
# Usage:
#   ./launch_ultra_256n.sh
#   WALLTIME=4:00:00 ./launch_ultra_256n.sh
#   NRL_MAX_STEPS=10 ./launch_ultra_256n.sh
#   DRY_RUN=1 ./launch_ultra_256n.sh           # print resolved config, don't submit
#
# Adjust node allocation:
#   NUM_TRAIN_NODES=32 NUM_GEN_NODES=80 NUM_GYM_NODES=8 ./launch_ultra_256n.sh
#
# Mount local code into the container (default: container built-in, no overlays):
#   EXTRA_MOUNTS="/path/to/nemo_rl:/opt/nemo-rl/nemo_rl" ./launch_ultra_256n.sh
#
# Extra positional arguments are forwarded as Hydra overrides:
#   ./launch_ultra_256n.sh grpo.max_num_steps=2 policy.precision=float32
#
# =============================================================================

# =============================================================================
# Required environment — fail fast with clear messages
# =============================================================================
if [[ -z "${HF_HOME:-}" ]]; then
  echo "ERROR: HF_HOME is not set. Export it to a shared HuggingFace cache directory." >&2
  echo "  Example: export HF_HOME=/lustre/.../hf_home" >&2
  exit 1
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "ERROR: WANDB_API_KEY is not set. W&B logging requires an API key." >&2
  echo "  Get yours at https://wandb.ai/authorize and export WANDB_API_KEY=<key>" >&2
  exit 1
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)
cd "${PROJECT_ROOT}"
PRECISION_RECIPE=${PRECISION_RECIPE:-mxfp8}

# =============================================================================
# Model configuration (used in job naming)
# =============================================================================
TP=8
CP=8
EP=64
PP="${PP:-1}"
GPP="${GPP:-16}"
PPS="${PPS:-256}"
GBS="${GBS:-4096}"
VAL_PERIOD="${VAL_PERIOD:-10000}"
SAVE_PERIOD="${SAVE_PERIOD:-10}"
ADVANTAGE_CLIP_LOW="${ADVANTAGE_CLIP_LOW:--20}"
ADVANTAGE_CLIP_HIGH="${ADVANTAGE_CLIP_HIGH:-20}"

# =============================================================================
# Job identity — fixed name for singleton.
# Must be deterministic so that queued submissions with
# --dependency=singleton correctly serialise instead of running in parallel.
# =============================================================================
JOB_PREFIX="${JOB_PREFIX:-pipeclean-ultra-rl}"
EXP_SUFFIX="${JOB_PREFIX}-test_ultra_sft_warping-muscox_tp${TP}_cp${CP}_ep${EP}_pp${PP}_gpp${GPP}_pps${PPS}_gbs${GBS}-${PRECISION_RECIPE}-20260403-guyueh"
JOB_NAME="${EXP_SUFFIX}"

# =============================================================================
# Output directories
# =============================================================================
# ray.sub reads BASE_LOG_DIR and creates $BASE_LOG_DIR/$SLURM_JOB_ID-logs/ for
# ray infrastructure logs (ray-head.log, ray-driver.log, ray-worker-*.log,
# topology probes, attach scripts, etc.).  Shared project path so all job logs
# are easy to find regardless of who submitted or from where.
export BASE_LOG_DIR="${BASE_LOG_DIR:-/lustre/fsw/portfolios/llmservice/projects/llmservice_nemotron_ultra/nemo_rl/logs}"
export BASE_LOG_DIR="${BASE_LOG_DIR}/${EXP_SUFFIX}"

# Checkpoints and per-submission run dirs live under the submission directory.
RESULTS_DIR="${RESULTS_DIR:-$(pwd)/results/${JOB_NAME}}"

# Checkpoint dir is constant across runs so GRPO auto-resumes from the latest
# checkpoint after preemption or resubmission. The CheckpointManager scans
# this directory for the most recent checkpoint on startup.
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RESULTS_DIR}/checkpoints}"

# Per-submission dirs for logs and slurm output (timestamped for history).
RUN_DIR="${RESULTS_DIR}/runs/$(date +%Y%m%d-%H%M)"
LOG_DIR="${RUN_DIR}/logs"
SLURM_LOG_DIR="${RUN_DIR}/slurm"
mkdir -p "${CHECKPOINT_DIR}" "${LOG_DIR}" "${SLURM_LOG_DIR}"
ln -sfn "${RUN_DIR}" "${RESULTS_DIR}/runs/latest"

# =============================================================================
# SLURM configuration
# =============================================================================
SLURM_ACCOUNT="${SLURM_ACCOUNT:-llmservice_nemotron_ultra}"
PARTITION="${PARTITION:-batch}"
SLURM_QOS="${SLURM_QOS:-short}"
SLURM_RESERVATION="${SLURM_RESERVATION:-}"
WALLTIME="${WALLTIME:-2:00:00}"
EXCLUDE_NODES="${EXCLUDE_NODES:-}"

# =============================================================================
# Container & mounts
# =============================================================================
export CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/llmservice/users/guyueh/pipe.47458030.sqsh}"
MOUNTS="/lustre:/lustre"

# GB200 NVL72: fixed at 4 GPUs/node.
export GPUS_PER_NODE=4
export CPUS_PER_WORKER="${CPUS_PER_WORKER:-144}"

# =============================================================================
# HuggingFace configuration
# =============================================================================
export HF_HOME
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

# =============================================================================
# W&B configuration
# =============================================================================
WANDB_PROJ="${WANDB_PROJ:-ultra-v3-pipeclean}"
WANDB_ENTITY="${WANDB_ENTITY:-nvidia}"
WANDB_NAME="${WANDB_NAME:-${JOB_NAME}-$(date +%Y%m%d-%H%M%S)}"
export WANDB_API_KEY
export WANDB_ENTITY

# =============================================================================
# Training
# =============================================================================
NRL_MAX_STEPS="${NRL_MAX_STEPS:-}"

# =============================================================================
# Job shape — specify the 3 intuitive node counts; everything else is derived.
#
#   Training:  64 nodes (256 GPUs)  — Megatron training backend
#   vLLM:     176 nodes (704 GPUs)  — 88 instances at TP=8 EP=8
#   Gym:       16 nodes ( 64 GPUs)  — judges (GenRM, NL2Bash, Safety)
#
# =============================================================================
NUM_TRAIN_NODES="${NUM_TRAIN_NODES:-64}"
NUM_GEN_NODES="${NUM_GEN_NODES:-176}"
NUM_GYM_NODES="${NUM_GYM_NODES:-16}"

NUM_TOTAL_NODES=$((NUM_TRAIN_NODES + NUM_GEN_NODES + NUM_GYM_NODES))

# Sanity checks — catch typos before wasting a Slurm allocation.
if (( NUM_TRAIN_NODES <= 0 )); then
  echo "ERROR: NUM_TRAIN_NODES must be > 0 (got ${NUM_TRAIN_NODES})" >&2; exit 1
fi
if (( NUM_GEN_NODES <= 0 )); then
  echo "ERROR: NUM_GEN_NODES must be > 0 (got ${NUM_GEN_NODES})" >&2; exit 1
fi
if (( NUM_GYM_NODES < 0 )); then
  echo "ERROR: NUM_GYM_NODES must be >= 0 (got ${NUM_GYM_NODES})" >&2; exit 1
fi

# GB200 NVL72 topology: 18 nodes per NVLink domain, allocate in groups of 16.
SEGMENT_SIZE="${SEGMENT_SIZE:-16}"
if (( NUM_TOTAL_NODES < SEGMENT_SIZE )); then
  echo "ERROR: NUM_TOTAL_NODES=${NUM_TOTAL_NODES} < SEGMENT_SIZE=${SEGMENT_SIZE}" >&2
  exit 1
fi
if (( NUM_TOTAL_NODES % SEGMENT_SIZE != 0 )); then
  echo "ERROR: NUM_TOTAL_NODES=${NUM_TOTAL_NODES} is not divisible by SEGMENT_SIZE=${SEGMENT_SIZE}." >&2
  echo "  Training=${NUM_TRAIN_NODES} + Generation=${NUM_GEN_NODES} + Gym=${NUM_GYM_NODES} = ${NUM_TOTAL_NODES}" >&2
  echo "  Adjust node counts so the total is a multiple of ${SEGMENT_SIZE}." >&2
  exit 1
fi

# =============================================================================
# Model and data paths
# =============================================================================
NRL_TRAIN_PATH="${NRL_TRAIN_PATH:-/lustre/fsw/portfolios/llmservice/users/jiaqiz/data/gym/rl-data-tools/blends/curriculum_v29_warping-muskox.train.no_swerl.jsonl}"
NRL_VAL_PATH="${NRL_VAL_PATH:-/lustre/fsw/portfolios/llmservice/users/jiaqiz/data/gym/rl-data-tools/blends/curriculum_v29_warping-muskox.val.no_swerl.jsonl}"
NRL_MODEL_PATH="/lustre/fsw/portfolios/llmservice/users/soumyes/sft-runs/eval_and_sleep/indiv-evals/ultra-v3-sft-bf16-hybridep-ep64-cp32-bindpcie-recompute-offload-mar5-blend-512k-iter-380/hf"
NRL_GENRM_MODEL_PATH="${NRL_GENRM_MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/ansubramania/models/qwen235b_principle_comparison_genrm_step1230}"
NRL_NL2BASH_JUDGE_MODEL_PATH="${NRL_NL2BASH_JUDGE_MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/ansubramania/models/Qwen3-235B-A22B-Instruct-2507-FP8}"
NRL_SAFETY_MODEL_PATH="${NRL_SAFETY_MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/ansubramania/super_v3/model_checkpoints/Nemotron-Content-Safety-Reasoning-4B}"

# =============================================================================
# Lean4 sandbox (for math_formal_lean)
# =============================================================================
export SANDBOX_CONTAINER="${SANDBOX_CONTAINER:-/lustre/fsw/portfolios/llmservice/users/igitman/images/nemo-skills-sandbox-latest.sqsh}"
export SANDBOX_COMMAND="${SANDBOX_COMMAND:-/start-with-nginx.sh}"
export NEMO_SKILLS_SANDBOX_PORT="${NEMO_SKILLS_SANDBOX_PORT:-6000}"

# =============================================================================
# Ray log sync
# =============================================================================
export RAY_LOG_SYNC_FREQUENCY="${RAY_LOG_SYNC_FREQUENCY:-60}"

CODE_ROOT="/opt/nemo-rl"
VLLM_ENV_SOURCE="source /opt/nemo-rl/3rdparty/vllm/nemo-rl.env && "

# =============================================================================
# Persistent cache directories
# =============================================================================
# Lustre holds the warm persistent cache. At job start, SETUP_COMMAND clears
# stale /tmp caches then seeds node-local /tmp from Lustre. JIT writes go to
# /tmp to avoid Lustre metadata contention from parallel compilation.
if [[ -z "${PERSISTENT_CACHE:-}" ]]; then
  _access_group="${SLURM_ACCOUNT%%_*}"
  PERSISTENT_CACHE="/lustre/fsw/portfolios/${_access_group}/users/${USER}/.cache/nemotron_ultra"
fi
LUSTRE_VLLM_CACHE="${PERSISTENT_CACHE}/vllm_compile_cache"
LUSTRE_FLASHINFER_CUBIN_CACHE="${PERSISTENT_CACHE}/flashinfer_cubins"
LUSTRE_INDUCTOR_CACHE="${PERSISTENT_CACHE}/inductor_cache"
LUSTRE_TRITON_CACHE="${PERSISTENT_CACHE}/triton_cache"
NRL_VLLM_LOCAL_CACHE_DIR="/tmp/nemo_rl_vllm_cache"
NRL_VLLM_CACHE_SEED_DIR="/tmp/nemo_rl_vllm_cache_warm"
INDUCTOR_CACHE_DIR="/tmp/nemo_rl_inductor_cache"
TRITON_CACHE_DIR="/tmp/nemo_rl_triton_cache"
CACHE_SYNC_FREQUENCY="${CACHE_SYNC_FREQUENCY:-120}"

export LUSTRE_VLLM_CACHE
export LUSTRE_INDUCTOR_CACHE
export LUSTRE_TRITON_CACHE
export NRL_VLLM_LOCAL_CACHE_DIR
export INDUCTOR_CACHE_DIR
export TRITON_CACHE_DIR
export CACHE_SYNC_FREQUENCY

mkdir -p "${LUSTRE_VLLM_CACHE}" "${LUSTRE_FLASHINFER_CUBIN_CACHE}" \
  "${LUSTRE_INDUCTOR_CACHE}" "${LUSTRE_TRITON_CACHE}"

# =============================================================================
# Code snapshot
# =============================================================================
# Snapshot the git-tracked source tree so the code is frozen at submission time.
# This guarantees we know exactly which code was used for a given experiment.
# The snapshot directory path is recorded in the summary output and logs.
#
# Set USE_SNAPSHOT=0 to skip (runs from container built-in or live checkout).
USE_SNAPSHOT="${USE_SNAPSHOT:-1}"

if [[ "${USE_SNAPSHOT}" == "1" ]]; then
  SNAPSHOT_DIR=$(bash "${PROJECT_ROOT}/tools/code_snapshot.sh" "${JOB_NAME}")

  if [[ -d "${PROJECT_ROOT}/3rdparty/vllm" ]] && [[ ! -e "${SNAPSHOT_DIR}/3rdparty/vllm" ]]; then
    mkdir -p "${SNAPSHOT_DIR}/3rdparty"
    ln -s "${PROJECT_ROOT}/3rdparty/vllm" "${SNAPSHOT_DIR}/3rdparty/vllm"
  fi

  echo "Code snapshot: ${SNAPSHOT_DIR}"
  OVERLAY_SOURCE="${SNAPSHOT_DIR}"
else
  OVERLAY_SOURCE="${PROJECT_ROOT}"
fi

# =============================================================================
# Container mounts
# =============================================================================
# By default, nemo_rl (Python package) and examples/configs (YAML configs) from
# the code snapshot are overlaid into the container. Everything else uses the
# container's built-in code at /opt/nemo-rl.
#
# To overlay additional components, use EXTRA_MOUNTS with explicit host:container
# pairs. Examples:
#
#   # Mount Megatron-LM (will shadow prebuilt venvs — expect slow startup)
#   EXTRA_MOUNTS="/path/to/Megatron-LM:/opt/nemo-rl/3rdparty/Megatron-LM-workspace/Megatron-LM" ./scripts/launch_ultra_256n.sh
#
# Container paths for reference:
#   /opt/nemo-rl/nemo_rl                                              — Python package
#   /opt/nemo-rl/examples/configs                                     — YAML configs
#   /opt/nemo-rl/3rdparty/Megatron-LM-workspace/Megatron-LM          — Megatron-LM
#   /opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge   — Megatron-Bridge
#   /opt/nemo-rl/3rdparty/Gym-workspace/Gym                           — NeMo-Gym
#   /opt/nemo-rl/3rdparty/vllm                                        — vLLM
# =============================================================================
if [[ -d "${OVERLAY_SOURCE}/nemo_rl" ]]; then
  MOUNTS="${MOUNTS},${OVERLAY_SOURCE}/nemo_rl:/opt/nemo-rl/nemo_rl"
  echo "  Mount: nemo_rl → /opt/nemo-rl/nemo_rl"
fi
if [[ -d "${OVERLAY_SOURCE}/examples/configs" ]]; then
  MOUNTS="${MOUNTS},${OVERLAY_SOURCE}/examples/configs:/opt/nemo-rl/examples/configs"
  echo "  Mount: configs → /opt/nemo-rl/examples/configs"
fi


if [[ "${USE_SNAPSHOT}" == "1" ]]; then
  MOUNTS="${MOUNTS},${SNAPSHOT_DIR}:${SNAPSHOT_DIR}"
fi

if [[ -n "${EXTRA_MOUNTS:-}" ]]; then
  MOUNTS="${MOUNTS},${EXTRA_MOUNTS}"
  echo "  Extra mounts: ${EXTRA_MOUNTS}"
fi

export MOUNTS

# =============================================================================
# Resolve ray.sub
# =============================================================================
RAY_SUB="${RAY_SUB:-${PROJECT_ROOT}/ray.sub}"
if [[ ! -f "${RAY_SUB}" ]]; then
  echo "ERROR: ray.sub not found at ${RAY_SUB}"
  exit 1
fi

# =================================================================================================================
# Per-node cache seeding
# =================================================================================================================
# Triton, Inductor, and FlashInfer cubins compile/download to node-local /tmp to avoid Lustre race conditions
# and file lock contention during concurrent JIT compilation and cubin fetching.
# To avoid cold-start penalties, we seed /tmp from a warm Lustre cache before Ray starts (SETUP_COMMAND).
#
# IMPORTANT: Stale /tmp caches from previous jobs can cause hangs (e.g. triton_bundler
# skipping non-empty temp dirs). We rm -rf /tmp caches first, then seed fresh from Lustre.
# =================================================================================================================
read -r -d '' SETUP_COMMAND <<SETUPEOF || true
echo "[CACHE SEED] Clearing stale /tmp caches and seeding from Lustre..."
LOCAL_VLLM="${NRL_VLLM_LOCAL_CACHE_DIR}"
WARM_SEED="${NRL_VLLM_CACHE_SEED_DIR}"
LOCAL_IND="${INDUCTOR_CACHE_DIR}"
LOCAL_TRI="${TRITON_CACHE_DIR}"
L_VLLM="${LUSTRE_VLLM_CACHE}"
L_IND="${LUSTRE_INDUCTOR_CACHE}"
L_TRI="${LUSTRE_TRITON_CACHE}"

# vLLM caches are per-instance (VLLM_CACHE_ROOT_{seed}). Clear ALL from prior jobs.
rm -rf /tmp/nemo_rl_vllm_cache /tmp/nemo_rl_vllm_cache_*
rm -rf "\$LOCAL_IND" "\$LOCAL_TRI"
mkdir -p "\$LOCAL_IND" "\$LOCAL_TRI"

# Clean orphaned .tmp_* dirs left by crashed sync-back sidecars from prior jobs.
find "\$L_IND" -maxdepth 1 -name '.tmp_*' -mmin +30 -exec rm -rf {} + 2>/dev/null || true
find "\$L_TRI" -maxdepth 1 -name '.tmp_*' -mmin +30 -exec rm -rf {} + 2>/dev/null || true

_seed_cache() {
  local lustre="\$1" local_dir="\$2" name="\$3"
  if [ -d "\$lustre" ] && [ "\$(ls -A "\$lustre" 2>/dev/null)" ]; then
    rsync -a --exclude '.tmp_*' "\$lustre/" "\$local_dir/" 2>/dev/null \
      && echo "[CACHE SEED] \$name: seeded from Lustre (\$(du -sh "\$local_dir" 2>/dev/null | cut -f1))" \
      || echo "[CACHE SEED] \$name: seed failed (non-fatal)"
else
    echo "[CACHE SEED] \$name: no warm cache on Lustre yet"
fi
}

# Seed vLLM compile cache: find the most recently modified seed dir on Lustre.
# Compile hashes are seed-independent, so any prior seed dir is valid.
# Picking the newest maximises the chance it matches the current container.
_found_warm=""
if [ -n "\$L_VLLM" ]; then
  _base="\$(basename "\$L_VLLM")"
  _parent="\$(dirname "\$L_VLLM")"
  _found_warm="\$(
    ls -1dt "\${_parent}/\${_base}_"* 2>/dev/null \
      | while IFS= read -r d; do
          [ -d "\$d" ] && [ "\$(ls -A "\$d" 2>/dev/null)" ] && echo "\$d" && break
        done
  )"
fi
if [ -n "\$_found_warm" ]; then
  rm -rf "\$WARM_SEED"
  _seed_cache "\$_found_warm" "\$WARM_SEED" "vLLM (from \$(basename "\$_found_warm"))"
else
  echo "[CACHE SEED] vLLM: no warm cache on Lustre yet"
  rm -rf "\$WARM_SEED"
fi
echo "[CACHE SEED] Done."
SETUPEOF
export SETUP_COMMAND

# =============================================================================
# Build the training command
# =============================================================================
TRAIN_CMD="cd ${CODE_ROOT} && date ; \
${VLLM_ENV_SOURCE}\
OMP_NUM_THREADS=16 \
RAY_DEDUP_LOGS=1 \
VLLM_CACHE_ROOT=${NRL_VLLM_LOCAL_CACHE_DIR} \
NRL_VLLM_CACHE_SEED_DIR=${NRL_VLLM_CACHE_SEED_DIR} \
DG_JIT_CACHE_DIR=${NRL_VLLM_LOCAL_CACHE_DIR}/deep_gemm \
TORCHINDUCTOR_CACHE_DIR=${INDUCTOR_CACHE_DIR} \
TRITON_CACHE_DIR=${TRITON_CACHE_DIR} \
UV_CACHE_DIR=${PERSISTENT_CACHE}/uv \
RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
UV_HTTP_TIMEOUT=10 \
VLLM_USE_FLASHINFER_MOE_FP8=1 \
VLLM_FLASHINFER_MOE_BACKEND=latency \
NRL_VLLM_ASYNC_TIMEOUT_SECONDS=1800 \
HF_HOME=${HF_HOME} \
HF_TOKEN=${HF_TOKEN:-} \
uv run ./examples/nemo_gym/run_grpo_nemo_gym.py \
--config examples/configs/grpo_ultra_256n4g_${PRECISION_RECIPE}.yaml \
policy.model_name=${NRL_MODEL_PATH} \
cluster.gpus_per_node=4 \
cluster.num_nodes=${NUM_TOTAL_NODES} \
grpo.num_prompts_per_step=${PPS} \
grpo.num_generations_per_prompt=${GPP} \
grpo.val_period=${VAL_PERIOD} \
grpo.advantage_clip_low=${ADVANTAGE_CLIP_LOW} \
grpo.advantage_clip_high=${ADVANTAGE_CLIP_HIGH} \
grpo.penalize_invalid_tool_call=false \
grpo.penalize_malformed_thinking=false \
policy.train_global_batch_size=${GBS} \
policy.megatron_cfg.tensor_model_parallel_size=${TP} \
policy.megatron_cfg.context_parallel_size=${CP} \
policy.megatron_cfg.expert_model_parallel_size=${EP} \
policy.megatron_cfg.pipeline_model_parallel_size=${PP} \
policy.megatron_cfg.empty_unused_memory_level=1 \
policy.generation.vllm_cfg.expert_parallel_size=1 \
policy.generation.colocated.enabled=False \
policy.generation.colocated.resources.num_nodes=${NUM_GEN_NODES} \
policy.generation.colocated.resources.gpus_per_node=4 \
checkpointing.save_period=${SAVE_PERIOD} \
env.nemo_gym.num_gpu_nodes=${NUM_GYM_NODES} \
env.nemo_gym.safety_judge_model.responses_api_models.vllm_model.server_args.max_num_seqs=256 \
env.nemo_gym.nl2bash_judge_model.responses_api_models.vllm_model.server_args.max_num_seqs=256 \
env.nemo_gym.genrm_model.responses_api_models.vllm_model.server_args.max_num_seqs=256 \
env.nemo_gym.genrm_model.responses_api_models.vllm_model.model=${NRL_GENRM_MODEL_PATH} \
env.nemo_gym.nl2bash_judge_model.responses_api_models.vllm_model.model=${NRL_NL2BASH_JUDGE_MODEL_PATH} \
env.nemo_gym.safety_judge_model.responses_api_models.vllm_model.model=${NRL_SAFETY_MODEL_PATH} \
data.train.data_path=${NRL_TRAIN_PATH} \
data.validation.data_path=${NRL_VAL_PATH} \
checkpointing.checkpoint_dir=${CHECKPOINT_DIR} \
logger.log_dir=${LOG_DIR} \
logger.wandb_enabled=True \
logger.wandb.name=${WANDB_NAME} \
logger.wandb.project=${WANDB_PROJ} \
${NRL_MAX_STEPS:+grpo.max_num_steps=${NRL_MAX_STEPS}} \
${*}"

export COMMAND="${TRAIN_CMD}"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "================================================================"
echo "  GRPO Ultra V3 — ${NUM_TOTAL_NODES}-node scale test"
echo "================================================================"
echo "  Job name:    ${JOB_NAME}  (singleton — only one runs at a time)"
echo "  Nodes:       ${NUM_TOTAL_NODES} total  (segment=${SEGMENT_SIZE})"
echo "    Training:  ${NUM_TRAIN_NODES}  ($((NUM_TRAIN_NODES * GPUS_PER_NODE)) GPUs)"
echo "    vLLM gen:  ${NUM_GEN_NODES}  ($((NUM_GEN_NODES * GPUS_PER_NODE)) GPUs)"
echo "    Gym:       ${NUM_GYM_NODES}  ($((NUM_GYM_NODES * GPUS_PER_NODE)) GPUs)"
echo "  Walltime:    ${WALLTIME}"
echo ""
echo "  Checkpoints: ${CHECKPOINT_DIR}  (stable — auto-resumes across jobs)"
echo "  Run dir:     ${RUN_DIR}"
echo "  Logs:        ${LOG_DIR}"
echo "  Slurm logs:  ${SLURM_LOG_DIR}"
echo "  W&B:         ${WANDB_ENTITY}/${WANDB_PROJ} / ${WANDB_NAME}"
echo ""
echo "  Model:       ${NRL_MODEL_PATH}"
echo "  Container:   ${CONTAINER}"
if [[ "${USE_SNAPSHOT}" == "1" ]]; then
echo "  Snapshot:    ${SNAPSHOT_DIR}"
fi
echo ""
echo "  Monitor:  squeue -u \$USER -n ${JOB_NAME}"
echo "  Logs:     tail -f ${SLURM_LOG_DIR}/*.out"
echo "  Latest:   ls -la ${RESULTS_DIR}/runs/latest"
echo ""
echo "================================================================"
echo ""

# =============================================================================
# Record code provenance in the run directory
# =============================================================================
{
  echo "timestamp: $(date -Iseconds)"
  echo "branch: $(git -C "${PROJECT_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  echo "commit: $(git -C "${PROJECT_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "dirty: $(git -C "${PROJECT_ROOT}" status --porcelain 2>/dev/null | head -20)"
  echo "snapshot: ${USE_SNAPSHOT}"
  if [[ "${USE_SNAPSHOT}" == "1" ]]; then
    echo "snapshot_dir: ${SNAPSHOT_DIR}"
  fi
  echo "container: ${CONTAINER}"
  echo "command: ${TRAIN_CMD}"
} > "${RUN_DIR}/provenance.txt"

# =============================================================================
# Dry-run mode: print everything, don't submit
# =============================================================================
DRY_RUN="${DRY_RUN:-0}"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1 — printing TRAIN_CMD and exiting without submission."
  echo ""
  echo "--- TRAIN_CMD ---"
  echo "${TRAIN_CMD}"
  echo "--- end ---"
  exit 0
fi

# =============================================================================
# Submit
# =============================================================================
SBATCH_OUTPUT=$(sbatch \
  --nodes="${NUM_TOTAL_NODES}" \
  --account="${SLURM_ACCOUNT}" \
  --job-name="${JOB_NAME}" \
  --partition="${PARTITION}" \
  --time="${WALLTIME}" \
  --gres=gpu:4 \
  --exclusive \
  --mem=0 \
  --dependency=singleton \
  --segment="${SEGMENT_SIZE}" \
  --output="${SLURM_LOG_DIR}/%j.out" \
  --error="${SLURM_LOG_DIR}/%j.err" \
  ${SLURM_QOS:+--qos="${SLURM_QOS}"} \
  ${SLURM_RESERVATION:+--reservation="${SLURM_RESERVATION}"} \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES}"} \
  "${RAY_SUB}")

echo "${SBATCH_OUTPUT}"
JOB_ID=$(echo "${SBATCH_OUTPUT}" | grep -oP '\d+$')

if [[ -n "${JOB_ID}" ]]; then
  echo ""
  echo "  Ray logs:    ${BASE_LOG_DIR}/${JOB_ID}-logs/"
  echo ""
fi
