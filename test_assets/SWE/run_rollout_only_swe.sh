#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ============================================================================
# ROLLOUT-ONLY latency harness, derived from run_grpo_repro_bihu_swe2.sh.
#
# Runs ONLY the rollout path (AsyncTrajectoryCollector + ReplayBuffer +
# run_async_nemo_gym_rollout) with NO training and NO weight refit. vLLM loads
# the model once and weights stay frozen. ALL GPUs go to vLLM so we can sweep
# the data-parallel replica count:
#
#   replica_count = (NUM_NODES * GPUS_PER_NODE) / VLLM_TP
#
# Faithful to bihu's rollout path EXCEPT: no in-flight weight-update pauses
# (frozen weights). In-flight concurrency (= PPS * max_trajectory_age_steps),
# per-prompt streaming, agent loop, and vLLM config are identical.
#
# Defaults (NUM_NODES=8, VLLM_TP=2 -> 32 replicas) match bihu's rollout side
# (wandb nvidia/binhu-nemo-rl/dc3m70us) for the Step-1 alignment check.
#
# Sweep axes (env vars): VLLM_TP, NUM_NODES, TARGET_STEPS.
#
# Usage:  VLLM_TP=2 NUM_NODES=8 bash test_assets/SWE/run_rollout_only_swe.sh
# ============================================================================

set -e

# ============================ Paths ============================
REPO_ROOT="/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/biguo/workspace/RL/"
CONFIG_FILE="${REPO_ROOT}/test_assets/SWE/grpo_qwen3_30b_async_swe.yaml"
ENTRYPOINT="${REPO_ROOT}/examples/nemo_gym/run_rollout_only_swe.py"
TRAIN_DATA_PATH="/lustre/fsw/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/sdevare/repos/nano/dataset/rl/swe_all_datasets_train_w_agent_ref_r2e_gym_subset.jsonl"
VAL_DATA_PATH="${TRAIN_DATA_PATH}"
# SWE1 step_230 HF checkpoint (exactly what dc3m70us used).
DEFAULT_MODEL_PATH="/lustre/fsw/portfolios/coreai/users/bihu/repos/nemo-rl-async-swe/results/qwen3-30b-thinking-swe1-async-age1-pps64-gpp8-gbs512-lr1e-06/step_230_hf"
MODEL_PATH="${1:-${MODEL_PATH:-${DEFAULT_MODEL_PATH}}}"

# ================ Container and mount config ================
# bihu's nliang container — the vLLM here lets the hermes tool parser patch apply.
export CONTAINER=${CONTAINER:-/lustre/fsw/portfolios/coreai/users/nliang/enroot-images/docker_images:nliang-qwen3-swe-training-e19dee3ba-x86_64-051626.squashfs}
GYM_CODE="${REPO_ROOT}/3rdparty/Gym-workspace/Gym"
export MOUNTS="/lustre:/lustre,$PWD:$PWD,${GYM_CODE}:/opt/nemo-rl/3rdparty/Gym-workspace/Gym"

# ======================= Cluster / resources =======================
# ALL nodes are used for inference (no training cluster).
NUM_NODES=${NUM_NODES:-8}
NUM_GPU=8
export GPUS_PER_NODE=${NUM_GPU}
export CPUS_PER_WORKER=114

# ==================== vLLM parallelism (sweep axis) ====================
VLLM_TP=${VLLM_TP:-2}
VLLM_PP=${VLLM_PP:-1}
# expert_parallel_size: 1 = EP off (default). To enable EP in async rollout it MUST
# equal TP (vLLM EP = DP*TP; EP>TP requires async_engine=False — vllm_generation.py:76-87).
VLLM_EP=${VLLM_EP:-1}
REPLICA_COUNT=$(( (NUM_NODES * NUM_GPU) / (VLLM_TP * VLLM_PP) ))

# ===================== Sequence length =====================
SEQLEN=131072

# ============== Rollout / concurrency (keep == bihu's run) ==============
# In-flight concurrency = PPS * MAX_TRAJECTORY_AGE_STEPS (prompt-groups) and
# total in-flight agents = PPS * age * GPP. GBS (samples/step) = PPS * GPP.
# Sweep GBS via GPP, keeping PPS=8 fixed: with shuffle=false this keeps the
# per-step prompt SET identical across all GBS levels (paired comparison,
# removes prompt-difficulty confound), and steps = TARGET/PPS stays at 5.
#   GBS=32  -> GPP=4  (PPS=8, TARGET_STEPS=5)
#   GBS=64  -> GPP=8  (PPS=8, TARGET_STEPS=5)  [baseline == bihu]
#   GBS=128 -> GPP=16 (PPS=8, TARGET_STEPS=5)
PPS=${PPS:-8}
GPP=${GPP:-8}
MAX_TRAJECTORY_AGE_STEPS=${MAX_TRAJECTORY_AGE_STEPS:-1}
TEMPERATURE=1.0

# ============================ SWE agent ============================
AGENT_MAX_TURNS=200
AGENT_TIMEOUT=1800

# ===================== Measurement stop conditions =====================
# Run exactly TARGET_STEPS fake-trainer steps (each step consumes PPS prompt
# groups = PPS*GPP individual rollouts). Default 5 steps.
export TARGET_STEPS=${TARGET_STEPS:-5}
# Wall-clock safety cap (3.5h) so the harness stops + flushes the jsonl/summary
# before SLURM hard-kills the 4h job.
export ROLLOUT_MAX_SECONDS=${ROLLOUT_MAX_SECONDS:-12600}
EXPECTED_STEPS=${TARGET_STEPS}

# ============================== Logging ==============================
WANDB_PROJ=${WANDB_PROJ:-"biguo-swe-rollout-latency"}
LOG_GYM_RESPONSES=false

# ========================= SLURM submission =========================
SBATCH_ACCOUNT="nemotron_sw_post"
# SBATCH_ACCOUNT="coreai_dlalgo_nemorl"
SBATCH_PARTITION="batch"
SBATCH_TIME="4:0:0"

# ========================= Experiment naming =========================
# Auto-compose the name from the factors that drive rollout latency.
EXP_SUFFIX="${EXP_SUFFIX:-rollout-only-nodes${NUM_NODES}-vllmtp${VLLM_TP}-pp${VLLM_PP}-ep${VLLM_EP}-rep${REPLICA_COUNT}-seqlen${SEQLEN}-pps${PPS}-gpp${GPP}-age${MAX_TRAJECTORY_AGE_STEPS}-turns${AGENT_MAX_TURNS}}"
WANDB_NAME="${EXP_SUFFIX}"
SNAPSHOT_DIR="${REPO_ROOT}"

# ============= Unified SLURM/Ray log location =============
export BASE_LOG_DIR="${BASE_LOG_DIR:-${SNAPSHOT_DIR}/logs/rollout_only}"
mkdir -p "${BASE_LOG_DIR}"

# ========================= Environment variables =========================
source "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/biguo/profiles/env.sh"
export HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:-${HF_TOKEN}}"
export GITLAB_TOKEN="${GITLAB_TOKEN:-}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
# Persist the uv cache on Lustre (not node-local /tmp) so the one-time
# transformer-engine source build is cached and reused across jobs/nodes,
# instead of recompiling (~20-30min) on every fresh node.
export UV_CACHE_DIR=${UV_CACHE_DIR:-/lustre/fs1/portfolios/coreai/users/biguo/.cache/uv_cache}
mkdir -p "${UV_CACHE_DIR}"
export UV_LOCK_TIMEOUT=3600
export RAY_DEDUP_LOGS=1
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export OMP_NUM_THREADS=16

# ========================= Node-local cache config =========================
PERSISTENT_CACHE="/lustre/fs1/portfolios/coreai/users/biguo/.cache/qwen3_30b_thinking_swe_rollout_only"
export LUSTRE_VLLM_CACHE="${PERSISTENT_CACHE}/vllm_compile_cache"
export LUSTRE_INDUCTOR_CACHE="${PERSISTENT_CACHE}/inductor_cache"
export LUSTRE_TRITON_CACHE="${PERSISTENT_CACHE}/triton_cache"
export NRL_VLLM_LOCAL_CACHE_DIR="/tmp/nemo_rl_vllm_cache"
export NRL_VLLM_CACHE_SEED_DIR="/tmp/nemo_rl_vllm_cache_warm"
export INDUCTOR_CACHE_DIR="/tmp/nemo_rl_inductor_cache"
export TRITON_CACHE_DIR="/tmp/nemo_rl_triton_cache"
export CACHE_SYNC_FREQUENCY=120
mkdir -p "${LUSTRE_VLLM_CACHE}" "${LUSTRE_INDUCTOR_CACHE}" "${LUSTRE_TRITON_CACHE}"

# ============================== Summary ==============================
echo "=========================================="
echo "ROLLOUT-ONLY | Experiment: ${EXP_SUFFIX}"
echo "Nodes: ${NUM_NODES}, GPUs/node: ${NUM_GPU} (all inference)"
echo "vLLM: TP=${VLLM_TP}, PP=${VLLM_PP} -> replica_count=${REPLICA_COUNT}"
echo "Rollout: PPS=${PPS}, GPP=${GPP}, age=${MAX_TRAJECTORY_AGE_STEPS}, in-flight=$((PPS*MAX_TRAJECTORY_AGE_STEPS))"
echo "Stop: TARGET_STEPS=${TARGET_STEPS} steps, MAX_SECONDS=${ROLLOUT_MAX_SECONDS}"
echo "Model: ${MODEL_PATH}"
echo "WandB: ${WANDB_PROJ}/${WANDB_NAME}"
echo "=========================================="

cd "${SNAPSHOT_DIR}"

# ================ SETUP_COMMAND (bihu's: install apptainer + seed caches + uv sync) ================
read -r -d '' SETUP_COMMAND <<SETUPEOF || true
echo "[SETUP] Installing apptainer for SWE sandbox..."
apt-get update && apt-get install -y git build-essential gcc wget 2>/dev/null || true
RET=1
RETRIES=3
for attempt in \$(seq 1 \$RETRIES); do
  if command -v apptainer >/dev/null 2>&1 || command -v singularity >/dev/null 2>&1; then
    echo "[SETUP] singularity/apptainer already available"
    RET=0
    break
  fi
  cd /tmp && \
  wget --no-check-certificate -q https://github.com/apptainer/apptainer/releases/download/v1.3.1/apptainer_1.3.1_amd64.deb && \
  apt install -y ./apptainer_1.3.1_amd64.deb && \
  ln -sf /usr/bin/apptainer /usr/bin/singularity
  if command -v apptainer >/dev/null 2>&1; then
    echo "[SETUP] apptainer installed successfully"
    RET=0
    break
  fi
  echo "[SETUP] apptainer install attempt \$attempt failed, retrying..."
  sleep 10
done
if [ \$RET -ne 0 ]; then
  echo "[SETUP] WARNING: apptainer installation failed after \$RETRIES attempts"
fi

echo "[CACHE SEED] Clearing stale /tmp caches and seeding from Lustre..."
rm -rf /tmp/nemo_rl_vllm_cache /tmp/nemo_rl_vllm_cache_*
rm -rf "${INDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"
mkdir -p "${INDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

find "${LUSTRE_INDUCTOR_CACHE}" -maxdepth 1 -name '.tmp_*' -mmin +30 -exec rm -rf {} + 2>/dev/null || true
find "${LUSTRE_TRITON_CACHE}" -maxdepth 1 -name '.tmp_*' -mmin +30 -exec rm -rf {} + 2>/dev/null || true

_seed_cache() {
  local lustre="\$1" local_dir="\$2" name="\$3"
  if [ -d "\$lustre" ] && [ "\$(ls -A "\$lustre" 2>/dev/null)" ]; then
    rsync -a --exclude '.tmp_*' "\$lustre/" "\$local_dir/" 2>/dev/null \
      && echo "[CACHE SEED] \$name: seeded from Lustre" \
      || echo "[CACHE SEED] \$name: seed failed (non-fatal)"
  else
    echo "[CACHE SEED] \$name: no warm cache on Lustre yet"
  fi
}

_seed_cache "${LUSTRE_INDUCTOR_CACHE}" "${INDUCTOR_CACHE_DIR}" "Inductor"
_seed_cache "${LUSTRE_TRITON_CACHE}" "${TRITON_CACHE_DIR}" "Triton"
echo "[CACHE SEED] Done."

UV_HTTP_TIMEOUT=3600 \
  uv sync --frozen --extra mcore
SETUPEOF
export SETUP_COMMAND

# ================ Rollout-only command (bihu-style env: uv run --frozen) ================
export COMMAND="NRL_VLLM_USE_V1=1 \
  NRL_WG_USE_RAY_REF=1 \
  WANDB_API_KEY=${WANDB_API_KEY} \
  HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN} \
  GITHUB_TOKEN=${GITHUB_TOKEN} \
  GITLAB_TOKEN=${GITLAB_TOKEN} \
  HF_HOME=${HF_HOME} \
  HF_DATASETS_CACHE=${HF_DATASETS_CACHE} \
  UV_CACHE_DIR=${UV_CACHE_DIR} \
  VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  VLLM_CACHE_ROOT=${LUSTRE_VLLM_CACHE} \
  DG_JIT_CACHE_DIR=${LUSTRE_VLLM_CACHE}/deep_gemm \
  VLLM_DEEP_GEMM_WARMUP=skip \
  NRL_FORCE_REBUILD_VENVS=false \
  NRL_IGNORE_VERSION_MISMATCH=1 \
  RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
  UV_HTTP_TIMEOUT=3600 \
  UV_LOCK_TIMEOUT=900 \
  TORCH_CUDA_ARCH_LIST='9.0 10.0' \
  NEMO_GYM_SKIP_VENV_IF_PRESENT=1 \
  TARGET_STEPS=${TARGET_STEPS} \
  ROLLOUT_MAX_SECONDS=${ROLLOUT_MAX_SECONDS} \
  uv run --frozen --extra mcore ${ENTRYPOINT} \
  --config=${CONFIG_FILE} \
  cluster.num_nodes=${NUM_NODES} \
  cluster.gpus_per_node=${NUM_GPU} \
  ++data.train.data_path=${TRAIN_DATA_PATH} \
  ++data.validation.data_path=${VAL_DATA_PATH} \
  grpo.num_prompts_per_step=${PPS} \
  grpo.num_generations_per_prompt=${GPP} \
  grpo.async_grpo.enabled=True \
  grpo.async_grpo.max_trajectory_age_steps=${MAX_TRAJECTORY_AGE_STEPS} \
  grpo.val_at_start=False \
  grpo.val_period=0 \
  env.should_log_nemo_gym_responses=${LOG_GYM_RESPONSES} \
  policy.generation.colocated.enabled=False \
  policy.model_name=${MODEL_PATH} \
  policy.max_total_sequence_length=${SEQLEN} \
  policy.generation.temperature=${TEMPERATURE} \
  policy.generation.vllm_cfg.tensor_parallel_size=${VLLM_TP} \
  policy.generation.vllm_cfg.pipeline_parallel_size=${VLLM_PP} \
  policy.generation.vllm_cfg.expert_parallel_size=${VLLM_EP} \
  policy.generation.vllm_cfg.gpu_memory_utilization=0.8 \
  policy.generation.vllm_cfg.skip_tokenizer_init=False \
  env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.agent_max_turns=${AGENT_MAX_TURNS} \
  env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.swebench_agent_timeout=${AGENT_TIMEOUT} \
  env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.agent_max_turns=${AGENT_MAX_TURNS} \
  env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.swebench_agent_timeout=${AGENT_TIMEOUT} \
  logger.wandb_enabled=True \
  logger.wandb.name=${WANDB_NAME} \
  logger.wandb.project=${WANDB_PROJ}"

# ================ Submit job ================
sbatch \
  --nodes="${NUM_NODES}" \
  --account="${SBATCH_ACCOUNT}" \
  --job-name="${WANDB_NAME}" \
  --partition="${SBATCH_PARTITION}" \
  --time="${SBATCH_TIME}" \
  --gres=gpu:${NUM_GPU} \
  --output="${BASE_LOG_DIR}/slurm-%j.out" \
  --exclusive \
  --dependency=singleton \
  --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"180","reason":"data_loading","description":"Rollout-only SWE latency harness"}}' \
  ray.sub | tee /dev/stderr | grep -o '[0-9]\+' > latest_rollout_only_job_id.txt

JOB_ID="$(cat latest_rollout_only_job_id.txt)"
echo "=========================================="
echo "Job submitted: ${EXP_SUFFIX}"
echo "Job ID: ${JOB_ID}"
echo "Monitor with: squeue -j ${JOB_ID}"
echo "Ray/SLURM logs: ${BASE_LOG_DIR}/${JOB_ID}-logs/"
echo "=========================================="

cd - > /dev/null
