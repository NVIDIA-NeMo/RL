#!/bin/bash
# ppo-qwen35-4bactor-4bcritic-offline1189-step225 recipe at 128K context and 150 agent turns.
# Usage: bash examples/nemo_gym/hlgauss_swe/run_ppo_hlgauss_128k.sh
set -euo pipefail

WS="/lustre/fsw/portfolios/coreai/users/bihu/repos/multi-turn-swe"
NEMO_RL="${WS}/nemo-rl"
GYM_CODE="${WS}/gym"
MBRIDGE_CODE="${NEMO_RL}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"
CRITIC_SIZE=4b
CRITIC_STEP=285
NUM_NODES=8
NUM_TRAIN_NODES=4
VALUE_TP=4
VALUE_PP=2
VALUE_CP=2
CONFIG_FILE="${NEMO_RL}/examples/nemo_gym/hlgauss_swe/configs/ppo_qwen35_4bactor_4bcritic_offline1189_hlgauss101_step285_seq128k_turn150_dualclip3_logprob2_formal.yaml"
CONTAINER_IMAGE="/lustre/fsw/portfolios/coreai/users/terryk/enroot-images/nvcr.io/nvidian/nemo-rl:15f91e0-60262614.squashfs"
MODEL_PATH="/lustre/fsw/portfolios/coreai/users/bihu/hf_models/Qwen3.5-4B"
VALUE_MODEL_PATH="/lustre/fsw/portfolios/coreai/users/bihu/hf_models/Qwen3.5-${CRITIC_SIZE^^}"
TRAIN_DATA="/lustre/fsw/portfolios/coreai/users/bihu/rl-results/qwen35-4b-jiaqiz3145-openhands-gpp8-rollouts/profile/ppo_pass_2_to_6_of_8.jsonl"
SOURCE_CHECKPOINT="/lustre/fsw/portfolios/coreai/users/bihu/rl-results/qwen35-${CRITIC_SIZE}-critic-offline-1189-hlgauss101/eval_milestones/step_${CRITIC_STEP}"
EXPECTED_RL_COMMIT="${EXPECTED_RL_COMMIT:-}"
EXPECTED_GYM_COMMIT="8aa07d476b7482406d3e35355bcfa6efb1b4049a"
EXPECTED_MBRIDGE_COMMIT="a056408d29ad1070fb926512991075998c9e023f"
EXPECTED_MEGATRON_COMMIT="cf2f07d7b1315c96c05554c670c43207c6783e5e"
RUNTIME_ENV_FILE="/lustre/fsw/portfolios/coreai/users/bihu/rl-scaling/env.sh"

source "${RUNTIME_ENV_FILE}"
: "${WANDB_API_KEY:?WANDB_API_KEY is not configured by ${RUNTIME_ENV_FILE}}"

NUM_GEN_NODES=4
GPUS_PER_NODE_VALUE=8
POLICY_TP=4
POLICY_PP=2
POLICY_CP=4
PPS=128
GPP=1
GBS=128
SEQLEN=131072
AGENT_MAX_TURNS=150
STEPS=500
EXP_NAME="ppo-qwen35-4bactor-4bcritic-offline1189-hlgauss101-step285-seq128k-turn150-dualclip3-logprob2"
RESULTS_ROOT="${RESULTS_ROOT:-/lustre/fsw/portfolios/coreai/users/bihu/rl-results}"
RESULT_DIR="${RESULTS_ROOT}/${EXP_NAME}"
JOB_ID_FILE="${WS}/latest_ppo_qwen35_4bactor_4bcritic_offline1189_step225_seq128k_turn150_job_id.txt"
RESULT_DIR_FILE="${WS}/latest_ppo_qwen35_4bactor_4bcritic_offline1189_step225_seq128k_turn150_result_dir.txt"

for required_path in \
  "${CONFIG_FILE}" "${CONTAINER_IMAGE}" "${MODEL_PATH}" "${VALUE_MODEL_PATH}" "${TRAIN_DATA}" \
  "${MBRIDGE_CODE}/src/megatron/bridge/training/checkpointing.py" \
  "${GYM_CODE}/responses_api_agents/swe_agents/swe_openhands_setup/OpenHands"; do
  [[ -e "${required_path}" ]] || { echo "Required path does not exist: ${required_path}" >&2; exit 1; }
done
if [[ -z "${SLURM_DEPENDENCY:-}" ]]; then
  [[ -f "${SOURCE_CHECKPOINT}/training_info.json" && -d "${SOURCE_CHECKPOINT}/value/weights" ]] || {
    echo "Complete source checkpoint is missing: ${SOURCE_CHECKPOINT}" >&2
    exit 1
  }
fi

for venv_python in \
  "${GYM_CODE}/responses_api_agents/swe_agents/.venv/bin/python" \
  "${GYM_CODE}/responses_api_models/vllm_model/.venv/bin/python"; do
  [[ -L "${venv_python}" ]] || { echo "Required Gym venv launcher is missing: ${venv_python}" >&2; exit 1; }
done

actual_rl_commit="$(git -C "${NEMO_RL}" rev-parse HEAD)"
actual_gym_commit="$(git -C "${GYM_CODE}" rev-parse HEAD)"
actual_mbridge_commit="$(git -C "${MBRIDGE_CODE}" rev-parse HEAD)"
actual_megatron_commit="$(git -C "${MBRIDGE_CODE}/3rdparty/Megatron-LM" rev-parse HEAD)"
[[ -z "${EXPECTED_RL_COMMIT}" || "${actual_rl_commit}" == "${EXPECTED_RL_COMMIT}" ]] || { echo "NeMo-RL commit mismatch: expected ${EXPECTED_RL_COMMIT}, got ${actual_rl_commit}" >&2; exit 1; }
[[ "${actual_gym_commit}" == "${EXPECTED_GYM_COMMIT}" ]] || { echo "Gym commit mismatch: expected ${EXPECTED_GYM_COMMIT}, got ${actual_gym_commit}" >&2; exit 1; }
[[ "${actual_mbridge_commit}" == "${EXPECTED_MBRIDGE_COMMIT}" ]] || { echo "Megatron-Bridge commit mismatch: expected ${EXPECTED_MBRIDGE_COMMIT}, got ${actual_mbridge_commit}" >&2; exit 1; }
[[ "${actual_megatron_commit}" == "${EXPECTED_MEGATRON_COMMIT}" ]] || { echo "Megatron-LM commit mismatch: expected ${EXPECTED_MEGATRON_COMMIT}, got ${actual_megatron_commit}" >&2; exit 1; }

TRAIN_GPUS=$((NUM_TRAIN_NODES * GPUS_PER_NODE_VALUE))
POLICY_MP=$((POLICY_TP * POLICY_PP * POLICY_CP))
VALUE_MP=$((VALUE_TP * VALUE_PP * VALUE_CP))
(( NUM_NODES - NUM_GEN_NODES == NUM_TRAIN_NODES )) || { echo "Invalid node split" >&2; exit 1; }
(( TRAIN_GPUS % POLICY_MP == 0 && TRAIN_GPUS % VALUE_MP == 0 )) || { echo "Invalid model parallelism" >&2; exit 1; }
POLICY_DP=$((TRAIN_GPUS / POLICY_MP))
VALUE_DP=$((TRAIN_GPUS / VALUE_MP))
(( GBS % POLICY_DP == 0 && GBS % VALUE_DP == 0 && SEQLEN % (2 * POLICY_TP * POLICY_CP) == 0 && SEQLEN % (2 * VALUE_TP * VALUE_CP) == 0 )) || { echo "Invalid batch/sequence divisibility" >&2; exit 1; }

mkdir -p "${RESULT_DIR}"

printf -v COMMAND \
  'cd %q && ulimit -c 0 && mkdir -p /tmp/nemo_rl_triton_cache /tmp/nemo_rl_inductor_cache /tmp/nemo_rl_vllm_cache/deep_gemm /tmp/uv_cache && export PYTHONPATH=%q TRITON_CACHE_DIR=/tmp/nemo_rl_triton_cache TORCHINDUCTOR_CACHE_DIR=/tmp/nemo_rl_inductor_cache VLLM_CACHE_ROOT=/tmp/nemo_rl_vllm_cache DG_JIT_CACHE_DIR=/tmp/nemo_rl_vllm_cache/deep_gemm UV_CACHE_DIR=/tmp/uv_cache UV_LOCK_TIMEOUT=1200 NRL_VLLM_USE_V1=1 NRL_WG_USE_RAY_REF=1 NRL_IGNORE_TP_ACCURACY_CHECK=1 NRL_FORCE_REBUILD_VENVS=false NRL_IGNORE_VERSION_MISMATCH=1 RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_DEEP_GEMM_WARMUP=skip TORCH_CUDA_ARCH_LIST="9.0 10.0" NEMO_GYM_SKIP_VENV_IF_PRESENT=1 && date && /opt/nemo_rl_venv/bin/python examples/nemo_gym/run_ppo_nemo_gym.py --config %q' \
  "${NEMO_RL}" "${NEMO_RL}" "${CONFIG_FILE}"
printf -v CHECKPOINT_GUARD 'test -f %q && test -d %q && ' \
  "${SOURCE_CHECKPOINT}/training_info.json" "${SOURCE_CHECKPOINT}/value/weights"
COMMAND="${CHECKPOINT_GUARD}${COMMAND}"

read -r -d '' SETUP_COMMAND <<'SETUPEOF' || true
if ! command -v apptainer >/dev/null 2>&1 && ! command -v singularity >/dev/null 2>&1; then
  apt-get update && apt-get install -y git build-essential gcc wget 2>/dev/null || true
  cd /tmp && wget --no-check-certificate -q https://github.com/apptainer/apptainer/releases/download/v1.3.1/apptainer_1.3.1_amd64.deb \
    && apt install -y ./apptainer_1.3.1_amd64.deb && ln -sf /usr/bin/apptainer /usr/bin/singularity || true
fi
SETUPEOF

export COMMAND SETUP_COMMAND
export CONTAINER="${CONTAINER_IMAGE}"
export MOUNTS="/lustre:/lustre,${NEMO_RL}:${NEMO_RL},${GYM_CODE}:/opt/nemo-rl/3rdparty/Gym-workspace/Gym,${GYM_CODE}:${NEMO_RL}/3rdparty/Gym-workspace/Gym,${MBRIDGE_CODE}:/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge,${MBRIDGE_CODE}:${NEMO_RL}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"
export GPUS_PER_NODE="${GPUS_PER_NODE_VALUE}"
export BASE_LOG_DIR="${RESULT_DIR}"

echo "Experiment:  ${EXP_NAME}"
echo "NeMo-RL:     ${actual_rl_commit}"
echo "Gym:         ${actual_gym_commit}"
echo "Image:       ${CONTAINER_IMAGE}"
echo "Topology:    ${NUM_TRAIN_NODES} train + ${NUM_GEN_NODES} rollout nodes"
echo "Parallelism: policy TP${POLICY_TP} PP${POLICY_PP} CP${POLICY_CP} DP${POLICY_DP}; value TP${VALUE_TP} PP${VALUE_PP} CP${VALUE_CP} DP${VALUE_DP}"
echo "Shape:       PPS=${PPS}, GPP=${GPP}, GBS=${GBS}, seq=${SEQLEN}, turns=${AGENT_MAX_TURNS}, steps=${STEPS}"
echo "Init:        actor=${MODEL_PATH} (origin); critic=${SOURCE_CHECKPOINT}/value/weights"
echo "Training:    actor epochs=1, critic epochs=2, HL-Gauss101 sigma0.75, dual clip3, seq logprob threshold2; no adv penalty"
echo "Checkpoint:  NeMo-RL auto-resumes from ${RESULT_DIR}/checkpoints when available"
echo "W&B:         binhu-swe-rl-multiturn / ${EXP_NAME}"
echo "Results:     ${RESULT_DIR}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "${COMMAND}"
  exit 0
fi

cp "${CONFIG_FILE}" "${RESULT_DIR}/$(basename "${CONFIG_FILE}")"
cp "$0" "${RESULT_DIR}/$(basename "$0")"
cp "${SOURCE_CHECKPOINT}/training_info.json" "${RESULT_DIR}/source_critic_training_info.json"

cd "${NEMO_RL}"
DEPENDENCY_ARGS=()
SBATCH_SCRIPT_ARGS=("${NEMO_RL}/ray.sub")
if [[ -n "${SBATCH_DEPENDENCY:-}" ]]; then
  DEPENDENCY_ARGS+=("--dependency=${SBATCH_DEPENDENCY}")
  SBATCH_SCRIPT_ARGS=(--wrap="sleep 60; exec bash ${NEMO_RL}/ray.sub")
fi
JOB_ID="$(
  WANDB_API_KEY="${WANDB_API_KEY}" \
  HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:-}" HF_TOKEN="${HF_TOKEN:-}" \
  GITHUB_TOKEN="${GITHUB_TOKEN:-}" GITLAB_TOKEN="${GITLAB_TOKEN:-}" \
  sbatch --parsable --nodes="${NUM_NODES}" --account=coreai_dlalgo_nemorl \
    --job-name="${EXP_NAME}" --partition=batch --time=4:00:00 \
    --gres="gpu:${GPUS_PER_NODE_VALUE}" --exclusive --requeue \
    "${DEPENDENCY_ARGS[@]}" \
    --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"120","reason":"data_loading","description":"Qwen3.5 SWE PPO: model initialization, long OpenHands rollouts, and policy/value phase switching"}}' \
    "${SBATCH_SCRIPT_ARGS[@]}"
)"

echo "${JOB_ID}" > "${JOB_ID_FILE}"
echo "${RESULT_DIR}" > "${RESULT_DIR_FILE}"
echo "${JOB_ID}" > "${RESULT_DIR}/latest_job_id.txt"
echo "Job submitted: ${JOB_ID}"
echo "Driver log: ${RESULT_DIR}/${JOB_ID}-logs/ray-driver.log"
