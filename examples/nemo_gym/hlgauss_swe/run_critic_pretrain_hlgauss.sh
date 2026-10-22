#!/bin/bash
# Value-only offline HL-Gauss critic pretraining on the same 1189 stored groups.
# Usage: bash run_qwen35_4b_critic_offline_1189_hlgauss101.sh 4b smoke|formal
set -euo pipefail

MODEL="${1:?Specify 4b or 9b}"
MODE="${2:?Specify smoke or formal}"
case "${MODEL}:${MODE}" in
  4b:smoke|4b:formal) NUM_NODES=2 ;;
  *) echo "Usage: $0 4b smoke|formal" >&2; exit 2 ;;
esac

WS="/lustre/fsw/portfolios/coreai/users/bihu/repos/multi-turn-swe"
NEMO_RL="${WS}/nemo-rl"
MBRIDGE_CODE="${WS}/nemo-rl-9b-critic-run-246fca/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"
CONFIG_FILE="${NEMO_RL}/examples/nemo_gym/hlgauss_swe/configs/ppo_qwen35_${MODEL}_critic_offline_1189_hlgauss101${MODE/smoke/_smoke}.yaml"
if [[ "${MODE}" == formal ]]; then
  CONFIG_FILE="${NEMO_RL}/examples/nemo_gym/hlgauss_swe/configs/ppo_qwen35_${MODEL}_critic_offline_1189_hlgauss101.yaml"
  PARTITION=batch
else
  PARTITION=interactive
fi
EXP_NAME="qwen35-${MODEL}-critic-offline-1189-hlgauss101${MODE/smoke/-smoke}"
if [[ "${MODE}" == formal ]]; then EXP_NAME="qwen35-${MODEL}-critic-offline-1189-hlgauss101"; fi
RESULT_DIR="/lustre/fsw/portfolios/coreai/users/bihu/rl-results/${EXP_NAME}"
CONTAINER_IMAGE="/lustre/fsw/portfolios/coreai/users/terryk/enroot-images/nvcr.io/nvidian/nemo-rl:15f91e0-60262614.squashfs"
RUNTIME_ENV_FILE="/lustre/fsw/portfolios/coreai/users/bihu/rl-scaling/env.sh"

for required_path in "${NEMO_RL}/examples/nemo_gym/run_critic_pretrain.py" \
  "${NEMO_RL}/nemo_rl/algorithms/critic_pretrain.py" "${CONFIG_FILE}" \
  "${CONTAINER_IMAGE}" "${MBRIDGE_CODE}/src/megatron/bridge/training/checkpointing.py" \
  "${NEMO_RL}/examples/nemo_gym/hlgauss_swe/configs/ppo_qwen35_4b_critic_offline_1189.yaml" \
  "/lustre/fsw/portfolios/coreai/users/bihu/rl-results/qwen35-4b-jiaqiz3145-openhands-gpp8-rollouts/views/pass_2_to_6_of_8/selection.json"; do
  [[ -e "${required_path}" ]] || { echo "Missing ${required_path}" >&2; exit 1; }
done
git -C "${NEMO_RL}" merge-base --is-ancestor 246fca0b96e7688aecd0c58c44b3cddb343bac8e HEAD || {
  echo "Experiment checkout no longer descends from validated 246fca base" >&2; exit 1;
}
for repo in "${NEMO_RL}" "${MBRIDGE_CODE}" "${MBRIDGE_CODE}/3rdparty/Megatron-LM"; do
  [[ $(git -C "${repo}" diff --name-only --diff-filter=D | wc -l) -eq 0 ]] || {
    echo "Tracked files missing from ${repo}" >&2; exit 1;
  }
done
[[ $(git -C "${MBRIDGE_CODE}" rev-parse HEAD) == a056408d29ad1070fb926512991075998c9e023f ]] || {
  echo "Megatron-Bridge commit changed" >&2; exit 1;
}
[[ $(git -C "${MBRIDGE_CODE}/3rdparty/Megatron-LM" rev-parse HEAD) == cf2f07d7b1315c96c05554c670c43207c6783e5e ]] || {
  echo "Megatron-LM commit changed" >&2; exit 1;
}

# The entrypoint initializes only the value model; Gym and vLLM are not launched.
printf -v COMMAND \
  'cd %q && ulimit -c 0 && mkdir -p /tmp/nemo_rl_triton_cache /tmp/nemo_rl_inductor_cache /tmp/nemo_rl_vllm_cache/deep_gemm /tmp/uv_cache && export PYTHONPATH=%q TRITON_CACHE_DIR=/tmp/nemo_rl_triton_cache TORCHINDUCTOR_CACHE_DIR=/tmp/nemo_rl_inductor_cache VLLM_CACHE_ROOT=/tmp/nemo_rl_vllm_cache DG_JIT_CACHE_DIR=/tmp/nemo_rl_vllm_cache/deep_gemm UV_CACHE_DIR=/tmp/uv_cache UV_LOCK_TIMEOUT=1200 NRL_WG_USE_RAY_REF=1 NRL_IGNORE_TP_ACCURACY_CHECK=1 NRL_FORCE_REBUILD_VENVS=false NRL_IGNORE_VERSION_MISMATCH=1 RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_DEEP_GEMM_WARMUP=skip TORCH_CUDA_ARCH_LIST="9.0 10.0" && date && /opt/nemo_rl_venv/bin/python examples/nemo_gym/run_critic_pretrain.py --config %q' \
  "${NEMO_RL}" "${NEMO_RL}" "${CONFIG_FILE}"
read -r -d '' SETUP_COMMAND <<'SETUPEOF' || true
if ! command -v apptainer >/dev/null 2>&1 && ! command -v singularity >/dev/null 2>&1; then
  apt-get update && apt-get install -y git build-essential gcc wget 2>/dev/null || true
  cd /tmp && wget --no-check-certificate -q https://github.com/apptainer/apptainer/releases/download/v1.3.1/apptainer_1.3.1_amd64.deb \
    && apt install -y ./apptainer_1.3.1_amd64.deb && ln -sf /usr/bin/apptainer /usr/bin/singularity || true
fi
SETUPEOF

export COMMAND SETUP_COMMAND CONTAINER="${CONTAINER_IMAGE}"
export MOUNTS="/lustre:/lustre,${NEMO_RL}:${NEMO_RL},${MBRIDGE_CODE}:/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge,${MBRIDGE_CODE}:${NEMO_RL}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"
export GPUS_PER_NODE=8 BASE_LOG_DIR="${RESULT_DIR}"
echo "${EXP_NAME}: ${NUM_NODES} value nodes, ${PARTITION}, config=${CONFIG_FILE}"
if [[ "${DRY_RUN:-0}" == 1 ]]; then exit 0; fi

source "${RUNTIME_ENV_FILE}"
: "${WANDB_API_KEY:?WANDB_API_KEY is missing}"
mkdir -p "${RESULT_DIR}"
cp "${CONFIG_FILE}" "${RESULT_DIR}/$(basename "${CONFIG_FILE}")"
cp "$0" "${RESULT_DIR}/run_qwen35_4b_critic_offline_1189_hlgauss101.sh"
cd "${NEMO_RL}"
DEPENDENCY_ARGS=()
if [[ -n "${SBATCH_DEPENDENCY:-}" ]]; then
  DEPENDENCY_ARGS+=("--dependency=${SBATCH_DEPENDENCY}")
fi
JOB_ID="$(
  WANDB_API_KEY="${WANDB_API_KEY}" HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:-}" \
  HF_TOKEN="${HF_TOKEN:-}" GITHUB_TOKEN="${GITHUB_TOKEN:-}" \
  GITLAB_TOKEN="${GITLAB_TOKEN:-}" \
  sbatch --parsable --nodes="${NUM_NODES}" --account=coreai_dlalgo_nemorl \
    --job-name="${EXP_NAME}" --partition="${PARTITION}" --time=4:00:00 \
    --gres=gpu:8 --exclusive "${DEPENDENCY_ARGS[@]}" ray.sub
)"
echo "${JOB_ID}" > "${RESULT_DIR}/latest_job_id.txt"
echo "Job submitted: ${JOB_ID}; log: ${RESULT_DIR}/${JOB_ID}-logs/ray-driver.log"
