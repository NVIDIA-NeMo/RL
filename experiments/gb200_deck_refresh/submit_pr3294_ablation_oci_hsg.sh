#!/usr/bin/env bash

set -euo pipefail

ACTION=${ACTION:-render}
ARM=${ARM:-baseline}
STUDY=${STUDY:-full_ablation}
MAX_STEPS=${MAX_STEPS:-20}
RUN_SUFFIX=${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}

case "${STUDY}:${ARM}" in
  full_ablation:baseline)
    TARGET_COMMIT=${TARGET_COMMIT:-313f41a9654cd67e44d783128543fe1638c778da}
    REFIT_PREQUANTIZE=false; BATCHED_SHUFFLE=0; CACHED_LOADERS=0
    USE_RUNTIME_TOGGLES=true
    ;;
  full_ablation:optimized)
    TARGET_COMMIT=${TARGET_COMMIT:-313f41a9654cd67e44d783128543fe1638c778da}
    REFIT_PREQUANTIZE=true; BATCHED_SHUFFLE=1; CACHED_LOADERS=1
    USE_RUNTIME_TOGGLES=true
    ;;
  shuffle_only:baseline)
    TARGET_COMMIT=${TARGET_COMMIT:-e45e29da7266a7a219d2a0bc4adb0a1f78456985}
    REFIT_PREQUANTIZE=true; BATCHED_SHUFFLE=0; CACHED_LOADERS=unchanged
    USE_RUNTIME_TOGGLES=false
    ;;
  shuffle_only:optimized)
    TARGET_COMMIT=${TARGET_COMMIT:-d5fb8d044031420e9170aae66ee0c3166b798381}
    REFIT_PREQUANTIZE=true; BATCHED_SHUFFLE=1; CACHED_LOADERS=unchanged
    USE_RUNTIME_TOGGLES=false
    ;;
  *)
    echo "STUDY must be full_ablation or shuffle_only; ARM must be baseline or optimized" >&2
    exit 2
    ;;
esac

CONFIG=examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g-mxfp8-rollout.yaml
RUN_ARGS=(
  --config "${CONFIG}"
  "cluster.num_nodes=4"
  "cluster.gpus_per_node=4"
  "cluster.segment_size=4"
  "grpo.async_grpo.enabled=false"
  "policy.generation.colocated.enabled=true"
  "policy.generation.refit_transport=null"
  "policy.generation.real_quant_export_cpu_offload=true"
  "policy.generation.vllm_cfg.enforce_eager=false"
  "policy.generation.vllm_cfg.async_engine=false"
  "policy.generation.vllm_cfg.refit_prequantize=${REFIT_PREQUANTIZE}"
  "grpo.max_num_steps=${MAX_STEPS}"
  "grpo.seed=42"
  "grpo.val_at_start=false"
  "++grpo.val_at_end=false"
  "checkpointing.enabled=false"
)
RUNTIME_TOGGLE_EXPORTS=
if [[ "${USE_RUNTIME_TOGGLES}" == true ]]; then
  RUNTIME_TOGGLE_EXPORTS=$(cat <<EOF
export NRL_MXFP8_BATCHED_SHUFFLE=${BATCHED_SHUFFLE}
export NRL_REFIT_CACHED_LOADERS=${CACHED_LOADERS}
export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY=NRL_MXFP8_BATCHED_SHUFFLE,NRL_REFIT_CACHED_LOADERS
EOF
)
fi
printf -v RUN_COMMAND '%q ' uv run --frozen examples/run_grpo.py "${RUN_ARGS[@]}"

if [[ "${ACTION}" == render ]]; then
  if [[ -n "${RUNTIME_TOGGLE_EXPORTS}" ]]; then
    printf '%s\n' "${RUNTIME_TOGGLE_EXPORTS}"
  fi
  printf '%s\n' "${RUN_COMMAND}"
  exit 0
fi

case "${ACTION}" in
  test-only) SBATCH_ACTION=(--test-only) ;;
  submit) SBATCH_ACTION=() ;;
  *) echo "ACTION must be render, test-only, or submit" >&2; exit 2 ;;
esac

BASE=${BASE:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna}
CONTROLLER_REPO=${CONTROLLER_REPO:-${BASE}/RL-gb200-deck-refresh-20260818}
if [[ "${STUDY}" == full_ablation ]]; then
  DEFAULT_TARGET_REPO=${BASE}/RL-gb200-pr3294-ablation-313f
else
  DEFAULT_TARGET_REPO=${BASE}/RL-gb200-pr3478-${ARM}-${TARGET_COMMIT:0:8}
fi
TARGET_REPO=${TARGET_REPO:-${DEFAULT_TARGET_REPO}}
CONTAINER=${CONTAINER:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/mkar/containers/nemo-rl-nightly-ngc-20260815_212622.sqsh}
HF_HOME=${HF_HOME:-${BASE}/hf_home}
ACCOUNT=${SLURM_ACCOUNT:-nemotron_sw_post}
PARTITION=${PARTITION:-batch}
WALLTIME=${WALLTIME:-04:00:00}
RESULT_ROOT=${RESULT_ROOT:-${BASE}/experiments/gb200-deck-refresh/${STUDY}}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-${RESULT_ROOT}/${ARM}/${RUN_SUFFIX}}
CACHE_ROOT=${CACHE_ROOT:-${BASE}/.cache/gb200-deck-refresh/${STUDY}/${ARM}}
WORKER_VENV_ROOT=${WORKER_VENV_ROOT:-/tmp/nemo_rl_worker_venvs/gb200-deck-refresh/${STUDY}/${ARM}/${RUN_SUFFIX}}
RAY_RUNTIME_VENV=${RAY_RUNTIME_VENV:-${BASE}/.cache/gb200-deck-refresh/ray-runtime-py31314}
WANDB_PROJECT=${WANDB_PROJECT:-sna-gb200-${STUDY}}
WANDB_NAME=${WANDB_NAME:-qwen30-sync-${STUDY}-${ARM}-${MAX_STEPS}step-${RUN_SUFFIX}}

git -C "${CONTROLLER_REPO}" pull --ff-only origin sna/exp-gb200-deck-refresh-20260818
test "$(git -C "${TARGET_REPO}" rev-parse HEAD)" = "${TARGET_COMMIT}"
test -z "$(git -C "${TARGET_REPO}" status --porcelain --untracked-files=no)"
git -C "${TARGET_REPO}" submodule update --init --recursive
if git -C "${TARGET_REPO}" submodule status --recursive | grep -q '^-'; then
  echo "All pinned target submodules must be initialized" >&2
  exit 2
fi
for path in "${TARGET_REPO}/${CONFIG}" "${CONTROLLER_REPO}/ray.sub" \
  "${CONTAINER}" "${HF_HOME}" "${RAY_RUNTIME_VENV}/bin/python" \
  "${RAY_RUNTIME_VENV}/bin/ray" "${RAY_RUNTIME_VENV}/READY"; do
  test -e "${path}"
done

SETUP_COMMAND=$(cat <<EOF
set -euo pipefail
rm -f /opt/nemo_rl_venv/bin/ray
ln -s '${RAY_RUNTIME_VENV}/bin/ray' /opt/nemo_rl_venv/bin/ray
'${RAY_RUNTIME_VENV}/bin/python' -c \
  'import ray, requests, urllib3; assert ray.__version__ == "2.56.1"'
ray --version
EOF
)
export SETUP_COMMAND

mkdir -p "${EXPERIMENT_ROOT}" "${CACHE_ROOT}"
WANDB_KEY_FILE=${CACHE_ROOT}/.wandb_key
if [[ -f "${HOME}/.netrc" ]]; then
  umask 077
  awk '
    { for (i = 1; i <= NF; i++) {
        if ($i == "machine" && $(i + 1) == "api.wandb.ai") found = 1
        if (found && $i == "password") { print $(i + 1); exit }
    }}
  ' "${HOME}/.netrc" >"${WANDB_KEY_FILE}"
elif [[ -n "${WANDB_API_KEY:-}" ]]; then
  umask 077
  printf '%s\n' "${WANDB_API_KEY}" >"${WANDB_KEY_FILE}"
fi
test -s "${WANDB_KEY_FILE}"

cat >"${EXPERIMENT_ROOT}/metadata.env" <<EOF
repo_sha=${TARGET_COMMIT}
config=${CONFIG}
container=${CONTAINER}
hardware=GB200
model=Qwen3-30B-A3B
mode=sync_colocated
study=${STUDY}
arm=${ARM}
nodes=4
gpus_per_node=4
vllm=0.25.1
flashinfer=0.6.13
refit_prequantize=${REFIT_PREQUANTIZE}
batched_mxfp8_shuffle=${BATCHED_SHUFFLE}
cached_loaders=${CACHED_LOADERS}
cuda_graphs=enabled
aggregate_steps=3-20
max_steps=${MAX_STEPS}
seed=42
wandb_project=${WANDB_PROJECT}
wandb_name=${WANDB_NAME}
EOF

COMMAND=$(cat <<EOF
set -euo pipefail
cd ${TARGET_REPO}
export HF_HOME=${HF_HOME}
export HF_DATASETS_CACHE=${HF_HOME}/cache
export HUGGINGFACE_HUB_CACHE=${HF_HOME}/hub
export NCCL_NVLS_ENABLE=0
export RAY_CGRAPH_get_timeout=2400
export NRL_FORCE_REBUILD_VENVS=true
export NEMO_RL_VENV_DIR=${WORKER_VENV_ROOT}
export NVTE_CUDA_ARCHS=100
export TORCH_CUDA_ARCH_LIST=10.0
export PYTHONPATH=${TARGET_REPO}
export UV_CACHE_DIR=${CACHE_ROOT}/uv-cache
export UV_PROJECT_ENVIRONMENT=${CACHE_ROOT}/driver-venv
export UV_PYTHON_INSTALL_DIR=${CACHE_ROOT}/uv-python
export UV_LOCK_TIMEOUT=7200
export WANDB_API_KEY="\$(cat ${WANDB_KEY_FILE})"
${RUNTIME_TOGGLE_EXPORTS}
printf 'NEMO_RL_SOURCE_COMMIT=%s\n' "\$(git rev-parse HEAD)"
${RUN_COMMAND} \
  logger.log_dir=${EXPERIMENT_ROOT}/logs \
  logger.wandb_enabled=true \
  logger.tensorboard_enabled=true \
  logger.monitor_gpus=true \
  ++logger.wandb.entity=nvidia \
  logger.wandb.project=${WANDB_PROJECT} \
  logger.wandb.name=${WANDB_NAME}
uv run --frozen tests/json_dump_tb_logs.py ${EXPERIMENT_ROOT}/logs \
  --output_path ${EXPERIMENT_ROOT}/metrics.json || true
EOF
)

export BASE_LOG_DIR=${EXPERIMENT_ROOT}
export COMMAND
export CONTAINER
export CONTAINER_REMAP_ROOT=1
export GPUS_PER_NODE=4
export MOUNTS=/lustre:/lustre
export UV_CACHE_DIR_OVERRIDE=${CACHE_ROOT}/uv-cache

exec sbatch "${SBATCH_ACTION[@]}" \
  --nodes=4 \
  --gres=gpu:4 \
  --exclusive \
  --account="${ACCOUNT}" \
  --partition="${PARTITION}" \
  --time="${WALLTIME}" \
  --segment=4 \
  --job-name="${ACCOUNT}-gb200-deck.${STUDY}-${ARM}" \
  --output="${EXPERIMENT_ROOT}/slurm-%j.out" \
  --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"180","reason":"model_loading","description":"NeMo-RL environment build, model load, FlashInfer autotuning, and CUDA Graph capture"}}' \
  "${CONTROLLER_REPO}/ray.sub"
