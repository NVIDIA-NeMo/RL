#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Submit or run GRPO for the Nemotron-Diffusion Ministral-3B AR checkpoint with
# SGLang generation and Megatron policy training.
#
# Usage:
#   --sbatch    Submit via sbatch.
#   --local     Run on the current machine/container.
# If neither flag is passed, defaults to --local.

set -euo pipefail

MODE="local"
for arg in "$@"; do
  case "${arg}" in
    --sbatch) MODE="sbatch" ;;
    --local)  MODE="local" ;;
    --run)    MODE="run" ;;
    *) echo "Unknown flag: ${arg}" >&2; exit 1 ;;
  esac
done

# Slurm/container settings. These are only used by --sbatch.
#ACCOUNT="${ACCOUNT:-coreai_dlalgo_genai}"
export ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}"
export PARTITION="${PARTITION:-batch}"
export TIME="${TIME:-04:00:00}"
export NODES="${NODES:-1}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export CPUS_PER_TASK="${CPUS_PER_TASK:-128}"
export JOB_NAME="${JOB_NAME:-grpo_nd_3b_ar}"
export CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_llm/users/sfawzy/nemo-rl-nightly.sqsh}"
export CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-${HOME}:/home/snorouzi,/lustre:/lustre}"
export DEPENDENCY="${DEPENDENCY:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
export CONFIG="${CONFIG:-examples/configs/gsm8k_nemotron_labs_diffusion_3b_sglang_ar_megatron.yaml}"

export RUN_NAME="${RUN_NAME:-grpo_nemotron_ar_megatron_policy_5k}"
export RUN_ROOT="${RUN_ROOT:-/lustre/fsw/portfolios/coreai/users/snorouzi/runs/diffusion_rl}"
export RUNDIR="${RUNDIR:-${RUN_ROOT}/${RUN_NAME}}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RUNDIR}/checkpoints}"
export LOG_DIR="${LOG_DIR:-${RUNDIR}/logs}"

export MEGATRON_PATCH_DIR="${MEGATRON_PATCH_DIR:-/lustre/fsw/portfolios/coreai/users/snorouzi/nemorl_runtime_patches}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_NAME}}"
export WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-/home/snorouzi/wandb_api_key.txt}"
export ENV_TAG="${ENV_TAG:-gsm8k_nd3b_sglang_a652eb48_mb500dac75}"
export RUST_DIR="${RUST_DIR:-/lustre/fsw/portfolios/coreai/users/snorouzi/rust}"
export RUSTUP_HOME="${RUSTUP_HOME:-${RUST_DIR}/rustup}"
export CARGO_HOME="${CARGO_HOME:-${RUST_DIR}/cargo}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-${RUST_DIR}/target}"
export RUSTUP_TOOLCHAIN="${RUSTUP_TOOLCHAIN:-stable}"
export PROTOC_HOME="${PROTOC_HOME:-/lustre/fsw/portfolios/coreai/users/snorouzi/protoc}"
export PROTOC="${PROTOC:-${PROTOC_HOME}/bin/protoc}"
export PATH="${CARGO_HOME}/bin:${PROTOC_HOME}/bin:${PATH}"
if [[ -z "${NEMO_RL_VENV_DIR:-}" || "${NEMO_RL_VENV_DIR:-}" == "/opt/ray_venvs" ]]; then
  export NEMO_RL_VENV_DIR="/lustre/fsw/portfolios/coreai/users/snorouzi/nemo_rl_worker_venvs_${ENV_TAG}"
fi
if [[ -z "${NRL_FORCE_REBUILD_VENVS+x}" ]]; then
  export NRL_FORCE_REBUILD_VENVS="false"
else
  export NRL_FORCE_REBUILD_VENVS
fi
export FORCE_REINSTALL_PACKAGES="${FORCE_REINSTALL_PACKAGES:-false}"
export FORCE_REINSTALL_NEMO_RL="${FORCE_REINSTALL_NEMO_RL:-${FORCE_REINSTALL_PACKAGES}}"
export FORCE_REINSTALL_SGLANG="${FORCE_REINSTALL_SGLANG:-${FORCE_REINSTALL_PACKAGES}}"
export FORCE_REINSTALL_MEGATRON_BRIDGE="${FORCE_REINSTALL_MEGATRON_BRIDGE:-${FORCE_REINSTALL_PACKAGES}}"
export FORCE_REINSTALL_MEGATRON_CORE="${FORCE_REINSTALL_MEGATRON_CORE:-${FORCE_REINSTALL_PACKAGES}}"
export UV_NO_BINARY_PACKAGE="${UV_NO_BINARY_PACKAGE:-}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export NEMO_RL_SGLANG_KERNEL_SOURCE="${NEMO_RL_SGLANG_KERNEL_SOURCE:-/lustre/fsw/portfolios/coreai/users/snorouzi/wheels/sglang_kernel_torch210_cu129_py313/sglang_kernel-0.4.1-cp310-abi3-linux_x86_64.whl}"
export NEMO_RL_SGLANG_FLASHINFER_SPECS="${NEMO_RL_SGLANG_FLASHINFER_SPECS:-flashinfer_python==0.6.7.post3 flashinfer_cubin==0.6.7.post3}"
export EXTRA_CONFIG_OVERRIDES="${EXTRA_CONFIG_OVERRIDES:-}"

run_training() {
  mkdir -p "${RUNDIR}"
  cd "${REPO_DIR}"

  if [[ -z "${UV_PROJECT_ENVIRONMENT:-}" || "${UV_PROJECT_ENVIRONMENT:-}" == "/opt/nemo_rl_venv" ]]; then
    export UV_PROJECT_ENVIRONMENT="/lustre/fsw/portfolios/coreai/users/snorouzi/nemorl_uv_driver_envs/diffusion_RL_RL_${ENV_TAG}"
  fi
  export UV_CACHE_DIR="${UV_CACHE_DIR:-/lustre/fsw/portfolios/coreai/users/snorouzi/uv_cache_${ENV_TAG}}"
  export NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR:-/lustre/fsw/portfolios/coreai/users/snorouzi/nemo_rl_megatron_ckpts}"
  export MEGATRON_CONFIG_LOCK_DIR="${MEGATRON_CONFIG_LOCK_DIR:-/lustre/fsw/portfolios/coreai/users/snorouzi/hf_home/megatron_config_locks}"
  export HF_HOME="${HF_HOME:-/lustre/fsw/portfolios/coreai/users/snorouzi/hf_home}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/lustre/fsw/portfolios/coreai/users/snorouzi/hf_home/datasets}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/lustre/fsw/portfolios/coreai/users/snorouzi/hf_home/transformers}"

  if [[ -z "${HF_TOKEN:-}" && -f /home/snorouzi/hf_token.txt ]]; then
    export HF_TOKEN
    HF_TOKEN="$(tr -d "\n\r" < /home/snorouzi/hf_token.txt)"
  fi
  export HF_HUB_TOKEN="${HF_HUB_TOKEN:-${HF_TOKEN:-}}"
  export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"

  if [[ -z "${WANDB_API_KEY:-}" && -f "${WANDB_API_KEY_FILE}" ]]; then
    export WANDB_API_KEY
    WANDB_API_KEY="$(tr -d "\n\r" < "${WANDB_API_KEY_FILE}")"
  fi

  export HOME="${CONTAINER_HOME:-/lustre/fsw/portfolios/coreai/users/snorouzi/container_home}"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
  export NRL_IGNORE_VERSION_MISMATCH="${NRL_IGNORE_VERSION_MISMATCH:-1}"
  export RAY_raylet_start_wait_time_s="${RAY_raylet_start_wait_time_s:-120}"
  export NRL_REFIT_BUFFER_MEMORY_RATIO="${NRL_REFIT_BUFFER_MEMORY_RATIO:-0.3}"

  # Do not expose the Megatron bridge patch to the Ray driver. Pass it only to
  # Megatron policy workers via policy.megatron_cfg.env_vars.
  unset PYTHONPATH

  export NEMO_RL_GIT_PATH="${REPO_DIR}"
  export NEMO_RL_GIT_BRANCH="$(git -C "${REPO_DIR}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  export NEMO_RL_GIT_COMMIT="$(git -C "${REPO_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
  export NEMO_RL_GIT_STATUS_COUNT="$(git -C "${REPO_DIR}" status --short 2>/dev/null | wc -l | tr -d ' ')"
  sglang_source_line_with_number="$(grep -n -E 'sglang = .*(git =|path =)' pyproject.toml | head -1 || true)"
  sglang_source_line="${sglang_source_line_with_number#*:}"
  sglang_path="$(printf '%s\n' "${sglang_source_line}" | sed -n 's/.*path = "\([^"]*\)".*/\1/p')"
  export SGLANG_GIT_URL="$(printf '%s\n' "${sglang_source_line}" | sed -n 's/.*git = "\([^"]*\)".*/\1/p')"
  export SGLANG_GIT_COMMIT="$(printf '%s\n' "${sglang_source_line}" | sed -n 's/.*rev = "\([^"]*\)".*/\1/p')"
  export SGLANG_GIT_SUBDIRECTORY="$(printf '%s\n' "${sglang_source_line}" | sed -n 's/.*subdirectory = "\([^"]*\)".*/\1/p')"
  if [[ -n "${sglang_path}" ]]; then
    sglang_repo_path="${sglang_path%/python}"
    export SGLANG_GIT_URL="${sglang_path}"
    export SGLANG_GIT_COMMIT="$(git -C "${sglang_repo_path}" rev-parse HEAD 2>/dev/null || echo unknown)"
    export SGLANG_GIT_SUBDIRECTORY="python"
  fi
  export MEGATRON_BRIDGE_GIT_PATH="$(realpath 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge)"
  export MEGATRON_BRIDGE_GIT_BRANCH="$(git -C "${MEGATRON_BRIDGE_GIT_PATH}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  export MEGATRON_BRIDGE_GIT_COMMIT="$(git -C "${MEGATRON_BRIDGE_GIT_PATH}" rev-parse HEAD 2>/dev/null || echo unknown)"
  export MEGATRON_BRIDGE_GIT_STATUS_COUNT="$(git -C "${MEGATRON_BRIDGE_GIT_PATH}" status --short 2>/dev/null | wc -l | tr -d ' ')"
  export MEGATRON_LM_GIT_PATH="$(realpath 3rdparty/Megatron-LM-workspace/Megatron-LM)"
  export MEGATRON_LM_GIT_BRANCH="$(git -C "${MEGATRON_LM_GIT_PATH}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  export MEGATRON_LM_GIT_COMMIT="$(git -C "${MEGATRON_LM_GIT_PATH}" rev-parse HEAD 2>/dev/null || echo unknown)"
  export MEGATRON_LM_GIT_STATUS_COUNT="$(git -C "${MEGATRON_LM_GIT_PATH}" status --short 2>/dev/null | wc -l | tr -d ' ')"

  echo "SGLANG_SOURCE=${sglang_source_line_with_number}"
  echo "MEGATRON_BRIDGE_WORKSPACE=$(realpath 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge)"
  git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge rev-parse --abbrev-ref HEAD || true
  git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge rev-parse HEAD || true
  echo "MEGATRON_LM_WORKSPACE=$(realpath 3rdparty/Megatron-LM-workspace/Megatron-LM)"
  git -C 3rdparty/Megatron-LM-workspace/Megatron-LM rev-parse --abbrev-ref HEAD || true
  git -C 3rdparty/Megatron-LM-workspace/Megatron-LM rev-parse HEAD || true

  reinstall_args=()
  if [[ "${FORCE_REINSTALL_NEMO_RL}" == "true" || "${FORCE_REINSTALL_NEMO_RL}" == "1" ]]; then
    reinstall_args+=(--reinstall-package nemo-rl)
  fi
  if [[ "${FORCE_REINSTALL_SGLANG}" == "true" || "${FORCE_REINSTALL_SGLANG}" == "1" ]]; then
    reinstall_args+=(--reinstall-package sglang)
  fi
  if [[ "${FORCE_REINSTALL_MEGATRON_BRIDGE}" == "true" || "${FORCE_REINSTALL_MEGATRON_BRIDGE}" == "1" ]]; then
    reinstall_args+=(--reinstall-package megatron-bridge)
  fi
  if [[ "${FORCE_REINSTALL_MEGATRON_CORE}" == "true" || "${FORCE_REINSTALL_MEGATRON_CORE}" == "1" ]]; then
    reinstall_args+=(--reinstall-package megatron-core)
  fi

  extra_config_overrides=()
  if [[ -n "${EXTRA_CONFIG_OVERRIDES}" ]]; then
    read -r -a extra_config_overrides <<< "${EXTRA_CONFIG_OVERRIDES}"
  fi

  uv run \
    "${reinstall_args[@]}" \
    --extra mcore \
    --with onnx==1.19.1 \
    python examples/run_grpo.py \
    --config "${CONFIG}" \
    checkpointing.enabled=true \
    "checkpointing.checkpoint_dir=${CHECKPOINT_DIR}" \
    checkpointing.save_optimizer=true \
    checkpointing.save_consolidated=false \
    "logger.log_dir=${LOG_DIR}" \
    "logger.wandb.name=${WANDB_RUN_NAME}" \
    policy.generation.vllm_cfg.enforce_eager=true \
    "policy.megatron_cfg.env_vars={PYTHONPATH:${MEGATRON_PATCH_DIR}}" \
    "${extra_config_overrides[@]}" \
    2>&1 | tee -a "${RUNDIR}/run.log"
}

if [[ "${MODE}" == "local" || "${MODE}" == "run" ]]; then
  run_training
  exit 0
fi

mkdir -p "${RUNDIR}"

SBATCH_ARGS=(
  --account="${ACCOUNT}"
  --partition="${PARTITION}"
  --time="${TIME}"
  --nodes="${NODES}"
  --gres="gpu:${GPUS_PER_NODE}"
  --cpus-per-task="${CPUS_PER_TASK}"
  --job-name="${JOB_NAME}"
  --output="${RUNDIR}/slurm-%j.out"
  --open-mode=append
)

if [[ -n "${DEPENDENCY}" ]]; then
  SBATCH_ARGS+=(--dependency="${DEPENDENCY}")
fi

if [[ "${NODES}" -gt 1 ]]; then
  # NeMo RL ray.sub owns multi-node Ray bringup. This script only defines
  # the driver command and environment used once ray.sub has started the cluster.
  export COMMAND="${COMMAND:-cd ${REPO_DIR} && tools/nemotron_diffusion/submit_grpo_nemotron_ar_megatron_sbatch.sh --run}"
  export CONTAINER="${CONTAINER:-${CONTAINER_IMAGE}}"
  export MOUNTS="${MOUNTS:-${CONTAINER_MOUNTS}}"
  export CPUS_PER_WORKER="${CPUS_PER_WORKER:-${CPUS_PER_TASK}}"
  export BASE_LOG_DIR="${BASE_LOG_DIR:-${RUNDIR}}"
  export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/lustre/fsw/portfolios/coreai/users/snorouzi/nemorl_uv_driver_envs/diffusion_RL_RL_${ENV_TAG}}"
  export UV_CACHE_DIR_OVERRIDE="${UV_CACHE_DIR_OVERRIDE:-/lustre/fsw/portfolios/coreai/users/snorouzi/uv_cache_${ENV_TAG}}"
  export RAY_raylet_start_wait_time_s="${RAY_raylet_start_wait_time_s:-240}"
  export RAY_START_CMD="${RAY_START_CMD:-RAY_raylet_start_wait_time_s=${RAY_raylet_start_wait_time_s} UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT} UV_CACHE_DIR=/root/.cache/uv uv run --locked --directory ${REPO_DIR} ray start}"
  export RAY_STATUS_CMD="${RAY_STATUS_CMD:-UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT} UV_CACHE_DIR=/root/.cache/uv uv run --locked --directory ${REPO_DIR} ray status}"

  cd "${REPO_DIR}"
  sbatch "${SBATCH_ARGS[@]}" ray.sub
  exit 0
fi

sbatch "${SBATCH_ARGS[@]}" <<SBATCH
#!/usr/bin/env bash
set -euo pipefail

export REPO_DIR="${REPO_DIR}"
export CONFIG="${CONFIG}"
export RUN_NAME="${RUN_NAME}"
export RUN_ROOT="${RUN_ROOT}"
export RUNDIR="${RUNDIR}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR}"
export LOG_DIR="${LOG_DIR}"
export MEGATRON_PATCH_DIR="${MEGATRON_PATCH_DIR}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME}"
export WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE}"
export ENV_TAG="${ENV_TAG}"
export RUST_DIR="${RUST_DIR}"
export RUSTUP_HOME="${RUSTUP_HOME}"
export CARGO_HOME="${CARGO_HOME}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR}"
export RUSTUP_TOOLCHAIN="${RUSTUP_TOOLCHAIN}"
export PROTOC_HOME="${PROTOC_HOME}"
export PROTOC="${PROTOC}"
export PATH="${CARGO_HOME}/bin:${PROTOC_HOME}/bin:${PATH}"
export NEMO_RL_VENV_DIR="${NEMO_RL_VENV_DIR}"
export NRL_FORCE_REBUILD_VENVS="${NRL_FORCE_REBUILD_VENVS}"
export FORCE_REINSTALL_PACKAGES="${FORCE_REINSTALL_PACKAGES}"
export FORCE_REINSTALL_NEMO_RL="${FORCE_REINSTALL_NEMO_RL}"
export FORCE_REINSTALL_SGLANG="${FORCE_REINSTALL_SGLANG}"
export FORCE_REINSTALL_MEGATRON_BRIDGE="${FORCE_REINSTALL_MEGATRON_BRIDGE}"
export FORCE_REINSTALL_MEGATRON_CORE="${FORCE_REINSTALL_MEGATRON_CORE}"
export UV_NO_BINARY_PACKAGE="${UV_NO_BINARY_PACKAGE}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT}"
export NEMO_RL_SGLANG_KERNEL_SOURCE="${NEMO_RL_SGLANG_KERNEL_SOURCE}"
export NEMO_RL_SGLANG_FLASHINFER_SPECS="${NEMO_RL_SGLANG_FLASHINFER_SPECS}"
export EXTRA_CONFIG_OVERRIDES="${EXTRA_CONFIG_OVERRIDES}"
export NEMO_RL_GIT_PATH="${NEMO_RL_GIT_PATH:-}"
export NEMO_RL_GIT_BRANCH="${NEMO_RL_GIT_BRANCH:-}"
export NEMO_RL_GIT_COMMIT="${NEMO_RL_GIT_COMMIT:-}"
export NEMO_RL_GIT_STATUS_COUNT="${NEMO_RL_GIT_STATUS_COUNT:-}"
export SGLANG_GIT_URL="${SGLANG_GIT_URL:-}"
export SGLANG_GIT_COMMIT="${SGLANG_GIT_COMMIT:-}"
export SGLANG_GIT_SUBDIRECTORY="${SGLANG_GIT_SUBDIRECTORY:-}"
export MEGATRON_BRIDGE_GIT_PATH="${MEGATRON_BRIDGE_GIT_PATH:-}"
export MEGATRON_BRIDGE_GIT_BRANCH="${MEGATRON_BRIDGE_GIT_BRANCH:-}"
export MEGATRON_BRIDGE_GIT_COMMIT="${MEGATRON_BRIDGE_GIT_COMMIT:-}"
export MEGATRON_BRIDGE_GIT_STATUS_COUNT="${MEGATRON_BRIDGE_GIT_STATUS_COUNT:-}"
export MEGATRON_LM_GIT_PATH="${MEGATRON_LM_GIT_PATH:-}"
export MEGATRON_LM_GIT_BRANCH="${MEGATRON_LM_GIT_BRANCH:-}"
export MEGATRON_LM_GIT_COMMIT="${MEGATRON_LM_GIT_COMMIT:-}"
export MEGATRON_LM_GIT_STATUS_COUNT="${MEGATRON_LM_GIT_STATUS_COUNT:-}"

srun --kill-on-bad-exit=1 \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${CONTAINER_MOUNTS}" \
  --container-workdir="${REPO_DIR}" \
  bash -lc 'cd "${REPO_DIR}" && tools/nemotron_diffusion/submit_grpo_nemotron_ar_megatron_sbatch.sh --run'
SBATCH
