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
export CONFIG="${CONFIG:-examples/configs/grpo_math_nemotron_diffusion_3b_ar_megatron.yaml}"

export RUN_NAME="${RUN_NAME:-grpo_nemotron_ar_megatron_policy_5k}"
export RUN_ROOT="${RUN_ROOT:-/lustre/fsw/portfolios/coreai/users/snorouzi/runs/diffusion_rl}"
export RUNDIR="${RUNDIR:-${RUN_ROOT}/${RUN_NAME}}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RUNDIR}/checkpoints}"
export LOG_DIR="${LOG_DIR:-/tmp/${RUN_NAME}}"

export MEGATRON_PATCH_DIR="${MEGATRON_PATCH_DIR:-/lustre/fsw/portfolios/coreai/users/snorouzi/nemorl_runtime_patches}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_NAME}}"
export WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-/home/snorouzi/wandb_api_key.txt}"
export NEMO_RL_VENV_DIR="${NEMO_RL_VENV_DIR:-/lustre/fsw/portfolios/coreai/users/snorouzi/nemo_rl_worker_venvs}"
export NRL_FORCE_REBUILD_VENVS="${NRL_FORCE_REBUILD_VENVS:-false}"
export UV_NO_BINARY_PACKAGE="${UV_NO_BINARY_PACKAGE:-}"
export NEMO_RL_SGLANG_KERNEL_SOURCE="${NEMO_RL_SGLANG_KERNEL_SOURCE:-/lustre/fsw/portfolios/coreai/users/snorouzi/wheels/sglang_kernel_torch210_cu129_py313/sglang_kernel-0.4.1-cp310-abi3-linux_x86_64.whl}"
export NEMO_RL_SGLANG_FLASHINFER_SPECS="${NEMO_RL_SGLANG_FLASHINFER_SPECS:-flashinfer_python==0.6.7.post3 flashinfer_cubin==0.6.7.post3}"

run_training() {
  mkdir -p "${RUNDIR}"
  cd "${REPO_DIR}"

  export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/lustre/fsw/portfolios/coreai/users/snorouzi/nemorl_uv_driver_envs/diffusion_RL_RL_mcore}"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-/lustre/fsw/portfolios/coreai/users/snorouzi/uv_cache}"
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

  uv run --reinstall-package nemo-rl --extra mcore --with onnx==1.19.1 python examples/run_grpo.py \
    --config "${CONFIG}" \
    checkpointing.enabled=true \
    "checkpointing.checkpoint_dir=${CHECKPOINT_DIR}" \
    checkpointing.save_optimizer=true \
    checkpointing.save_consolidated=false \
    "logger.log_dir=${LOG_DIR}" \
    "logger.wandb.name=${WANDB_RUN_NAME}" \
    policy.generation.vllm_cfg.enforce_eager=true \
    "policy.megatron_cfg.env_vars={PYTHONPATH:${MEGATRON_PATCH_DIR}}" \
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
export NEMO_RL_VENV_DIR="${NEMO_RL_VENV_DIR}"
export NRL_FORCE_REBUILD_VENVS="${NRL_FORCE_REBUILD_VENVS}"
export UV_NO_BINARY_PACKAGE="${UV_NO_BINARY_PACKAGE}"
export NEMO_RL_SGLANG_KERNEL_SOURCE="${NEMO_RL_SGLANG_KERNEL_SOURCE}"
export NEMO_RL_SGLANG_FLASHINFER_SPECS="${NEMO_RL_SGLANG_FLASHINFER_SPECS}"

srun --kill-on-bad-exit=1 \\
  --container-image="${CONTAINER_IMAGE}" \\
  --container-mounts="${CONTAINER_MOUNTS}" \\
  --container-workdir="${REPO_DIR}" \\
  bash -lc 'cd "\${REPO_DIR}" && tools/nemotron_diffusion/submit_grpo_nemotron_ar_megatron_sbatch.sh --run'
SBATCH
