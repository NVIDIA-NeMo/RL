#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-test-only}"
SELECTION="${2:-all}"

REPO_DIR="${REPO_DIR:-$(git rev-parse --show-toplevel)}"
ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}"
PARTITION="${PARTITION:-batch}"
CONTAINER="${CONTAINER:-/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/nemo_rl_nightly_20260705.sqsh}"
HF_HOME="${HF_HOME:-/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home}"
WANDB_PROJECT="${WANDB_PROJECT:-nemorl-vllm024-perfcfg-ptyche}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-/lustre/fsw/coreai_dlalgo_llm/users/sna/.secrets/wandb_api_key}"
RUN_TAG="${RUN_TAG:-vllm024-perfcfg-step10-20260707}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${REPO_DIR}/experiments/vllm_024_upgrade/runs/${RUN_TAG}}"
WALLTIME="${WALLTIME:-04:00:00}"

if [[ "${MODE}" != "dry-run" && ! -f "${CONTAINER}" ]]; then
  echo "ERROR: container not found: ${CONTAINER}" >&2
  exit 2
fi

if [[ "${MODE}" == "submit" && -z "${WANDB_API_KEY:-}" ]]; then
  if [[ ! -r "${WANDB_API_KEY_FILE}" ]]; then
    echo "ERROR: set WANDB_API_KEY or create ${WANDB_API_KEY_FILE}" >&2
    exit 2
  fi
  WANDB_API_KEY="$(<"${WANDB_API_KEY_FILE}")"
  export WANDB_API_KEY
fi

case "${SELECTION}" in
  all) labels=(qwen30ba3b qwen32b qwen235b) ;;
  qwen30ba3b|qwen32b|qwen235b) labels=("${SELECTION}") ;;
  *)
    echo "ERROR: selection must be all, qwen30ba3b, qwen32b, or qwen235b" >&2
    exit 2
    ;;
esac

submit_one() {
  local label="$1"
  local config
  local num_nodes
  case "${label}" in
    qwen30ba3b)
      config="examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml"
      num_nodes=4
      ;;
    qwen32b)
      config="examples/configs/recipes/llm/performance/grpo-qwen3-32b-4n4g.yaml"
      num_nodes=4
      ;;
    qwen235b)
      config="examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n4g.yaml"
      num_nodes=16
      ;;
  esac
  local segment="${num_nodes}"
  local run_dir="${EXPERIMENT_ROOT}/${label}"
  local wandb_name="${RUN_TAG}-${label}-cg-on"
  local command

  printf -v command '%q ' \
    /opt/nemo_rl_venv/bin/python \
    examples/run_grpo.py \
    --config "${config}" \
    grpo.max_num_steps=10 \
    checkpointing.enabled=false \
    "checkpointing.checkpoint_dir=${run_dir}/checkpoints" \
    policy.generation.vllm_cfg.enforce_eager=false \
    "cluster.segment_size=${segment}" \
    logger.wandb_enabled=true \
    "logger.wandb.project=${WANDB_PROJECT}" \
    "logger.wandb.name=${wandb_name}" \
    "logger.log_dir=${run_dir}/nemo_logs"
  command="${command% }"

  local environment=(
    "CONTAINER=${CONTAINER}"
    "MOUNTS=/lustre:/lustre"
    "COMMAND=${command}"
    "BASE_LOG_DIR=${run_dir}"
    "GPUS_PER_NODE=4"
    "HF_HOME=${HF_HOME}"
    "NEMO_RL_VENV_DIR=${run_dir}/venvs"
    "NRL_FORCE_REBUILD_VENVS=true"
    "PYTHONDONTWRITEBYTECODE=1"
    "RAY_LOG_SYNC_FREQUENCY=60"
  )
  local sbatch_args=(
    --account="${ACCOUNT}"
    --partition="${PARTITION}"
    --nodes="${num_nodes}"
    --ntasks-per-node=1
    --exclusive
    --time="${WALLTIME}"
    --segment="${segment}"
    --job-name="${ACCOUNT}-nemorl.v024-${label}"
    --output="${run_dir}/slurm-%j.out"
    --comment=metrics
  )

  case "${MODE}" in
    dry-run)
      printf '[DRY-RUN] env'
      printf ' %q' "${environment[@]}"
      printf ' sbatch'
      printf ' %q' "${sbatch_args[@]}"
      printf ' %q\n' "${REPO_DIR}/ray.sub"
      ;;
    test-only)
      mkdir -p "${run_dir}"
      env "${environment[@]}" sbatch --test-only "${sbatch_args[@]}" "${REPO_DIR}/ray.sub"
      ;;
    submit)
      mkdir -p "${run_dir}"
      local job_id
      job_id="$(env "${environment[@]}" sbatch --parsable "${sbatch_args[@]}" "${REPO_DIR}/ray.sub")"
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(date --iso-8601=seconds)" "${label}" "${job_id}" "${num_nodes}" \
        "${segment}" "$(git -C "${REPO_DIR}" rev-parse HEAD)" \
        | tee -a "${EXPERIMENT_ROOT}/submissions.tsv"
      ;;
    *)
      echo "ERROR: mode must be dry-run, test-only, or submit" >&2
      exit 2
      ;;
  esac
}

if [[ "${MODE}" != "dry-run" ]]; then
  mkdir -p "${EXPERIMENT_ROOT}"
fi
for label in "${labels[@]}"; do
  submit_one "${label}"
done
