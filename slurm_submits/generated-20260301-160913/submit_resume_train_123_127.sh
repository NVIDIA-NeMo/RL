#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

: "${RESUME_CKPT_DIR_123:?Set RESUME_CKPT_DIR_123 to a checkpoint dir containing step_*}"
: "${RESUME_CKPT_DIR_127:?Set RESUME_CKPT_DIR_127 to a checkpoint dir containing step_*}"

extra=()
if [[ -n "${SBATCH_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra=(${SBATCH_EXTRA_ARGS})
fi

base_export_vars="${SBATCH_EXPORT_VARS:-ALL,NEMO_RL_PY_EXECUTABLES_SYSTEM=1,NRL_FORCE_REBUILD_VENVS=false,RAY_ENABLE_UV_RUN_RUNTIME_ENV=0}"

submit_resume_seed() {
  local seed="$1"
  local ckpt_dir="$2"
  local script="submit_eagle_tardrift_seed${seed}.sbatch"
  local export_vars="${base_export_vars},RESUME_CHECKPOINT_DIR=${ckpt_dir}"
  local job_id=""

  if [[ ! -d "$ckpt_dir" ]]; then
    echo "ERROR: checkpoint dir does not exist for seed ${seed}: ${ckpt_dir}" >&2
    exit 2
  fi

  job_id=$(sbatch --parsable "${extra[@]}" --export="${export_vars}" "$script" train)
  echo "resume train ${script} -> ${job_id} (checkpoint_dir=${ckpt_dir})"
}

echo "Submitting RESUME TRAIN phase for seeds 123 and 127..."
submit_resume_seed 123 "$RESUME_CKPT_DIR_123"
submit_resume_seed 127 "$RESUME_CKPT_DIR_127"
echo "Done."
