#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

extra=()
if [[ -n "${SBATCH_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra=(${SBATCH_EXTRA_ARGS})
fi

export_vars="${SBATCH_EXPORT_VARS:-ALL,NEMO_RL_PY_EXECUTABLES_SYSTEM=0,NRL_FORCE_REBUILD_VENVS=true,RAY_ENABLE_UV_RUN_RUNTIME_ENV=0}"

echo "Submitting TRAIN phase for all seeds..."
job_id=$(sbatch --parsable "${extra[@]}" --export="${export_vars}" "submit_eagle_tardrift_seed123.sbatch" train)
echo "train submit_eagle_tardrift_seed123.sbatch -> $job_id"
job_id=$(sbatch --parsable "${extra[@]}" --export="${export_vars}" "submit_eagle_tardrift_seed124.sbatch" train)
echo "train submit_eagle_tardrift_seed124.sbatch -> $job_id"
job_id=$(sbatch --parsable "${extra[@]}" --export="${export_vars}" "submit_eagle_tardrift_seed125.sbatch" train)
echo "train submit_eagle_tardrift_seed125.sbatch -> $job_id"
job_id=$(sbatch --parsable "${extra[@]}" --export="${export_vars}" "submit_eagle_tardrift_seed126.sbatch" train)
echo "train submit_eagle_tardrift_seed126.sbatch -> $job_id"
job_id=$(sbatch --parsable "${extra[@]}" --export="${export_vars}" "submit_eagle_tardrift_seed127.sbatch" train)
echo "train submit_eagle_tardrift_seed127.sbatch -> $job_id"
echo "Done."
