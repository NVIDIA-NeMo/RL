#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

extra=()
if [[ -n "${SBATCH_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra=(${SBATCH_EXTRA_ARGS})
fi

echo "Submitting all seed jobs..."
job_id=$(sbatch --parsable "${extra[@]}" "submit_eagle_tardrift_seed123.sbatch")
echo "submitted submit_eagle_tardrift_seed123.sbatch -> $job_id"
job_id=$(sbatch --parsable "${extra[@]}" "submit_eagle_tardrift_seed124.sbatch")
echo "submitted submit_eagle_tardrift_seed124.sbatch -> $job_id"
job_id=$(sbatch --parsable "${extra[@]}" "submit_eagle_tardrift_seed125.sbatch")
echo "submitted submit_eagle_tardrift_seed125.sbatch -> $job_id"
job_id=$(sbatch --parsable "${extra[@]}" "submit_eagle_tardrift_seed126.sbatch")
echo "submitted submit_eagle_tardrift_seed126.sbatch -> $job_id"
job_id=$(sbatch --parsable "${extra[@]}" "submit_eagle_tardrift_seed127.sbatch")
echo "submitted submit_eagle_tardrift_seed127.sbatch -> $job_id"
echo "Done."
