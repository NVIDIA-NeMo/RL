#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

extra=()
if [[ -n "${SBATCH_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra=(${SBATCH_EXTRA_ARGS})
fi

echo "Submitting TRAIN phase for all seeds..."
job_id=$(sbatch --parsable "${extra[@]}" "submit_eagle_tardrift_seed123.sbatch" train)
echo "train submit_eagle_tardrift_seed123.sbatch -> $job_id"
job_id=$(sbatch --parsable "${extra[@]}" "submit_eagle_tardrift_seed124.sbatch" train)
echo "train submit_eagle_tardrift_seed124.sbatch -> $job_id"
job_id=$(sbatch --parsable "${extra[@]}" "submit_eagle_tardrift_seed125.sbatch" train)
echo "train submit_eagle_tardrift_seed125.sbatch -> $job_id"
job_id=$(sbatch --parsable "${extra[@]}" "submit_eagle_tardrift_seed126.sbatch" train)
echo "train submit_eagle_tardrift_seed126.sbatch -> $job_id"
job_id=$(sbatch --parsable "${extra[@]}" "submit_eagle_tardrift_seed127.sbatch" train)
echo "train submit_eagle_tardrift_seed127.sbatch -> $job_id"
echo "Done."
