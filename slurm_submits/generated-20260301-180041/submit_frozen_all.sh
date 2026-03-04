#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

extra=()
if [[ -n "${SBATCH_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra=(${SBATCH_EXTRA_ARGS})
fi

echo "Submitting FROZEN phase for all seeds..."
job_id=$(sbatch --parsable "${extra[@]}" "submit_eagle_tardrift_seed123.sbatch" frozen)
echo "frozen submit_eagle_tardrift_seed123.sbatch -> $job_id"
job_id=$(sbatch --parsable "${extra[@]}" "submit_eagle_tardrift_seed124.sbatch" frozen)
echo "frozen submit_eagle_tardrift_seed124.sbatch -> $job_id"
job_id=$(sbatch --parsable "${extra[@]}" "submit_eagle_tardrift_seed125.sbatch" frozen)
echo "frozen submit_eagle_tardrift_seed125.sbatch -> $job_id"
job_id=$(sbatch --parsable "${extra[@]}" "submit_eagle_tardrift_seed126.sbatch" frozen)
echo "frozen submit_eagle_tardrift_seed126.sbatch -> $job_id"
job_id=$(sbatch --parsable "${extra[@]}" "submit_eagle_tardrift_seed127.sbatch" frozen)
echo "frozen submit_eagle_tardrift_seed127.sbatch -> $job_id"
echo "Done."
