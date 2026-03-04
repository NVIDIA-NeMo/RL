#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "Submitting all seed jobs..."
job_id=$(sbatch --parsable "submit_eagle_tardrift_seed123.sbatch")
echo "submitted submit_eagle_tardrift_seed123.sbatch -> $job_id"
job_id=$(sbatch --parsable "submit_eagle_tardrift_seed124.sbatch")
echo "submitted submit_eagle_tardrift_seed124.sbatch -> $job_id"
job_id=$(sbatch --parsable "submit_eagle_tardrift_seed125.sbatch")
echo "submitted submit_eagle_tardrift_seed125.sbatch -> $job_id"
job_id=$(sbatch --parsable "submit_eagle_tardrift_seed126.sbatch")
echo "submitted submit_eagle_tardrift_seed126.sbatch -> $job_id"
job_id=$(sbatch --parsable "submit_eagle_tardrift_seed127.sbatch")
echo "submitted submit_eagle_tardrift_seed127.sbatch -> $job_id"
echo "Done."
