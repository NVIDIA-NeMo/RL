#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=cpu
#SBATCH --qos=cpu-normal
#SBATCH --time=02:00:00
#SBATCH --mem=0
set -euo pipefail
: "${CONTAINER_DIR:?Set shared image directory}"
: "${SOURCE_COMMIT:?Set executed source commit}"
SOURCE_IMAGE=${SOURCE_IMAGE:-nvcr.io/nvidian/nemo-rl:nightly}
export ENROOT_CACHE_PATH="/raid/scratch/${USER}/nr-stage-${SLURM_JOB_ID}/cache"
export ENROOT_DATA_PATH="/raid/scratch/${USER}/nr-stage-${SLURM_JOB_ID}/data"
export TMPDIR="/raid/scratch/${USER}/nr-stage-${SLURM_JOB_ID}/tmp"
mkdir -p "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH" "$TMPDIR" "$CONTAINER_DIR"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}nvcr.io"
export no_proxy="$NO_PROXY"
output="${CONTAINER_DIR}/nemo_rl_nightly_20261006_${SLURM_JOB_ID}.sqsh"
test ! -e "$output"
trap 'rm -f "${output}.partial"' EXIT
enroot import -o "${output}.partial" "docker://${SOURCE_IMAGE%%/*}#${SOURCE_IMAGE#*/}"
hash=$(sha256sum "${output}.partial" | cut -d ' ' -f 1)
printf 'source_image=%s\nsource_commit=%s\ndownloaded_at_utc=%s\nslurm_job_id=%s\nsha256=%s\n' \
  "$SOURCE_IMAGE" "$SOURCE_COMMIT" "$(date -u +%FT%TZ)" "$SLURM_JOB_ID" "$hash" > "${output}.metadata.txt"
mv "${output}.partial" "$output"
printf 'container=%s\n' "$output"
