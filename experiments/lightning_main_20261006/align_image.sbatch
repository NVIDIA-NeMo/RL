#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --partition=batch
#SBATCH --time=04:00:00
#SBATCH --mem=0
set -euo pipefail
: "${BASE_CONTAINER:?Set original immutable nightly image}"
: "${SOURCE_ARCHIVE:?Set prepared source archive}"
: "${SOURCE_COMMIT:?Set expected source commit}"
: "${CONTAINER_DIR:?Set shared image output directory}"
export LOCAL_ROOT="/raid/scratch/${USER}/nr-align-${SLURM_JOB_ID}"
output="${CONTAINER_DIR}/nemo_rl_main_aligned_$(date -u +%Y%m%d)_${SLURM_JOB_ID}.sqsh"
test ! -e "$output"
mkdir -p "$CONTAINER_DIR"
grep -Fx "source_commit=${SOURCE_COMMIT}" "${SOURCE_ARCHIVE}.metadata.txt"
archive_hash=$(sha256sum "$SOURCE_ARCHIVE" | cut -d ' ' -f 1)
grep -Fx "${archive_hash}  ${SOURCE_ARCHIVE}" "${SOURCE_ARCHIVE}.metadata.txt"
srun --gres=gpu:4 --mem=0 bash -c '
  set -euo pipefail
  mkdir -p "$LOCAL_ROOT/source" "$LOCAL_ROOT/tmp"
  df -h "$LOCAL_ROOT"
  tar -xf "$SOURCE_ARCHIVE" -C "$LOCAL_ROOT/source"
'
srun --gres=gpu:4 --mem=0 --no-container-mount-home --container-remap-root \
  --container-image="$BASE_CONTAINER" --container-writable \
  --container-mounts="/lustre:/lustre,/raid/scratch:/raid/scratch" \
  --container-save="${output}.partial" \
  bash "${LOCAL_ROOT}/source/experiments/lightning_main_20261006/align_image.sh"
hash=$(sha256sum "${output}.partial" | cut -d ' ' -f 1)
printf 'base_container=%s\nsource_commit=%s\nbuilt_at_utc=%s\nslurm_job_id=%s\nsha256=%s\n' \
  "$BASE_CONTAINER" "$SOURCE_COMMIT" "$(date -u +%FT%TZ)" "$SLURM_JOB_ID" "$hash" > "${output}.metadata.txt"
mv "${output}.partial" "$output"
printf 'validated_aligned_container=%s\n' "$output"
