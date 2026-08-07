#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --segment=1
#SBATCH --time=02:00:00
#SBATCH --job-name=stage_nemo_rl_nightly

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

: "${SOURCE_IMAGE:?Set SOURCE_IMAGE to registry/repository:tag}"
: "${OUTPUT_PREFIX:?Set OUTPUT_PREFIX to a filesystem-safe name}"
: "${CONTAINER_DIR:?Set CONTAINER_DIR to a canonical /lustre path}"
SOURCE_COMMIT=${SOURCE_COMMIT:-unknown}

[[ "$SOURCE_IMAGE" == */* ]] || {
  printf 'ERROR: SOURCE_IMAGE must include a registry and repository\n' >&2
  exit 2
}
[[ "$OUTPUT_PREFIX" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
  printf 'ERROR: OUTPUT_PREFIX is not filesystem-safe\n' >&2
  exit 2
}
resolved_container_dir=$(realpath -m -- "$CONTAINER_DIR")
[[ "$resolved_container_dir" == "$CONTAINER_DIR" && "$CONTAINER_DIR" == /lustre/* ]] || {
  printf 'ERROR: CONTAINER_DIR must be a canonical /lustre path\n' >&2
  exit 2
}

registry=${SOURCE_IMAGE%%/*}
repository_and_tag=${SOURCE_IMAGE#*/}
registry_host=${registry%%:*}
enroot_image_uri="docker://${registry}#${repository_and_tag}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${registry_host},${registry}"
export no_proxy=$NO_PROXY

date_stamp=$(date -u +%Y%m%d)
output_file=$CONTAINER_DIR/${OUTPUT_PREFIX}_${date_stamp}_${SLURM_JOB_ID}.sqsh
metadata_file=$output_file.metadata.txt
current_link=$CONTAINER_DIR/$OUTPUT_PREFIX.sqsh
partial_file=$output_file.partial.$SLURM_JOB_ID
partial_metadata_file=$metadata_file.partial.$SLURM_JOB_ID
partial_current_link=$current_link.partial.$SLURM_JOB_ID

mkdir -p "$CONTAINER_DIR"
export ENROOT_CACHE_PATH=${ENROOT_CACHE_PATH:-$CONTAINER_DIR/.enroot_cache}
export ENROOT_DATA_PATH=${ENROOT_DATA_PATH:-$CONTAINER_DIR/.enroot_data}
mkdir -p "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH"

cleanup() {
  rm -f -- "$partial_file" "$partial_metadata_file" "$partial_current_link"
}
trap cleanup EXIT

[[ ! -e "$output_file" && ! -e "$metadata_file" ]] || {
  printf 'ERROR: immutable output already exists: %s\n' "$output_file" >&2
  exit 1
}
command -v enroot >/dev/null
command -v sha256sum >/dev/null

printf 'source_image=%s\nsource_commit=%s\noutput_file=%s\n' \
  "$SOURCE_IMAGE" "$SOURCE_COMMIT" "$output_file"
enroot import -o "$partial_file" "$enroot_image_uri"
image_sha256=$(sha256sum "$partial_file" | awk '{print $1}')

{
  printf 'source_image=%s\n' "$SOURCE_IMAGE"
  printf 'imported_enroot_uri=%s\n' "$enroot_image_uri"
  printf 'source_commit=%s\n' "$SOURCE_COMMIT"
  printf 'slurm_job_id=%s\n' "$SLURM_JOB_ID"
  printf 'download_host=%s\n' "$(hostname)"
  printf 'downloaded_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'sha256=%s\n' "$image_sha256"
} > "$partial_metadata_file"

mv "$partial_metadata_file" "$metadata_file"
mv "$partial_file" "$output_file"
ln -s "$output_file" "$partial_current_link"
mv -Tf "$partial_current_link" "$current_link"
cleanup
trap - EXIT

printf 'container=%s\nmetadata=%s\n' "$output_file" "$metadata_file"
ls -lh "$output_file" "$metadata_file" "$current_link"
