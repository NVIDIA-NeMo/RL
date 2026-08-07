#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

set -euo pipefail

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
require_canonical_lustre_path() {
  local resolved
  resolved=$(realpath -m -- "$2") || die "$1 is invalid: $2"
  [[ "$resolved" == "$2" && "$resolved" == /lustre/* ]] || \
    die "$1 must be a canonical /lustre path: $2"
}

: "${CONTAINER:?CONTAINER is required}"
: "${HF_HOME:?HF_HOME is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"

MODEL_ID=Qwen/Qwen3-30B-A3B
MODEL_REVISION=${MODEL_REVISION:-main}
require_canonical_lustre_path CONTAINER "$CONTAINER"
require_canonical_lustre_path HF_HOME "$HF_HOME"
require_canonical_lustre_path OUTPUT_ROOT "$OUTPUT_ROOT"
[[ -f "$CONTAINER" ]] || die "container is missing: $CONTAINER"
mkdir -p "$HF_HOME" "$OUTPUT_ROOT"

export HF_HOME
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_DISABLE_TELEMETRY=1
export METADATA_PATH=$OUTPUT_ROOT/stage-metadata-$SLURM_JOB_ID.json
export MODEL_ID MODEL_REVISION OUTPUT_ROOT METADATA_PATH
MOUNTS="$HF_HOME:$HF_HOME,$OUTPUT_ROOT:$OUTPUT_ROOT"
CONTAINER_ENV=HF_HOME,HF_HUB_CACHE,HF_HUB_DISABLE_TELEMETRY,MODEL_ID,MODEL_REVISION,OUTPUT_ROOT,METADATA_PATH,SLURM_JOB_ID

srun --no-container-mount-home \
  --container-image="$CONTAINER" \
  --container-mounts="$MOUNTS" \
  --container-env="$CONTAINER_ENV" \
  --container-workdir="$OUTPUT_ROOT" \
  --nodes=1 --ntasks=1 \
  /opt/nemo_rl_venv/bin/python - <<'PY'
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download

snapshot_path = Path(
    snapshot_download(
        repo_id=os.environ["MODEL_ID"],
        revision=os.environ["MODEL_REVISION"],
        cache_dir=os.environ["HF_HUB_CACHE"],
        max_workers=16,
    )
).resolve()
files = [path for path in snapshot_path.rglob("*") if path.is_file()]
metadata = {
    "file_count": len(files),
    "logical_bytes": sum(path.stat().st_size for path in files),
    "model_id": os.environ["MODEL_ID"],
    "requested_revision": os.environ["MODEL_REVISION"],
    "snapshot_path": str(snapshot_path),
}
metadata_path = Path(os.environ["METADATA_PATH"])
metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
print(json.dumps(metadata, sort_keys=True), flush=True)
PY
