#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(realpath "${SCRIPT_DIR}/../../..")

cd "${REPO_ROOT}"

if [[ -f "${SCRIPT_DIR}/super.env" ]]; then
  set -a
  source "${SCRIPT_DIR}/super.env"
  set +a
fi

: "${PERSISTENT_CACHE:?PERSISTENT_CACHE is required}"

export HF_HOME="${PERSISTENT_CACHE}/hf_judge_models"
mkdir -p "${HF_HOME}"

python - <<'PY'
from huggingface_hub import snapshot_download

models = [
    # Full-size judges (small_stage1_rlvr_14node.yaml / 21node.yaml)
    "nvidia/Qwen3-Nemotron-235B-A22B-GenRM-2603",
    "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
    # Downsized judges (small_stage1_rlvr_nano30b_4node.yaml)
    "nvidia/Qwen3-Nemotron-32B-GenRM-Principle",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
]

for model in models:
    print(f"Downloading {model}", flush=True)
    snapshot_download(repo_id=model, resume_download=True)
    print(f"Done {model}", flush=True)
PY
