#!/bin/bash

set -euo pipefail

version=${NRL_NCCL_EXTENSIONS_VERSION:-0.1.0}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/raid/scratch/uv-cache-nccl-extensions}

venvs=(
  /opt/nemo_rl_venv
  /opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker
  /opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker
  /opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker
)

for venv in "${venvs[@]}"; do
  python_path="${venv}/bin/python"
  if [[ ! -x ${python_path} ]]; then
    echo "Missing Python executable: ${python_path}" >&2
    exit 1
  fi

  if ! "${python_path}" -c 'from nccl.m2n import reshard' 2>/dev/null; then
    uv pip install \
      --python "${python_path}" \
      --no-deps \
      "nccl-extensions==${version}"
  fi

  "${python_path}" - <<'PY'
from importlib.metadata import version
from nccl.m2n import reshard

print(f"nccl_extensions={version('nccl-extensions')} reshard={reshard}")
PY
done
