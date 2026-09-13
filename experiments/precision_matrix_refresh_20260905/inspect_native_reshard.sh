#!/usr/bin/env bash
set -euo pipefail
: "${CONTAINER:?}"
: "${RESULT_DIR:?}"
srun --ntasks=1 --kill-on-bad-exit=1 --wait=30 \
  --no-container-mount-home --container-image="${CONTAINER}" \
  --container-mounts="${RESULT_DIR}:/results,/raid/scratch:/raid/scratch" \
  --output="${RESULT_DIR}/native-reshard-%N.log" \
  timeout --signal=TERM --kill-after=30s 5m bash -s <<'CONTAINER_SCRIPT'
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
for role in \
  nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker \
  nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker; do
  PYTHON="/opt/ray_venvs/${role}/bin/python"
  test -x "$PYTHON"
  "$PYTHON" - "$role" <<'PY'
import importlib
import importlib.metadata
import json
import os
import sys
import traceback

result = {
    "role": sys.argv[1],
    "python": sys.executable,
    "versions": {},
    "overrides": {
        key: os.environ.get(key)
        for key in ("NRL_XFERDTENSOR_PYTHON", "NRL_XFERDTENSOR_GOLDEN")
    },
}
for name in ("nccl4py", "cuda-core", "torch", "nvidia-nccl-cu13", "vllm"):
    try:
        result["versions"][name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        result["versions"][name] = None
try:
    module = importlib.import_module("nccl.m2n")
    reshard = getattr(module, "reshard")
    result["native_import"] = "ok"
    result["module_file"] = module.__file__
    result["reshard_callable"] = callable(reshard)
except Exception as error:
    result["native_import"] = "failed"
    result["exception_type"] = type(error).__name__
    result["traceback"] = traceback.format_exc()
print(json.dumps(result, indent=2), flush=True)
PY
done
CONTAINER_SCRIPT
