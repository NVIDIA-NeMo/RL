#!/bin/bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bash "${script_dir}/repair_container_python.sh"

python - <<'PY'
import torch
import nemo_rl
import nemo_rl.distributed.ray_actor_environment_registry

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 4
print(
    {
        "torch": torch.__version__,
        "gpu_count": torch.cuda.device_count(),
        "gpu": torch.cuda.get_device_name(0),
    }
)
PY

found_vllm=0
found_mcore=0
for actor_venv in /opt/ray_venvs/*; do
  actor_python="${actor_venv}/bin/python"
  if [[ ${found_vllm} -eq 0 ]] && "${actor_python}" -c 'import vllm' 2>/dev/null; then
    echo "vllm_venv=${actor_venv}"
    "${actor_python}" -c 'import vllm; print(vllm.__version__)'
    found_vllm=1
  fi
  if [[ ${found_mcore} -eq 0 ]] && "${actor_python}" -c 'import megatron.core, transformer_engine.pytorch' 2>/dev/null; then
    echo "mcore_venv=${actor_venv}"
    found_mcore=1
  fi
done

if [[ ${found_vllm} -ne 1 || ${found_mcore} -ne 1 ]]; then
  echo "Missing required actor environment: vllm=${found_vllm}, mcore=${found_mcore}" >&2
  exit 1
fi
