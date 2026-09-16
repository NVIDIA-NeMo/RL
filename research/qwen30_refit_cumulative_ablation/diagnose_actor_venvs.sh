#!/bin/bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bash "${script_dir}/repair_container_python.sh"

ls -la /opt/ray_venvs
find /opt/ray_venvs -maxdepth 3 -name pyvenv.cfg -print

for actor_venv in /opt/ray_venvs/*; do
  actor_python="${actor_venv}/bin/python"
  echo "actor_venv=${actor_venv}"
  if [[ ! -x "${actor_python}" ]]; then
    echo "python=missing"
    continue
  fi
  "${actor_python}" - <<'PY'
import importlib.metadata
import importlib.util
import sys

print(f"executable={sys.executable}")
print(f"prefix={sys.prefix}")
for distribution in ("nvidia-nccl-cu13", "nvidia-nccl-cu12", "nccl-extensions"):
    try:
        print(f"{distribution}={importlib.metadata.version(distribution)}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{distribution}=missing")

for module in ("pytest", "vllm", "transformer_engine", "megatron", "nccl"):
    print(f"{module}={importlib.util.find_spec(module)}")

try:
    from nccl.m2n import reshard
except Exception as error:
    print(f"nccl.m2n.reshard=unavailable:{type(error).__name__}:{error}")
else:
    print(f"nccl.m2n.reshard={reshard}")
PY
done
