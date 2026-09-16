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
import importlib.util
import sys

print(f"executable={sys.executable}")
print(f"prefix={sys.prefix}")
for module in ("pytest", "vllm", "transformer_engine", "megatron"):
    print(f"{module}={importlib.util.find_spec(module)}")
PY
done
