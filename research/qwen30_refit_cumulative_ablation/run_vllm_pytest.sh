#!/bin/bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bash "${script_dir}/repair_container_python.sh"

for actor_venv in /opt/ray_venvs/*; do
  actor_python="${actor_venv}/bin/python"
  if "${actor_python}" -c 'import pytest, vllm' 2>/dev/null; then
    exec "${actor_python}" -m pytest "$@"
  fi
done

echo "No prebuilt actor environment contains both pytest and vllm." >&2
exit 1
