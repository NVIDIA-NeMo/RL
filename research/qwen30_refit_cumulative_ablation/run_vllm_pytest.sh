#!/bin/bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bash "${script_dir}/repair_container_python.sh"

for actor_venv in /opt/ray_venvs/*; do
  actor_python="${actor_venv}/bin/python"
  if "${actor_python}" -c 'import vllm' 2>/dev/null; then
    actor_site_packages="${actor_venv}/lib/python3.13/site-packages"
    export PYTHONPATH="${actor_site_packages}${PYTHONPATH:+:${PYTHONPATH}}"
    exec python -m pytest "$@"
  fi
done

echo "No prebuilt actor environment contains vllm." >&2
exit 1
