#!/bin/bash

set -euo pipefail

uv python install 3.13
python_path=$(uv python find 3.13)

if [[ ! -x "${python_path}" ]]; then
  echo "uv did not provide an executable Python: ${python_path}" >&2
  exit 1
fi

ln -sf "${python_path}" /opt/nemo_rl_venv/bin/python

shopt -s nullglob
actor_venvs=(/opt/ray_venvs/*)
for actor_venv in "${actor_venvs[@]}"; do
  if [[ -d "${actor_venv}/bin" ]]; then
    ln -sf "${python_path}" "${actor_venv}/bin/python"
  fi
done

python --version
