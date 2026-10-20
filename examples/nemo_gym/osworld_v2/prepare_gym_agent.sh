#!/bin/bash
set -euo pipefail

: "${OSWORLD_COMPONENT_PYTHON:?}"
: "${OSWORLD_GYM_ROOT:?}"
: "${NEMO_GYM_VENV_DIR:?}"

agent_dir="$OSWORLD_GYM_ROOT/responses_api_agents/osworld_agent"
agent_venv="$NEMO_GYM_VENV_DIR/responses_api_agents/osworld_agent/.venv"
agent_python="$agent_venv/bin/python"
model_dir="$OSWORLD_GYM_ROOT/responses_api_models/vllm_model"
model_venv="$NEMO_GYM_VENV_DIR/responses_api_models/vllm_model/.venv"
model_python="$model_venv/bin/python"
environment_dir="$OSWORLD_GYM_ROOT/environment_servers/legacy_agent"
environment_venv="$NEMO_GYM_VENV_DIR/environment_servers/legacy_agent/.venv"
environment_python="$environment_venv/bin/python"
driver_ray_version="$("$OSWORLD_DRIVER_PYTHON" -c 'import ray; print(ray.__version__)')"
setup_helper="$OSWORLD_GYM_ROOT/nemo_gym/cli/_venv_setup.py"
runtime_checker="$agent_dir/runtime_dependencies.py"
runtime_installer="$agent_dir/install_optional_runtime_deps.sh"

test -r "$agent_dir/requirements.txt"
test -r "$agent_dir/overrides.txt"
test -r "$agent_dir/uv.toml"
test -r "$model_dir/pyproject.toml"
test -r "$environment_dir/requirements.txt"
test -r "$setup_helper"
test -r "$runtime_checker"
test -x "$runtime_installer"

setup_command="$(cat <<EOF
set -euo pipefail
agent_dir=$(printf '%q' "$agent_dir")
agent_venv=$(printf '%q' "$agent_venv")
driver_ray_version=$(printf '%q' "$driver_ray_version")
agent_python=\$agent_venv/bin/python
if [[ ${OSWORLD_REFRESH_GYM_VENVS:-0} == 1 ]] \
  || [[ ! -x "\$agent_python" ]] \
  || ! "\$agent_python" -c 'import ray, sys; assert ray.__version__ == sys.argv[1], (ray.__version__, sys.argv[1])' "\$driver_ray_version"; then
  cd "\$agent_dir"
  uv venv --seed --allow-existing --python $(printf '%q' "$OSWORLD_COMPONENT_PYTHON") "\$agent_venv"
  source "\$agent_venv/bin/activate"
  uv pip install --config-file "\$agent_dir/uv.toml" --override "\$agent_dir/overrides.txt" -r "\$agent_dir/requirements.txt" "ray==\$driver_ray_version"
fi
bash $(printf '%q' "$runtime_installer") "\$agent_venv"
EOF
)"

"$OSWORLD_COMPONENT_PYTHON" "$setup_helper" \
  --venv "$agent_venv" \
  --command "$setup_command"

"$agent_python" - "$driver_ray_version" <<'PY'
import ray
import sys

if ray.__version__ != sys.argv[1]:
    raise RuntimeError(
        f"OSWorld agent Ray {ray.__version__} does not match cluster Ray {sys.argv[1]}"
    )
print("osworld-agent-ray-ready", ray.__version__)
PY
"$agent_python" "$runtime_checker" check

model_setup_command="$(cat <<EOF
set -euo pipefail
model_dir=$(printf '%q' "$model_dir")
model_venv=$(printf '%q' "$model_venv")
model_python=\$model_venv/bin/python
if [[ ${OSWORLD_REFRESH_GYM_VENVS:-0} == 1 ]] \
  || [[ ! -x "\$model_python" ]] \
  || ! "\$model_python" -c 'import aiohttp, fastapi, nemo_gym'; then
  cd "\$model_dir"
  uv venv --seed --allow-existing --python $(printf '%q' "$OSWORLD_COMPONENT_PYTHON") "\$model_venv"
  uv pip install --python "\$model_python" '-e .'
fi
EOF
)"

"$OSWORLD_COMPONENT_PYTHON" "$setup_helper" \
  --venv "$model_venv" \
  --command "$model_setup_command"
"$model_python" -c 'import aiohttp, fastapi, nemo_gym; print("osworld-policy-model-venv-ready")'

environment_setup_command="$(cat <<EOF
set -euo pipefail
environment_dir=$(printf '%q' "$environment_dir")
environment_venv=$(printf '%q' "$environment_venv")
environment_python=\$environment_venv/bin/python
if [[ ${OSWORLD_REFRESH_GYM_VENVS:-0} == 1 ]] \
  || [[ ! -x "\$environment_python" ]] \
  || ! "\$environment_python" -c 'import fastapi, nemo_gym'; then
  cd "\$environment_dir"
  uv venv --seed --allow-existing --python $(printf '%q' "$OSWORLD_COMPONENT_PYTHON") "\$environment_venv"
  uv pip install --python "\$environment_python" -r requirements.txt
fi
EOF
)"

"$OSWORLD_COMPONENT_PYTHON" "$setup_helper" \
  --venv "$environment_venv" \
  --command "$environment_setup_command"
"$environment_python" -c 'import fastapi, nemo_gym; print("osworld-environment-server-venv-ready")'
