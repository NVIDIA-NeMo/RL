#!/bin/bash
set -euo pipefail

: "${OSWORLD_COMPONENT_PYTHON:?}"
: "${OSWORLD_DRIVER_PYTHON:?}"
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
setup_helper="$OSWORLD_GYM_ROOT/nemo_gym/utils/venv_setup.py"
locked_env="$OSWORLD_GYM_ROOT/nemo_gym/server_runtime/supervisor/_locked_env.py"
runtime_checker="$agent_dir/runtime_dependencies.py"
runtime_installer="$agent_dir/install_optional_runtime_deps.sh"

for component_dir in "$agent_dir" "$model_dir" "$environment_dir"; do
  test -r "$component_dir/pyproject.toml"
  test -r "$component_dir/uv.lock"
done
test -r "$setup_helper"
test -r "$locked_env"
test -r "$runtime_checker"
test -x "$runtime_installer"
uv --version

# Components are installed from their committed uv.lock files; Ray must match the
# cluster exactly, so the driver's version is passed as an inherited constraint.
sync_component() {
  local component_dir="$1"
  local venv="$2"
  local ready_check="$3"
  local command
  command="$(cat <<EOF
set -euo pipefail
venv_python=$(printf '%q' "$venv/bin/python")
if [[ ${OSWORLD_REFRESH_GYM_VENVS:-0} == 1 ]] \
  || [[ ! -x "\$venv_python" ]] \
  || ! "\$venv_python" -c $(printf '%q' "$ready_check"); then
  $(printf '%q' "$OSWORLD_DRIVER_PYTHON") $(printf '%q' "$locked_env") \
    --project $(printf '%q' "$component_dir") \
    --venv $(printf '%q' "$venv") \
    --python $(printf '%q' "$OSWORLD_COMPONENT_PYTHON") \
    --constraint $(printf '%q' "ray==$driver_ray_version")
fi
EOF
)"
  "$OSWORLD_COMPONENT_PYTHON" "$setup_helper" --venv "$venv" --command "$command"
}

sync_component "$agent_dir" "$agent_venv" "import ray, sys; assert ray.__version__ == '$driver_ray_version'"
bash "$runtime_installer" "$agent_venv"

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

sync_component "$model_dir" "$model_venv" "import aiohttp, fastapi, nemo_gym"
"$model_python" -c 'import aiohttp, fastapi, nemo_gym; print("osworld-policy-model-venv-ready")'

sync_component "$environment_dir" "$environment_venv" "import fastapi, nemo_gym"
"$environment_python" -c 'import fastapi, nemo_gym; print("osworld-environment-server-venv-ready")'
