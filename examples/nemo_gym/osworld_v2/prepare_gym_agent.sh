#!/bin/bash
set -euo pipefail

: "${OSWORLD_COMPONENT_PYTHON:?}"
: "${OSWORLD_GYM_ROOT:?}"
: "${NEMO_GYM_VENV_DIR:?}"

agent_dir="$OSWORLD_GYM_ROOT/responses_api_agents/osworld_agent"
agent_venv="$NEMO_GYM_VENV_DIR/responses_api_agents/osworld_agent/.venv"
agent_python="$agent_venv/bin/python"
driver_ray_version="$("$OSWORLD_DRIVER_PYTHON" -c 'import ray; print(ray.__version__)')"
setup_helper="$OSWORLD_GYM_ROOT/nemo_gym/cli/_venv_setup.py"
runtime_checker="$agent_dir/runtime_dependencies.py"
runtime_installer="$agent_dir/install_optional_runtime_deps.sh"

test -r "$agent_dir/requirements.txt"
test -r "$agent_dir/overrides.txt"
test -r "$agent_dir/uv.toml"
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
