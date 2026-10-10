#!/bin/bash
set -euo pipefail

: "${OSWORLD_COMPONENT_PYTHON:?}"
: "${OSWORLD_GYM_ROOT:?}"
: "${NEMO_GYM_VENV_DIR:?}"

agent_dir="$OSWORLD_GYM_ROOT/responses_api_agents/osworld_agent"
agent_venv="$NEMO_GYM_VENV_DIR/responses_api_agents/osworld_agent/.venv"
agent_python="$agent_venv/bin/python"
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
agent_python=\$agent_venv/bin/python
if [[ ${OSWORLD_REFRESH_GYM_VENVS:-0} == 1 ]] \
  || [[ ! -x "\$agent_python" ]] \
  || ! "\$agent_python" -c 'import ray; version=tuple(int(part) for part in ray.__version__.split(".")[:2]); assert version >= (2, 58), ray.__version__'; then
  cd "\$agent_dir"
  uv venv --seed --allow-existing --python $(printf '%q' "$OSWORLD_COMPONENT_PYTHON") "\$agent_venv"
  source "\$agent_venv/bin/activate"
  uv pip install --config-file "\$agent_dir/uv.toml" --override "\$agent_dir/overrides.txt" -r "\$agent_dir/requirements.txt"
fi
bash $(printf '%q' "$runtime_installer") "\$agent_venv"
EOF
)"

"$OSWORLD_COMPONENT_PYTHON" "$setup_helper" \
  --venv "$agent_venv" \
  --command "$setup_command"

"$agent_python" - <<'PY'
import ray

version = tuple(int(part) for part in ray.__version__.split(".")[:2])
if version < (2, 58):
    raise RuntimeError(f"Expected Ray >=2.58 in the OSWorld agent venv, found {ray.__version__}")
print("osworld-agent-ray-ready", ray.__version__)
PY
"$agent_python" "$runtime_checker" check
