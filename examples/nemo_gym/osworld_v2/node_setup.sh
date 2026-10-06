#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/runtime_env.sh"

test -f "$OSWORLD_RL_ROOT/examples/nemo_gym/launch_osworld_v2_cc.py"
test -f "$OSWORLD_GYM_ROOT/resources_servers/osworld/app.py"
test -f "$OSWORLD_GYM_ROOT/responses_api_agents/nemotron_osworld/cc_app.py"
test -x "$OSWORLD_GYM_ROOT/responses_api_agents/osworld_agent/install_optional_runtime_deps.sh"

case "$OSWORLD_COMPONENT_PYTHON" in
  /root/*|/opt/*) ;;
  *)
    echo "Refusing non-container component Python: $OSWORLD_COMPONENT_PYTHON" >&2
    exit 2
    ;;
esac

"$OSWORLD_COMPONENT_PYTHON" -c \
  'import sys; assert sys.version_info[:3] == (3, 13, 14), sys.version'

"$OSWORLD_DRIVER_PYTHON" - <<'PY'
from pathlib import Path

import nemo_gym
import nemo_rl
import ray
import transfer_queue

expected = (
    (nemo_rl, Path("/opt/nemo-rl")),
    (nemo_gym, Path("/opt/nemo-rl/3rdparty/Gym-workspace/Gym")),
)
for module, root in expected:
    source = Path(module.__file__).resolve()
    if not source.is_relative_to(root):
        raise RuntimeError(f"{module.__name__} imported from unexpected path: {source}")

if ray.__version__ != "2.56.1":
    raise RuntimeError(f"Expected Ray 2.56.1, found {ray.__version__}")

print(
    "osworld-v2-node-preflight-ok",
    Path(nemo_rl.__file__).resolve(),
    Path(nemo_gym.__file__).resolve(),
    Path(transfer_queue.__file__).resolve(),
)
PY

cd "$OSWORLD_RL_ROOT"
"$OSWORLD_DRIVER_PYTHON" \
  examples/nemo_gym/launch_osworld_v2_cc.py \
  --molt-b8k8-checkpoint \
  --validate-only \
  "grpo.max_num_steps=${OSWORLD_GRPO_MAX_STEPS:-300}"
