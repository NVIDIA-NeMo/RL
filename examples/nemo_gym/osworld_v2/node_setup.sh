#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/runtime_env.sh"

test -f "$OSWORLD_RL_ROOT/examples/nemo_gym/launch_osworld_v2_cc.py"
test -f "$OSWORLD_RL_ROOT/examples/run_grpo.py"
test -f "nemo_rl/models/generation/vllm/reasoning_parsers/nano_v3_reasoning_parser.py"
test -f "$OSWORLD_GYM_ROOT/resources_servers/osworld/app.py"
test -f "$OSWORLD_GYM_ROOT/responses_api_agents/osworld_agent/app.py"
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
resolved_uv_python="$(uv python find 3.13.14)"
"$resolved_uv_python" -c \
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

ray_version = tuple(int(part) for part in ray.__version__.split(".")[:2])
if ray_version < (2, 58):
    raise RuntimeError(f"Expected Ray >=2.58, found {ray.__version__}")

print(
    "osworld-v2-node-preflight-ok",
    Path(nemo_rl.__file__).resolve(),
    Path(nemo_gym.__file__).resolve(),
    Path(transfer_queue.__file__).resolve(),
)
PY

"$OSWORLD_DRIVER_PYTHON" - <<'PY'
from vllm.parser.parser_manager import ParserManager
from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager

plugin_path = (
    "nemo_rl/models/generation/vllm/reasoning_parsers/"
    "nano_v3_reasoning_parser.py"
)
ReasoningParserManager.import_reasoning_parser(plugin_path)
ParserManager.get_reasoning_parser("nano_v3")
print("osworld-v2-reasoning-parser-preflight-ok", plugin_path)
PY

cd "$OSWORLD_RL_ROOT"
if [[ -n "${OSWORLD_EVAL_MODE:-}" ]]; then
  case "$OSWORLD_EVAL_MODE" in
    sft)
      eval_config=examples/nemo_gym/grpo_nemotron_omni_30ba3b_osworld_v2_inference_v1_parity.yaml
      ;;
    checkpoint)
      eval_config=examples/nemo_gym/grpo_nemotron_omni_30ba3b_osworld_v2_checkpoint_inference_v1_parity.yaml
      ;;
    *)
      echo "OSWORLD_EVAL_MODE must be sft or checkpoint" >&2
      exit 2
      ;;
  esac
  "$OSWORLD_DRIVER_PYTHON" - "$eval_config" <<'PY'
import sys

from omegaconf import OmegaConf

from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

register_omegaconf_resolvers()
config_path = sys.argv[1]
resolved = OmegaConf.to_container(load_config(config_path), resolve=True)
if not isinstance(resolved, dict):
    raise TypeError("OSWorld evaluation recipe did not resolve to a mapping")
MasterConfig(**resolved)
print("osworld-v2-eval-config-ok", config_path)
PY
else
  "$OSWORLD_DRIVER_PYTHON" \
    examples/nemo_gym/launch_osworld_v2_cc.py \
    --molt-b8k8-checkpoint \
    --validate-only \
    "grpo.max_num_steps=${OSWORLD_GRPO_MAX_STEPS:-300}"
fi
