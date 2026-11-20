#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/runtime_env.sh"
"$SCRIPT_DIR/prepare_gym_agent.sh"
cd "$OSWORLD_RL_ROOT"

case "${OSWORLD_EVAL_MODE:-}" in
  sft)
    CONFIG="${OSWORLD_EVAL_CONFIG:-examples/nemo_gym/grpo_nemotron_omni_30ba3b_osworld_v2_inference_v1_parity.yaml}"
    ;;
  checkpoint)
    : "${OSWORLD_EVAL_CHECKPOINT_DIR:?}"
    CONFIG="${OSWORLD_EVAL_CONFIG:-examples/nemo_gym/grpo_nemotron_omni_30ba3b_osworld_v2_checkpoint_inference_v1_parity.yaml}"
    ;;
  *)
    echo "OSWORLD_EVAL_MODE must be sft or checkpoint" >&2
    exit 2
    ;;
esac

exec "$OSWORLD_DRIVER_PYTHON" \
  examples/nemo_gym/run_grpo_nemo_gym.py \
  --config "$CONFIG" \
  "logger.log_dir=$OSWORLD_RUN_ROOT/exp_001" \
  "policy.generation.vllm_cfg.reasoning_parser_plugin=$OSWORLD_RL_ROOT/nemo_rl/models/generation/vllm/reasoning_parsers/nano_v3_reasoning_parser.py"
