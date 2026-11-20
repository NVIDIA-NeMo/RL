#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

usage() {
  cat <<EOF
Usage: $0 {sft|checkpoint} [ENV_FILE] [VAL_DATA]

ENV_FILE defaults to $SCRIPT_DIR/.env and must define OSWORLD_GRPO_VAL_DATA.
Checkpoint evaluation also requires OSWORLD_EVAL_CHECKPOINT_DIR.
VAL_DATA overrides OSWORLD_GRPO_VAL_DATA, which is useful for parallel shards.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if (( $# < 1 || $# > 3 )); then
  usage >&2
  exit 2
fi

case "$1" in
  sft|checkpoint)
    export OSWORLD_EVAL_MODE="$1"
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

export OSWORLD_DRIVER_SCRIPT=eval_driver.sh
export OSWORLD_NUM_NODES=1
export OSWORLD_JOB_NAME=osw-v2-eval
export OSWORLD_RUN_PREFIX="osworld-v2-${OSWORLD_EVAL_MODE}-eval"
if (( $# == 3 )); then
  export OSWORLD_GRPO_VAL_DATA_OVERRIDE="$3"
fi

exec "$SCRIPT_DIR/submit.sh" "${2:-$SCRIPT_DIR/.env}"
