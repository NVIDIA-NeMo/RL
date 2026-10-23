#!/bin/bash
export EXP_NAME="$(basename "$0" .sh)"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$SCRIPT_DIR/ppo-qwen2.5-1.5b-gsm8k-2n8g-megatron-valuetp2sp-dynbatch-noncolocated-async-single-controller-streaming32.sh" "$@"
