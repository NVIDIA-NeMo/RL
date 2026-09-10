#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
export EXP_NAME=$(basename "$0" .sh)
source "$SCRIPT_DIR/vlm_sft-qwen2.5-vl-3b-instruct-clevr-1n2g-megatrontp1-energon.v1.sh" "$@"
