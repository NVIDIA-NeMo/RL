#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
export EXP_NAME=$(basename "$0" .sh)
source "$SCRIPT_DIR/vlm_sft-nemotron-omni-30ba3b-clevr-1n8g-megatron-tp8ep8-energon.v1.sh" sft.max_num_steps=50 "$@"
