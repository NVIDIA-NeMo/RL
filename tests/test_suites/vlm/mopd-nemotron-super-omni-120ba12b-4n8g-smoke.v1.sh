#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)

export EXP_NAME="$(basename "$0" .sh)"
export NUM_NODES=4
export NUM_MINUTES=480

exec "$SCRIPT_DIR/mopd-nemotron-super-omni-120ba12b-10n8g-megatron-tp8ep16cp2-async-gym.v1.sh" "$@"
