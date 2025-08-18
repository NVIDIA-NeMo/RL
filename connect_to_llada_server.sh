#!/bin/bash

# Convenience wrapper for the LLaDA server connection helper
# This script calls the actual implementation in xp/llada_api/scripts/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTUAL_SCRIPT="$SCRIPT_DIR/xp/llada_api/scripts/connect_to_llada_server.sh"

if [[ ! -f "$ACTUAL_SCRIPT" ]]; then
    echo "Error: Connection helper script not found at: $ACTUAL_SCRIPT"
    echo "Make sure you're running from the NeMo-RL project root."
    exit 1
fi

# Forward all arguments to the actual script
exec "$ACTUAL_SCRIPT" "$@"