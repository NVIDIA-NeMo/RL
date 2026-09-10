#!/usr/bin/env bash
# Submit the 32-node step-120 combined SA-V + CAPRL run with R3 enabled.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export R3_ENABLED=true
exec "${script_dir}/async_grpo_super35_sav_caprl_step120_32n.sh" "$@"
