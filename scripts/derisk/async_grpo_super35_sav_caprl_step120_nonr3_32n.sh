#!/usr/bin/env bash
# Submit the matched 32-node step-120 combined SA-V + CAPRL non-R3 control.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export R3_ENABLED=false
exec "${script_dir}/async_grpo_super35_sav_caprl_step120_32n.sh" "$@"
