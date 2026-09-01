#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"

# One optimizer step from the SFT base checkpoint. This run has its own name
# and checkpoint namespace and is never eligible for the formal resume chain.
export MOLT_RUN_NAME=osworld361-molt-jianh-aligned-v1-b8k8-smoke
export MOLT_MAX_STEPS=1
export CHECKPOINTING_ENABLED=false
export MOLT_AUTO_RESUME_CHAIN=false
export SBATCH_TIME="${SBATCH_TIME:-04:00:00}"

exec bash "${ROOT}/examples/nemo_gym/submit_osworld_molt_nemogym_parity64_b8k8.sh"
