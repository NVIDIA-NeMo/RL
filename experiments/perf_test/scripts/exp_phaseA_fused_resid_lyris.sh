#!/bin/bash
# Lyris GB200 launcher wrapper for Phase A fused_residual_rmsnorm variants.
# See exp_v07_tier1_lyris.sh for Lyris-specific env var rationale.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export BASE="${BASE:-/lustre/fsw/coreai_dlalgo_llm/users/sna/Nemo-RL_feature_test}"
export CONTAINER="${CONTAINER:-${BASE}/RL/nemo_rl_nightly.sqsh}"
export HF_HOME="${HF_HOME:-/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home}"
export ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}"
export PARTITION="${PARTITION:-gb200}"
export GRES_FLAG="${GRES_FLAG-}"
export PROJECT_ROOT="${PROJECT_ROOT:-${BASE}/RL-v07-tier1-bench}"

exec bash "${SCRIPT_DIR}/exp_phaseA_fused_resid.sh" "$@"
