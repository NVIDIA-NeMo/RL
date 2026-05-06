#!/bin/bash
# Lyris GB200 launcher wrapper for v0.7 Tier 1 perf_test variants.
#
# Lyris differences vs OCI-HSG (handled here):
#   - Account:   coreai_dlalgo_llm
#   - Partition: gb200
#   - GRES flag: NOT passed (Lyris rejects --gres=gpu:N)
#   - Base dir:  /lustre/fsw/coreai_dlalgo_llm/users/sna/Nemo-RL_feature_test
#
# Usage (from a Lyris worktree root):
#   bash experiments/perf_test/scripts/exp_v07_tier1_lyris.sh           # submit all
#   bash experiments/perf_test/scripts/exp_v07_tier1_lyris.sh llama     # filter

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export BASE="${BASE:-/lustre/fsw/coreai_dlalgo_llm/users/sna/Nemo-RL_feature_test}"
export CONTAINER="${CONTAINER:-${BASE}/RL/nemo_rl_nightly.sqsh}"
# Shared HF cache lives one level above BASE, alongside other Lyris workspaces.
export HF_HOME="${HF_HOME:-/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home}"
export ACCOUNT="${ACCOUNT:-coreai_dlalgo_llm}"
export PARTITION="${PARTITION:-gb200}"
# Empty GRES_FLAG: cluster_config.sh respects existing (even empty) value via +x check.
export GRES_FLAG="${GRES_FLAG-}"
export PROJECT_ROOT="${PROJECT_ROOT:-${BASE}/RL-v07-tier1-bench}"

exec bash "${SCRIPT_DIR}/exp_v07_tier1.sh" "$@"
