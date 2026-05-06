#!/bin/bash
# Phase A — MLM PR #3384 fused_residual_rmsnorm paired benchmark.
# Re-runs v07 baselines alongside v07_11 variants in the same SLURM batch
# so within-batch noise (~±2%) is the only confounder. 8 jobs total.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/cluster_config.sh"
setup_cluster_config "${PARTITION:-batch}"
export_cluster_config

PROJECT_ROOT="${PROJECT_ROOT:-${BASE}/RL-v07-tier1-bench}"

declare -a JOBS=(
    "perf_test/v07_tier1/llama_8b/v07_00_baseline                2 llama8b-phaseA-base"
    "perf_test/v07_tier1/llama_8b/v07_11_fused_resid_rmsnorm     2 llama8b-phaseA-fusedresid"

    "perf_test/v07_tier1/qwen3_30ba3b/v07_00_baseline            4 qwen30ba3b-phaseA-base"
    "perf_test/v07_tier1/qwen3_30ba3b/v07_11_fused_resid_rmsnorm 4 qwen30ba3b-phaseA-fusedresid"

    "perf_test/v07_tier1/qwen3_32b/v07_00_baseline               4 qwen32b-phaseA-base"
    "perf_test/v07_tier1/qwen3_32b/v07_11_fused_resid_rmsnorm    4 qwen32b-phaseA-fusedresid"

    "perf_test/v07_tier1/qwen3_235b/v07_00_baseline              16 qwen235b-phaseA-base"
    "perf_test/v07_tier1/qwen3_235b/v07_11_fused_resid_rmsnorm   16 qwen235b-phaseA-fusedresid"
)

FILTER="${1:-}"

for job in "${JOBS[@]}"; do
    read -r config num_nodes suffix <<<"$job"
    if [[ -n "$FILTER" ]] && [[ "$config" != *"$FILTER"* ]] && [[ "$suffix" != *"$FILTER"* ]]; then
        continue
    fi
    submit_variant "$PROJECT_ROOT" "$config" "$num_nodes" "nrl-phaseA-${suffix}"
done

echo ""
echo "[MONITOR] squeue -u \$USER -o '%.18i %.30j %.8T %.10M %R'"
