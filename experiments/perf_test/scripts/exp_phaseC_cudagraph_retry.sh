#!/bin/bash
# Phase C — retry of v07_12_external_cuda_graph after wiring use_te_rng_tracker
# (and disabling activation_checkpointing for llama_8b). Pairs each retry with
# a fresh baseline in the same submit batch for noise control.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/cluster_config.sh"
setup_cluster_config "${PARTITION:-batch}"
export_cluster_config

PROJECT_ROOT="${PROJECT_ROOT:-${BASE}/RL-v07-tier1-bench}"

declare -a JOBS=(
    "perf_test/v07_tier1/llama_8b/v07_00_baseline             2 llama8b-phaseCr-base"
    "perf_test/v07_tier1/llama_8b/v07_12_external_cuda_graph  2 llama8b-phaseCr-extgraph"

    "perf_test/v07_tier1/qwen3_30ba3b/v07_00_baseline            4 qwen30ba3b-phaseCr-base"
    "perf_test/v07_tier1/qwen3_30ba3b/v07_12_external_cuda_graph 4 qwen30ba3b-phaseCr-extgraph"

    "perf_test/v07_tier1/qwen3_32b/v07_00_baseline             4 qwen32b-phaseCr-base"
    "perf_test/v07_tier1/qwen3_32b/v07_12_external_cuda_graph  4 qwen32b-phaseCr-extgraph"
)

for job in "${JOBS[@]}"; do
    read -r config num_nodes suffix <<<"$job"
    submit_variant "$PROJECT_ROOT" "$config" "$num_nodes" "nrl-phaseCr-${suffix}"
done

echo ""
echo "[MONITOR] squeue -u \$USER -o '%.18i %.30j %.8T %.10M %R'"
